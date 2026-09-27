"""Codex CLI runtime adapter for Workflow 00.

Creator and Reviewer are isolated at the Codex thread/process boundary.

- Creator uses one persistent Codex thread and resumes it across correction turns.
- Every Reviewer cycle launches a fresh ephemeral Codex process.
- Reviewer receives only a freeze-bound whitelist context, live web search, no shell
  tool, and a read-only sandbox in a temporary non-repository working directory.
- Authentication is forced to ChatGPT login so this runtime uses Codex/ChatGPT
  entitlement rather than an OpenAI API key.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Protocol
from uuid import uuid4

from src.status_management.git_state import repository_root

ARTICLE_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_ROOT = ARTICLE_ROOT / "docs/10_each_lore/0000_tutorial"
WORKFLOW_10 = TUTORIAL_ROOT / "0000_workflow_10_each_lore_analysis.md"
WORKFLOW_20 = TUTORIAL_ROOT / "0000_workflow_20_review.md"
SCHEMA_ROOT = TUTORIAL_ROOT / "schemas"
TAXONOMY = ARTICLE_ROOT / "docs/00_research_overview/taxonomy_catalog.json"
CODING_RULES = ARTICLE_ROOT / "docs/00_research_overview/30_urban_legend_analysis_coding_rules.md"


class CodexCLIUnavailable(RuntimeError):
    """Raised when the Codex CLI executable is unavailable."""


class CodexCLIAuthError(RuntimeError):
    """Raised when Codex is not authenticated with ChatGPT."""


class CodexCLIExecutionError(RuntimeError):
    """Raised when codex exec fails or emits malformed output."""


@dataclass
class RuntimeSession:
    role: str
    session_id: str
    thread_id: str | None = None


@dataclass(frozen=True)
class RuntimeResult:
    output: str


class OrchestrationRuntime(Protocol):
    def create_creator_session(self, entry_id: str) -> RuntimeSession: ...
    def create_reviewer_session(self, entry_id: str, review_seq: int) -> RuntimeSession: ...
    def run_creator(self, session: RuntimeSession, prompt: str) -> RuntimeResult: ...
    def run_reviewer(
        self,
        session: RuntimeSession,
        context: Mapping[str, Any],
        prompt: str,
    ) -> RuntimeResult: ...


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "git " + " ".join(args) + " failed: " + proc.stderr.strip()
        )
    return proc.stdout


def _frozen_file(repo: Path, commit_sha: str, blob_sha: str, path: str) -> str:
    """Read exactly the blob that was frozen by review_writer.prepare."""
    actual = _run_git(repo, "rev-parse", f"{commit_sha}:{path}").strip()
    if actual != blob_sha:
        raise RuntimeError(
            f"frozen target mismatch for {path}: expected blob={blob_sha}, actual={actual}"
        )
    return _run_git(repo, "show", f"{commit_sha}:{path}")


def build_reviewer_context(
    entry_id: str,
    cycle: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only context that may cross the Reviewer boundary."""
    repo = repository_root()
    targets = cycle.get("targets")
    review_seq = cycle.get("review_seq")
    if not isinstance(targets, Mapping) or set(targets) != {"00", "10", "20"}:
        raise ValueError("cycle.targets must contain exactly 00, 10, 20")
    if not isinstance(review_seq, int) or review_seq < 1:
        raise ValueError("cycle.review_seq must be a positive integer")

    frozen: dict[str, Any] = {}
    for artifact in ("00", "10", "20"):
        raw = targets[artifact]
        if not isinstance(raw, Mapping):
            raise ValueError(f"cycle.targets.{artifact} must be an object")
        artifact_path = raw.get("artifact_path")
        commit_sha = raw.get("commit_sha")
        blob_sha = raw.get("blob_sha")
        if not all(isinstance(x, str) and x for x in (artifact_path, commit_sha, blob_sha)):
            raise ValueError(f"cycle.targets.{artifact} is incomplete")
        payload = _frozen_file(repo, commit_sha, blob_sha, artifact_path)
        frozen[artifact] = {
            "target": {
                "artifact_path": artifact_path,
                "commit_sha": commit_sha,
                "blob_sha": blob_sha,
            },
            "canonical_json": json.loads(payload),
        }

    schemas = {
        path.name: json.loads(_read_text(path))
        for path in sorted(SCHEMA_ROOT.glob("*.json"))
    }
    return {
        "entry_id": entry_id,
        "review_seq": review_seq,
        "context_contract": {
            "source": "reviewer_whitelist",
            "creator_conversation_history": "FORBIDDEN",
            "creator_rationale": "FORBIDDEN",
            "creator_intermediate_notes": "FORBIDDEN",
            "mutable_working_tree_artifacts": "FORBIDDEN",
            "instruction": (
                "Reconstruct the Review independently from the frozen canonical target, "
                "Workflow 20, schemas, taxonomy/coding rules, and external evidence only."
            ),
        },
        "workflow_20": _read_text(WORKFLOW_20),
        "frozen_canonical": frozen,
        "schemas": schemas,
        "taxonomy_catalog": json.loads(_read_text(TAXONOMY)),
        "coding_rules": _read_text(CODING_RULES),
    }


def _parse_codex_jsonl(stdout: str) -> tuple[str, str]:
    """Extract the thread id and final agent message from codex exec --json."""
    thread_id: str | None = None
    messages: list[str] = []

    for lineno, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexCLIExecutionError(
                f"codex --json emitted invalid JSONL at line {lineno}: {exc}"
            ) from exc

        if event.get("type") == "thread.started":
            candidate = event.get("thread_id")
            if isinstance(candidate, str) and candidate:
                thread_id = candidate

        if event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    messages.append(text)

    if not thread_id:
        raise CodexCLIExecutionError("codex --json did not emit thread.started")
    if not messages:
        raise CodexCLIExecutionError("codex --json did not emit a final agent_message")
    return thread_id, messages[-1]


class CodexCLIRuntime:
    """Production Workflow 00 runtime backed by the locally authenticated Codex CLI."""

    def __init__(
        self,
        *,
        binary: str | None = None,
        model: str | None = None,
    ) -> None:
        configured = binary or os.environ.get("WORKFLOW_CODEX_BIN") or "codex"
        resolved = shutil.which(configured)
        if resolved is None:
            raise CodexCLIUnavailable(
                "Codex CLI is required. Install Codex, run 'codex login', "
                "then verify with 'codex login status'."
            )
        self.binary = resolved
        self.model = model or os.environ.get("WORKFLOW_CODEX_MODEL")
        self._auth_checked = False

    def _ensure_chatgpt_login(self) -> None:
        if self._auth_checked:
            return
        proc = subprocess.run(
            [self.binary, "login", "status"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        status = "\n".join(x for x in (proc.stdout.strip(), proc.stderr.strip()) if x)
        if proc.returncode != 0 or "Logged in using ChatGPT" not in status:
            raise CodexCLIAuthError(
                "Workflow 00 requires Codex CLI authenticated with ChatGPT "
                "(not an OpenAI API key). Run 'codex login' and then confirm "
                "'codex login status' prints 'Logged in using ChatGPT'. "
                f"Current status: {status or 'unknown'}"
            )
        self._auth_checked = True

    def _common_exec_args(
        self,
        *,
        sandbox_mode: str,
        ephemeral: bool = False,
        skip_git_repo_check: bool = False,
        shell_enabled: bool = True,
        workspace_network: bool = False,
    ) -> list[str]:
        args = [
            self.binary,
            "exec",
            "--json",
            "-c",
            'forced_login_method="chatgpt"',
            "-c",
            'web_search="live"',
            "-c",
            f'sandbox_mode="{sandbox_mode}"',
        ]
        if not shell_enabled:
            args += ["-c", "features.shell_tool=false"]
        if workspace_network:
            args += ["-c", "sandbox_workspace_write.network_access=true"]
        if self.model:
            args += ["--model", self.model]
        if ephemeral:
            args.append("--ephemeral")
        if skip_git_repo_check:
            args.append("--skip-git-repo-check")
        return args

    def _invoke(
        self,
        args: list[str],
        prompt: str,
        *,
        cwd: Path,
    ) -> tuple[str, str]:
        proc = subprocess.run(
            [*args, "-"],
            cwd=cwd,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise CodexCLIExecutionError(
                f"codex exec failed with exit code {proc.returncode}: {detail}"
            )
        return _parse_codex_jsonl(proc.stdout)

    def create_creator_session(self, entry_id: str) -> RuntimeSession:
        self._ensure_chatgpt_login()
        return RuntimeSession(
            role="creator",
            session_id=f"workflow00:{entry_id}:creator:{uuid4().hex}",
        )

    def create_reviewer_session(self, entry_id: str, review_seq: int) -> RuntimeSession:
        self._ensure_chatgpt_login()
        return RuntimeSession(
            role="reviewer",
            session_id=f"workflow00:{entry_id}:review:{review_seq}:{uuid4().hex}",
        )

    def run_creator(self, session: RuntimeSession, prompt: str) -> RuntimeResult:
        if session.role != "creator":
            raise ValueError("run_creator requires a creator session")

        args = self._common_exec_args(
            sandbox_mode="workspace-write",
            workspace_network=True,
        )

        if session.thread_id is None:
            workflow_10 = _read_text(WORKFLOW_10)
            creator_prompt = f"""You are the Workflow 10 Creator.

Follow the repository's current Workflow 10 exactly:
{workflow_10}

Boundary rules:
- Work only as Workflow 10: research, generate/correct canonical 00/10/20,
  deterministic validation, canonical commit/push, and required Workflow 90 events.
- Do not execute Workflow 20 and do not write Review JSON.
- Use current external evidence and current schema/taxonomy/coding rules.
- Do not treat legacy artifacts or previous Review conclusions as semantic truth.
- Return control to Workflow 00 once the canonical chain is review-ready.

Current Workflow 00 task:
{prompt}
"""
            thread_id, output = self._invoke(args, creator_prompt, cwd=ARTICLE_ROOT)
            session.thread_id = thread_id
            return RuntimeResult(output)

        resume_args = [*args, "resume", session.thread_id]
        thread_id, output = self._invoke(resume_args, prompt, cwd=ARTICLE_ROOT)
        if thread_id != session.thread_id:
            raise CodexCLIExecutionError(
                "Creator resume returned a different thread id: "
                f"expected={session.thread_id}, actual={thread_id}"
            )
        return RuntimeResult(output)

    def run_reviewer(
        self,
        session: RuntimeSession,
        context: Mapping[str, Any],
        prompt: str,
    ) -> RuntimeResult:
        if session.role != "reviewer":
            raise ValueError("run_reviewer requires a reviewer session")
        if session.thread_id is not None:
            raise ValueError("Reviewer sessions must never be resumed")

        reviewer_prompt = f"""You are an independent Workflow 20 semantic Reviewer.

Isolation contract:
- You have no access to the Creator conversation and must not infer or request it.
- Use only the whitelist context embedded below plus live web search.
- Local shell execution is disabled for this run.
- Do not read the repository working tree and do not modify canonical artifacts.
- Reconstruct the Review independently from the frozen target.

Return one JSON object only:
{{"reviews":{{"00":{{...}},"10":{{...}},"20":{{...}}}}}}

Each body must satisfy the supplied Review schemas and Workflow 20 semantics.
Do not include writer-managed fields: schema_version, entry_id, artifact,
review_seq, target, verdict. The deterministic review_writer derives/verifies them.

Whitelist context:
{json.dumps(context, ensure_ascii=False, sort_keys=True)}

Task:
{prompt}
"""

        args = self._common_exec_args(
            sandbox_mode="read-only",
            ephemeral=True,
            skip_git_repo_check=True,
            shell_enabled=False,
        )
        with tempfile.TemporaryDirectory(prefix="workflow00-review-") as tmp:
            thread_id, output = self._invoke(args, reviewer_prompt, cwd=Path(tmp))
        session.thread_id = thread_id
        return RuntimeResult(output)
