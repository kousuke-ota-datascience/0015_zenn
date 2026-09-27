"""OpenAI Agents SDK adapter for Workflow 00.

Creator and Reviewer are deliberately separated at the session and tool layer.
The Creator may operate on the repository.  Every Review cycle gets a fresh
Reviewer session whose input is rebuilt from a strict whitelist of frozen
canonical facts and review rules.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
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

DEFAULT_MODEL = "gpt-5.6-sol"


class AgentsAPIUnavailable(RuntimeError):
    """Raised when the optional Agents SDK runtime is unavailable."""


@dataclass(frozen=True)
class AgentSession:
    role: str
    session_id: str
    agent: Any
    session: Any


@dataclass(frozen=True)
class AgentRunResult:
    output: str


class AgentsRuntime(Protocol):
    def create_creator_session(self, entry_id: str) -> AgentSession: ...
    def create_reviewer_session(self, entry_id: str, review_seq: int) -> AgentSession: ...
    def run_creator(self, session: AgentSession, prompt: str) -> AgentRunResult: ...
    def run_reviewer(
        self,
        session: AgentSession,
        context: Mapping[str, Any],
        prompt: str,
    ) -> AgentRunResult: ...


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
    """Build the only context that may cross the Reviewer boundary.

    Intentionally excluded:
    - Creator session/history
    - Creator rationale or intermediate notes
    - prior agent messages
    - mutable working-tree copies of the canonical artifacts
    """
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


class OpenAIAgentsAPI:
    """Production Agents SDK runtime.

    The dependency is imported lazily so the deterministic fixture suite does
    not require openai-agents. Runtime installation:
        pip install openai-agents
    """

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model or os.environ.get("WORKFLOW_AGENT_MODEL", DEFAULT_MODEL)

    @staticmethod
    def _sdk() -> tuple[Any, Any, Any, Any, Any]:
        try:
            from agents import Agent, Runner, ShellTool, SQLiteSession, WebSearchTool
        except ImportError as exc:
            raise AgentsAPIUnavailable(
                "Workflow 00 runtime requires the optional 'openai-agents' package"
            ) from exc
        return Agent, Runner, ShellTool, SQLiteSession, WebSearchTool

    @staticmethod
    async def _shell_executor(request: Any) -> str:
        commands = list(request.data.action.commands)
        timeout_ms = request.data.action.timeout_ms
        timeout = None if timeout_ms in (None, 0) else timeout_ms / 1000
        max_output = request.data.action.max_output_length
        outputs: list[str] = []

        for command in commands:
            proc = subprocess.run(
                command,
                cwd=ARTICLE_ROOT,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            text = proc.stdout or ""
            if max_output is not None:
                text = text[:max_output]
            outputs.append(
                json.dumps(
                    {
                        "command": command,
                        "exit_code": proc.returncode,
                        "output": text,
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(outputs)

    def create_creator_session(self, entry_id: str) -> AgentSession:
        Agent, _, ShellTool, SQLiteSession, WebSearchTool = self._sdk()
        session_id = f"workflow00:{entry_id}:creator:{uuid4().hex}"
        session = SQLiteSession(session_id)
        workflow_10 = _read_text(WORKFLOW_10)
        instructions = f"""You are the Workflow 10 Creator for Entry_ID {entry_id}.
Follow the repository's current Workflow 10 exactly:
{workflow_10}

Boundary rules:
- Work only as Workflow 10: research, generate/correct canonical 00/10/20,
  deterministic validation, canonical commit/push, and required Workflow 90 events.
- Do not execute Workflow 20 and do not write Review JSON.
- Use current external evidence and current schema/taxonomy/coding rules.
- Do not treat legacy artifacts or previous Review conclusions as semantic truth.
- Return control to Workflow 00 once the canonical chain is review-ready.
"""
        agent = Agent(
            name=f"Workflow10-Creator-{entry_id}",
            model=self.model,
            instructions=instructions,
            tools=[
                ShellTool(
                    executor=self._shell_executor,
                    environment={"type": "local"},
                ),
                WebSearchTool(),
            ],
        )
        return AgentSession("creator", session_id, agent, session)

    def create_reviewer_session(self, entry_id: str, review_seq: int) -> AgentSession:
        Agent, _, _, SQLiteSession, WebSearchTool = self._sdk()
        session_id = f"workflow00:{entry_id}:review:{review_seq}:{uuid4().hex}"
        session = SQLiteSession(session_id)
        agent = Agent(
            name=f"Workflow20-Reviewer-{entry_id}-{review_seq}",
            model=self.model,
            instructions="""You are an independent Workflow 20 semantic Reviewer.
You have no access to the Creator session and must not infer or request its rationale.
Use only the explicit whitelist context supplied in this run plus external evidence
obtained with web search. You have no shell/file-write tool and must never modify
canonical artifacts.

Return one JSON object only:
{"reviews":{"00":{...},"10":{...},"20":{...}}}

Each body must satisfy the supplied Review schemas and Workflow 20 semantics.
Do not include writer-managed fields: schema_version, entry_id, artifact,
review_seq, target, verdict. The deterministic review_writer derives/verifies them.
""",
            tools=[WebSearchTool()],
        )
        return AgentSession("reviewer", session_id, agent, session)

    def run_creator(self, session: AgentSession, prompt: str) -> AgentRunResult:
        if session.role != "creator":
            raise ValueError("run_creator requires a creator session")
        _, Runner, _, _, _ = self._sdk()
        result = Runner.run_sync(
            session.agent,
            prompt,
            session=session.session,
        )
        return AgentRunResult(str(result.final_output))

    def run_reviewer(
        self,
        session: AgentSession,
        context: Mapping[str, Any],
        prompt: str,
    ) -> AgentRunResult:
        if session.role != "reviewer":
            raise ValueError("run_reviewer requires a reviewer session")
        _, Runner, _, _, _ = self._sdk()
        reviewer_input = json.dumps(
            {
                "whitelist_context": context,
                "task": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        result = Runner.run_sync(
            session.agent,
            reviewer_input,
            session=session.session,
        )
        return AgentRunResult(str(result.final_output))
