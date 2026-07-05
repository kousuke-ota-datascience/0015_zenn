"""設定ファイルとパス解決に関する共通ユーティリティ。

因果探索側と因果推論側は、どちらも YAML を読み込み、CLI 引数で上書きし、
解決済み設定を artifacts に保存する。このモジュールはその共通処理を
一箇所に集約し、個別パイプライン側の設定クラスには「どの項目を持つか」
というドメイン固有の責務だけを残す。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def ensure_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """YAML から得た値が mapping であることを検証する。

    Args:
        value: 検証対象の値。
        field_name: 例外メッセージに表示する設定項目名。

    Returns:
        mapping として扱える検証済みの値。

    Raises:
        ValueError: ``value`` が mapping でない場合。
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def ensure_tuple(value: Any, field_name: str) -> tuple[Any, ...]:
    """YAML のリスト相当値を tuple に正規化する。

    Args:
        value: ``list``、``tuple``、または ``None`` を想定する値。
        field_name: 例外メッセージに表示する設定項目名。

    Returns:
        tuple に正規化した値。``None`` は空 tuple として扱う。

    Raises:
        ValueError: ``value`` が list/tuple/None のいずれでもない場合。
    """
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be a list")
    return tuple(value)


def clean_none_values(value: dict[str, Any]) -> dict[str, Any]:
    """値が ``None`` のキーを落とした辞書を返す。

    Args:
        value: YAML に戻す前の辞書。

    Returns:
        ``None`` 値を含まない新しい辞書。
    """
    return {key: child for key, child in value.items() if child is not None}


def validate_choice(value: str, choices: tuple[str, ...], field_name: str) -> str:
    """文字列が許可済み選択肢に含まれることを検証する。

    Args:
        value: 検証対象の文字列。
        choices: 許可する文字列の集合。
        field_name: 例外メッセージに表示する設定項目名。

    Returns:
        検証済みの文字列。

    Raises:
        ValueError: ``value`` が ``choices`` に含まれない場合。
    """
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {choices}: {value}")
    return value


def load_yaml_mapping(path: Path | str) -> dict[str, Any]:
    """YAML ファイルを読み込み、root が mapping であることを検証する。

    Args:
        path: 読み込む YAML ファイル。

    Returns:
        YAML root の辞書。空ファイルは空辞書として扱う。

    Raises:
        ValueError: YAML root が mapping でない場合。
    """
    yaml_path = Path(path)
    with yaml_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")
    return dict(data)


def resolve_project_path(path: Path, project_root: Path) -> Path:
    """絶対パスまたはプロジェクト相対パスを絶対パスへ解決する。

    Args:
        path: 解決対象のパス。
        project_root: repository root。

    Returns:
        ``path`` が相対パスなら ``project_root`` からの絶対パス、絶対パスなら
        そのままの値。
    """
    return path if path.is_absolute() else project_root / path


def write_yaml_snapshots(
    *,
    output_dir: Path,
    snapshots: Mapping[str, Mapping[str, Any]],
) -> None:
    """複数の YAML スナップショットを artifacts 配下へ保存する。

    Args:
        output_dir: 保存先ディレクトリ。
        snapshots: ファイル名から YAML 化する mapping への対応。

    Notes:
        ここで保存する設定は再現性のための生成物であり、ソースコードではない。
        呼び出し側は ``output_dir`` を artifacts 配下に解決してから渡す。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in snapshots.items():
        (output_dir / filename).write_text(
            yaml.safe_dump(
                dict(data),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
