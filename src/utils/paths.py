"""Repository-relative path helpers."""

from __future__ import annotations

from pathlib import Path


def project_root_from_config(config_path: Path) -> Path:
    """Return the repository root for ``configs/experiment.yaml``."""
    resolved = config_path.expanduser().resolve()
    if resolved.parent.name != "configs":
        raise ValueError(f"Config must be inside a configs directory: {resolved}")
    return resolved.parent.parent


def resolve_repository_path(root: Path, configured_path: str, *, must_exist: bool) -> Path:
    """Resolve a configured relative path without allowing root escape."""
    candidate = Path(configured_path)
    if candidate.is_absolute():
        raise ValueError(f"Configured paths must be repository-relative: {configured_path}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Configured path escapes repository root: {configured_path}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Configured path does not exist: {configured_path} (resolved to {resolved})"
        )
    return resolved
