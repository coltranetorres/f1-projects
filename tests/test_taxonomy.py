# tests/test_taxonomy.py
import pytest
from pydantic import ValidationError

from harness.evals.taxonomy import DEFAULT_PATH, load_taxonomy


def test_shipped_taxonomy_loads_and_starts_empty():
    assert load_taxonomy(DEFAULT_PATH).failure_modes == []


def test_valid_entry_loads(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("""
failure_modes:
  - id: FM-01
    name: Wrong driver code passed to explain_driver
    suite: tool_use
    layer: tool_arguments
    tool: explain_driver
    evidence: [abc123]
""")
    fm = load_taxonomy(p).failure_modes[0]
    assert fm.id == "FM-01" and fm.status == "observed" and fm.secondary_suite is None


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "t.yaml"
    entry = "  - {id: FM-01, name: x, suite: scope, layer: scope}\n"
    p.write_text("failure_modes:\n" + entry + entry)
    with pytest.raises(ValueError, match="duplicate"):
        load_taxonomy(p)


def test_unknown_layer_rejected(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("failure_modes:\n  - {id: FM-01, name: x, suite: scope, layer: vibes}\n")
    with pytest.raises(ValidationError):
        load_taxonomy(p)


def test_typo_key_rejected(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("failure_modes:\n  - {id: FM-01, name: x, suite: scope, layer: scope, evidance: [a]}\n")
    with pytest.raises(ValidationError):
        load_taxonomy(p)
