"""Scoring must not credit answer lists or silently omit failed questions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("memory_bench", Path(__file__).parents[1] / "scripts/memory_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_current_summary_instruction():
    instruction, source = bench.summary_instruction()
    assert instruction == "Суммаризируй предыдущий текст."
    path = Path(__file__).parents[1] / source["path"]
    assert source["sha256"] == bench.sha(path)


def test_flash_summary_instruction_and_provenance():
    instruction, source = bench.summary_instruction("qwen4exp")
    assert "для продолжения разговора" in instruction
    assert "без подтверждения выполнения" in instruction
    assert instruction != bench.summary_instruction("qwen35")[0]
    assert source["architecture"] == "qwen4exp"
    assert source["sha256"] == bench.sha(Path(__file__).parents[1] / source["path"])


def test_reject_unknown_summary_architecture():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        bench.summary_instruction("unknown")


@pytest.mark.parametrize("text,substring,strict", [
    ("The Red Boat.", True, True),
    ("Answer: Red Boat", True, False),
    ("Red Boat or Blue Boat", True, False),
    ("Blue Boat", False, False),
    ("", False, False),
])
def test_answer_enumeration(text, substring, strict):
    scores = bench.metrics(text, ["Red Boat"])
    assert scores["substring_exact_match"] == substring
    assert scores["strict_exact_match"] == strict


def test_event_first_line_matches_upstream():
    assert not bench.metrics("Unknown\nRed Boat", ["Red Boat"], event=True)["substring_exact_match"]
    assert bench.metrics("Unknown\nRed Boat", ["Red Boat"])["substring_exact_match"]


def setup_run(tmp_path):
    gold = {"ctx/q0": {"answers": ["Paris"], "family": "factconsolidation", "source": "test"}}
    conditions = {
        "shadow": {"status": "failed", "answers": []},
        "fresh": {"status": "skipped", "answers": []},
        "truncate": {"status": "completed", "answers": [{"id": "ctx/q0", "text": "Paris or Rome", "eos": True}]},
        "full": {"status": "completed", "answers": [{"id": "ctx/q0", "text": "Paris", "eos": True}]},
    }
    result = {"completed": True, "contexts": [{"id": "ctx", "conditions": conditions}]}
    args = SimpleNamespace(gold=tmp_path / "gold.json", result=tmp_path / "result.json", output=tmp_path / "score.json")
    args.gold.write_text(json.dumps(gold))
    args.result.write_text(json.dumps(result))
    return args, result


def test_failed_conditions_keep_denominator(tmp_path):
    args, _ = setup_run(tmp_path)
    bench.score(args)
    groups = json.loads(args.output.read_text())["groups"]
    assert all(x["total"] == 1 for x in groups.values())
    assert groups["test/shadow"]["answered"] == 0
    assert groups["test/truncate"]["strict_exact_match"] == 0
    assert groups["test/full"]["strict_exact_match"] == 1


def test_reject_incomplete_run(tmp_path):
    args, result = setup_run(tmp_path)
    result["completed"] = False
    args.result.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="unfinished"):
        bench.score(args)


def test_reject_missing_answers(tmp_path):
    args, result = setup_run(tmp_path)
    result["contexts"][0]["conditions"]["full"]["answers"] = []
    args.result.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="missing answers"):
        bench.score(args)
