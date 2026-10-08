#!/usr/bin/env python3
"""Prepare a pinned MemoryAgentBench pilot (gold separate), or score native results."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import string
import subprocess

DATASET_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
UPSTREAM_REVISION = "538026089d1a8a8eff05121d0db89b388f360eba"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_instruction():
    source = Path(__file__).resolve().parents[1] / "src/server_session.hpp"
    match = re.search(r'shadow\.instruction_override\s*=\s*((?:"(?:\\.|[^"\\])*"\s*)+);', source.read_text())
    if not match:
        raise ValueError(f"Missing literal summary instruction in {source}")
    instruction = "".join(ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"', match[1]))
    if not instruction.strip():
        raise ValueError("Empty summary instruction")
    return instruction, {"path": "src/server_session.hpp", "sha256": sha(source)}


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def parse_output(text):
    # Upstream eval_other_utils.py parse_output, pinned in the manifest.
    for pattern in (r"(?:Answer:)(.*)(?:\n|$)", r"(?:^)(.*)(?:\n|$)"):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return re.sub(r"^Answer:", "", m[1].strip(), flags=re.IGNORECASE).strip()
    return None


def metrics(prediction, answers, event=False):
    parsed = parse_output(prediction)
    candidates = [parsed or ""] if event else [prediction] + ([parsed] if parsed is not None else [])
    expected = [normalize(a) for a in answers]
    return {
        "substring_exact_match": any(a in normalize(p) for p in candidates for a in expected),
        "upstream_exact_match": any(a == normalize(p) for p in candidates for a in expected),
        "strict_exact_match": normalize(prediction) in expected,
    }


def prepare(a):
    import pyarrow.parquet as pq
    a.output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_REVISION:
        raise ValueError(f"Unexpected upstream revision {commit}")
    # Read the templates as literals; do not execute upstream modules/imports.
    tree = ast.parse((a.upstream / "utils/templates.py").read_text())
    system = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "SYSTEM_MESSAGE")
    templates_node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "BASE_TEMPLATES")

    class Literals(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id != "SYSTEM_MESSAGE":
                raise ValueError(node.id)
            return ast.Constant(system)

    templates = ast.literal_eval(Literals().visit(templates_node))
    instruction, instruction_source = summary_instruction()
    conflict = pq.read_table(a.data / "Conflict_Resolution.parquet").to_pylist()
    retrieval = pq.read_table(a.data / "Accurate_Retrieval.parquet").to_pylist()
    selected = [
        ("facts_6k", [(conflict[4], 4, 4), (conflict[0], 4, 0)], "factconsolidation"),
        ("facts_64k", [(conflict[6], 8, 6), (conflict[2], 8, 2)], "factconsolidation"),
        ("eventqa_64k_book0", [(retrieval[7], 8, 7)], "eventqa"),
    ]
    fixture = {"dataset_revision": DATASET_REVISION, "summary_instruction": instruction, "contexts": []}
    gold = {}
    rng = random.Random(a.seed)
    for cid, rows, family in selected:
        assert all(r[0]["context"] == rows[0][0]["context"] for r in rows)
        context = {"id": cid, "system": system, "context": rows[0][0]["context"], "max_output": 128 if family == "eventqa" else 64, "questions": []}
        for row, count, row_index in rows:
            source = row["metadata"]["source"]
            assert source.startswith(family + "_")
            total = len(row["questions"])
            indices = [rng.randrange(i * total // count, (i + 1) * total // count) for i in range(count)]
            for i in indices:
                # EventQA repeats qa_pair_ids between books: include context ID.
                qid = cid + "/" + row["metadata"]["qa_pair_ids"][i]
                assert qid not in gold
                prompt = templates[family]["query"]["long_context_agent"].format(question=row["questions"][i])
                context["questions"].append({"id": qid, "prompt": prompt})
                gold[qid] = {"answers": row["answers"][i], "family": family, "source": source, "row": row_index, "question_index": i}
        fixture["contexts"].append(context)
    write(a.output / "input.json", fixture)
    write(a.output / "gold.json", gold)
    provenance = a.output / "upstream"
    provenance.mkdir()
    files = ["LICENSE", "utils/templates.py", "utils/eval_other_utils.py", "conversation_creator.py"]
    for name in files:
        shutil.copyfile(a.upstream / name, provenance / Path(name).name)
    write(a.output / "manifest.json", {
        "dataset": "ai-hyz/MemoryAgentBench", "dataset_revision": DATASET_REVISION,
        "upstream_revision": commit, "seed": a.seed, "selection": "equal index strata, one seeded random index per stratum",
        "protocol": "adapted native streaming pilot; one user document, Qwen-token quantum64; no upstream gpt4 sentence chunking",
        "questions": len(gold), "ids": list(gold), "summary_instruction_source": instruction_source,
        "sha256": {p.name: sha(p) for p in [a.output / "input.json", a.output / "gold.json", *a.data.glob("*.parquet")]},
    })
    print(json.dumps({"contexts": len(selected), "questions": len(gold), "output": str(a.output)}))


def score(a):
    gold = json.loads(a.gold.read_text())
    result = json.loads(a.result.read_text())
    if not result.get("completed"):
        raise ValueError("Refusing to score an unfinished run")
    expected_contexts = {qid.split("/", 1)[0] for qid in gold}
    actual_contexts = [c["id"] for c in result["contexts"]]
    if len(actual_contexts) != len(set(actual_contexts)) or set(actual_contexts) != expected_contexts:
        raise ValueError("Missing or duplicate benchmark contexts")
    rows, groups = [], {}
    for ctx in result["contexts"]:
        if set(ctx["conditions"]) != {"shadow", "fresh", "truncate", "full"}:
            raise ValueError("Missing benchmark condition")
        expected_ids = {qid for qid in gold if qid.startswith(ctx["id"] + "/")}

        for mode, run in ctx["conditions"].items():
            answers = {x["id"]: x for x in run["answers"]}
            if len(answers) != len(run["answers"]) or not set(answers) <= expected_ids:
                raise ValueError("Duplicate or unexpected answer IDs")
            if run["status"] not in {"completed", "failed", "skipped"}:
                raise ValueError("Unfinished benchmark condition")
            if run["status"] == "completed" and set(answers) != expected_ids:
                raise ValueError("Completed condition has missing answers")
            for qid, target in gold.items():
                if not qid.startswith(ctx["id"] + "/"):
                    continue
                response = answers.get(qid)
                scores = metrics(response["text"], target["answers"], target["family"] == "eventqa") if response else {k: False for k in ("substring_exact_match", "upstream_exact_match", "strict_exact_match")}
                row = {"context": ctx["id"], "mode": mode, "id": qid, "source": target["source"], "status": run["status"], "prediction": response["text"] if response else None, "gold": target["answers"], "eos": response["eos"] if response else None, **scores}
                rows.append(row)
                key = target["source"] + "/" + mode
                group = groups.setdefault(key, {"total": 0, "answered": 0, "eos": 0, **{k: 0 for k in scores}})
                group["total"] += 1
                group["answered"] += response is not None
                group["eos"] += bool(response and response["eos"])
                for k, v in scores.items():
                    group[k] += v
    write(a.output, {"result_sha256": sha(a.result), "gold_sha256": sha(a.gold), "groups": groups, "rows": rows})
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--data", type=Path, required=True)
    q.add_argument("--upstream", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--seed", type=int, default=17)
    q = sub.add_parser("score")
    q.add_argument("--gold", type=Path, required=True)
    q.add_argument("--result", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    (prepare if args.command == "prepare" else score)(args)
