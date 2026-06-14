#!/usr/bin/env python3
"""Convert downloaded neuro-symbolic reasoning datasets to exp_sel_data_out.json schema.

Datasets:
  1. tasksource/ruletaker  — logical entailment over NL rules+facts
  2. tasksource/proofwriter — entailment with gold proof traces
  3. tasksource/folio      — FOL-grounded NL reasoning with FOL annotations
"""

import glob
import json
import re
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WS = Path(__file__).parent
TEMP = WS / "temp" / "datasets"
OUT_DIR = WS / "full_data_out"
CHUNK_SIZE = 15000


# ── helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list:
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return json.loads(path.read_text())


def parse_depth(config: str) -> int:
    """Extract numeric depth from config string like 'depth-3'."""
    m = re.search(r"depth[-_](\d+)", config or "")
    return int(m.group(1)) if m else 0


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, sentence) tuples for each sentence in text."""
    spans = []
    for m in re.finditer(r"[^.!?]+[.!?]?", text):
        s = m.group().strip()
        if s:
            spans.append((m.start(), m.end(), s))
    return spans


def find_span(text: str, fragment: str) -> list[int]:
    """Find character span of fragment in text; return [-1,-1] if not found."""
    idx = text.lower().find(fragment.lower().strip())
    if idx == -1:
        return [-1, -1]
    return [idx, idx + len(fragment.strip())]


def is_rule_sentence(s: str) -> bool:
    """Heuristic: sentence is a rule if it contains 'if' or 'all ... are'."""
    ls = s.lower()
    return bool(re.search(r"\bif\b", ls) or re.search(r"\ball\b.*\bare\b", ls))


def parse_ruletaker_facts(context: str) -> list[dict]:
    """Parse RuleTaker context sentences into extracted_facts entries."""
    facts = []
    for start, end, sent in sentence_spans(context):
        sent = sent.strip()
        if not sent:
            continue
        rule = is_rule_sentence(sent)
        entry: dict = {
            "predicate": sent,
            "arguments": [],
            "source_span": [start, start + len(sent)],
            "source_sentence": sent,
            "is_rule": rule,
        }
        if rule:
            entry["body"] = []
        facts.append(entry)
    return facts


def parse_proofwriter_facts(theory: str) -> list[dict]:
    """Parse ProofWriter theory field into extracted_facts entries."""
    return parse_ruletaker_facts(theory)


def parse_folio_facts(premises: str) -> list[dict]:
    """Parse FOLIO premises (newline-separated) into extracted_facts."""
    facts = []
    offset = 0
    for sent in premises.split("\n"):
        sent = sent.strip()
        if not sent:
            offset += 1
            continue
        rule = is_rule_sentence(sent)
        entry: dict = {
            "predicate": sent,
            "arguments": [],
            "source_span": [offset, offset + len(sent)],
            "source_sentence": sent,
            "is_rule": rule,
        }
        if rule:
            entry["body"] = []
        facts.append(entry)
        offset += len(sent) + 1
    return facts


def make_query(question: str) -> dict:
    """Wrap a question string into target_query dict."""
    return {
        "predicate": question.strip().rstrip("?").strip(),
        "arguments": [],
        "text": question.strip(),
    }


def normalize_answer(label: str) -> str:
    """Normalize label to 'true' or 'false'."""
    l = label.lower().strip()
    if l in ("entailment", "true", "yes", "1"):
        return "true"
    if l in ("not entailment", "false", "no", "0"):
        return "false"
    return "false"  # Unknown / other → false for binary tasks


# ── dataset converters ────────────────────────────────────────────────────────

def convert_ruletaker(rows: list, split: str) -> list[dict]:
    examples = []
    for i, row in enumerate(rows):
        context = row.get("context", "")
        question = row.get("question", "")
        label = row.get("label", "")
        config = row.get("config", "")
        depth = parse_depth(config)
        facts = parse_ruletaker_facts(context)
        num_facts = sum(1 for f in facts if not f["is_rule"])
        num_rules = sum(1 for f in facts if f["is_rule"])
        # input: full context + question prompt
        inp = f"Context: {context}\nQuery: {question}"
        out = normalize_answer(label)
        example = {
            "input": inp,
            "output": out,
            "metadata_dataset_id": "ruletaker",
            "metadata_record_id": f"ruletaker_{split}_{i}",
            "metadata_split": split,
            "metadata_reasoning_depth": depth,
            "metadata_num_facts": num_facts,
            "metadata_num_rules": num_rules,
            "metadata_config": config,
            "metadata_task_type": "logical_entailment",
            "metadata_source_text": context,
            "metadata_question": question,
            "metadata_original_label": label,
            "metadata_extracted_facts_json": json.dumps(facts),
            "metadata_target_query_json": json.dumps(make_query(question)),
        }
        examples.append(example)
    return examples


def convert_proofwriter(rows: list, split: str) -> list[dict]:
    examples = []
    for i, row in enumerate(rows):
        theory = row.get("theory", "")
        question = row.get("question", "")
        answer = str(row.get("answer", "False"))
        config = row.get("config", "")
        max_d = row.get("maxD", 0) or 0
        nfact = row.get("NFact", 0) or 0
        nrule = row.get("NRule", 0) or 0
        qdep = row.get("QDep", 0) or 0
        all_proofs = row.get("allProofs", "") or ""
        record_id = row.get("id", f"pw_{split}_{i}")
        facts = parse_proofwriter_facts(theory)
        inp = f"Theory: {theory}\nQuery: {question}"
        out = normalize_answer(answer)
        example = {
            "input": inp,
            "output": out,
            "metadata_dataset_id": "proofwriter",
            "metadata_record_id": str(record_id),
            "metadata_split": split,
            "metadata_reasoning_depth": int(qdep) if qdep else int(max_d),
            "metadata_num_facts": int(nfact),
            "metadata_num_rules": int(nrule),
            "metadata_config": config,
            "metadata_task_type": "logical_entailment_with_proof",
            "metadata_source_text": theory,
            "metadata_question": question,
            "metadata_original_label": answer,
            "metadata_gold_proof": all_proofs[:500] if all_proofs else "",
            "metadata_extracted_facts_json": json.dumps(facts),
            "metadata_target_query_json": json.dumps(make_query(question)),
        }
        examples.append(example)
    return examples


def convert_folio(rows: list, split: str) -> list[dict]:
    examples = []
    for i, row in enumerate(rows):
        premises = row.get("premises", "")
        premises_fol = row.get("premises-FOL", "")
        conclusion = row.get("conclusion", "")
        conclusion_fol = row.get("conclusion-FOL", "")
        label = str(row.get("label", "False"))
        story_id = row.get("story_id", f"folio_{split}_{i}")
        example_id = row.get("example_id", i)
        facts = parse_folio_facts(premises)
        num_facts = sum(1 for f in facts if not f["is_rule"])
        num_rules = sum(1 for f in facts if f["is_rule"])
        # input: premises (NL) + conclusion to verify
        inp = f"Premises:\n{premises}\nConclusion: {conclusion}"
        out = normalize_answer(label)
        example = {
            "input": inp,
            "output": out,
            "metadata_dataset_id": "folio",
            "metadata_record_id": f"folio_{story_id}_{example_id}",
            "metadata_split": split,
            "metadata_reasoning_depth": num_rules + 1,
            "metadata_num_facts": num_facts,
            "metadata_num_rules": num_rules,
            "metadata_task_type": "fol_entailment",
            "metadata_source_text": premises,
            "metadata_question": conclusion,
            "metadata_original_label": label,
            "metadata_premises_fol": premises_fol[:300] if premises_fol else "",
            "metadata_conclusion_fol": conclusion_fol[:200] if conclusion_fol else "",
            "metadata_extracted_facts_json": json.dumps(facts),
            "metadata_target_query_json": json.dumps(make_query(conclusion)),
        }
        examples.append(example)
    return examples


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== data.py: converting datasets to exp_sel_data_out schema ===")

    datasets_out = []

    # ── 1. ProofWriter ────────────────────────────────────────────────────────
    proofwriter_examples = []
    for split in ("train", "test"):
        path = TEMP / f"full_tasksource_proofwriter_{split}.json"
        if path.exists():
            rows = load_json(path)
            converted = convert_proofwriter(rows, split)
            proofwriter_examples.extend(converted)
            logger.info(f"proofwriter {split}: {len(converted)} examples")
        else:
            logger.warning(f"Missing: {path}")

    if proofwriter_examples:
        datasets_out.append({"dataset": "proofwriter", "examples": proofwriter_examples})
        logger.info(f"proofwriter total: {len(proofwriter_examples)} examples")

    # ── 3. FOLIO ──────────────────────────────────────────────────────────────
    folio_examples = []
    for split in ("train", "validation"):
        suffix = "default_train" if split == "train" else "default_validation"
        path = TEMP / f"full_tasksource_folio_{suffix}.json"
        if path.exists():
            rows = load_json(path)
            converted = convert_folio(rows, split)
            folio_examples.extend(converted)
            logger.info(f"folio {split}: {len(converted)} examples")
        else:
            logger.warning(f"Missing: {path}")

    if folio_examples:
        datasets_out.append({"dataset": "folio", "examples": folio_examples})
        logger.info(f"folio total: {len(folio_examples)} examples")

    # ── save (split into CHUNK_SIZE-example files) ────────────────────────────
    meta = {
        "description": "Neuro-symbolic reasoning datasets standardized for text-to-FOL pipeline",
        "datasets_included": [d["dataset"] for d in datasets_out],
        "total_examples": sum(len(d["examples"]) for d in datasets_out),
    }
    OUT_DIR.mkdir(exist_ok=True)
    # Remove stale parts
    for f in OUT_DIR.glob("full_data_out_*.json"):
        f.unlink()

    # Chunk each dataset and write parts
    part_num = 1
    for ds in datasets_out:
        examples = ds["examples"]
        for i in range(0, len(examples), CHUNK_SIZE):
            chunk = examples[i:i + CHUNK_SIZE]
            part = {"metadata": {**meta, "part": part_num}, "datasets": [{"dataset": ds["dataset"], "examples": chunk}]}
            out_path = OUT_DIR / f"full_data_out_{part_num}.json"
            out_path.write_text(json.dumps(part, indent=2))
            logger.info(f"Part {part_num}: {len(chunk)} examples → {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
            part_num += 1

    total = meta["total_examples"]
    logger.info(f"Saved {total} total examples across {part_num - 1} files in {OUT_DIR}/")

    # Usage hint for reading split files:
    # examples = []
    # for f in sorted(glob.glob(str(OUT_DIR / "full_data_out_*.json"))):
    #     data = json.loads(Path(f).read_text())
    #     for ds in data["datasets"]:
    #         examples.extend(ds["examples"])


if __name__ == "__main__":
    main()
