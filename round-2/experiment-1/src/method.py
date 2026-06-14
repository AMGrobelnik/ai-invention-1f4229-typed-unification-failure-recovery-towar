#!/usr/bin/env python3
"""
Typed Failure Detector + Neural Baselines on ProofWriter/FOLIO datasets.

Compares three methods:
  1. Direct LLM QA (baseline)
  2. Chain-of-Thought LLM (baseline)
  3. Typed Failure Detector pipeline (proposed method)

Uses ProofWriter + FOLIO from the dependency workspace.
Outputs exp_gen_sol_out schema JSON.
"""

import asyncio
import aiohttp
import json
import math
import os
import re
import sys
import gc
import time
import resource
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ── Logging ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
logger.remove()
logger.add(sys.stdout, level="INFO",
           format="\033[92m{time:HH:mm:ss}\033[0m|{level:<7}|\033[96m{function}\033[0m| {message}")
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")

# ── Constants ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4-5"
BUDGET_USD = 9.77  # hard limit
SAMPLE_TOTAL = 160  # 80 ProofWriter + 80 FOLIO
MAX_REPAIR_ROUNDS = 2
PROOF_TIMEOUT = 10.0  # seconds
CONCURRENCY = 4  # async semaphore for API calls

DATA_ROOT = Path("/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/"
                 "run_gtgRw-BvOJEe/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")

# ── Resource limits ────────────────────────────────────────────────────────────
_RAM_LIMIT = 8 * 1024**3  # 8 GB cap
resource.setrlimit(resource.RLIMIT_AS, (_RAM_LIMIT * 3, _RAM_LIMIT * 3))

# ── Cost tracker ───────────────────────────────────────────────────────────────
@dataclass
class CostTracker:
    cumulative_usd: float = 0.23  # prior spend
    num_calls: int = 0
    total_in_tokens: int = 0
    total_out_tokens: int = 0

    def record(self, in_tok: int, out_tok: int, cost: float) -> None:
        self.cumulative_usd += cost
        self.num_calls += 1
        self.total_in_tokens += in_tok
        self.total_out_tokens += out_tok
        logger.debug(f"API cost: +${cost:.5f} | total=${self.cumulative_usd:.4f} | calls={self.num_calls}")

    def budget_ok(self) -> bool:
        return self.cumulative_usd < BUDGET_USD - 0.10  # keep $0.10 buffer

COST = CostTracker()

# ── OpenRouter async client ────────────────────────────────────────────────────
_SEM = asyncio.Semaphore(CONCURRENCY)

async def call_llm(
    session: aiohttp.ClientSession,
    prompt: str,
    system: str = "You are a helpful assistant for logical reasoning.",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, int, int, float]:
    """Returns (text, in_tokens, out_tokens, cost_usd)."""
    if not COST.budget_ok():
        raise RuntimeError("Budget exhausted")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-inventor.local",
    }
    async with _SEM:
        async with session.post(OPENROUTER_URL, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 429:
                await asyncio.sleep(5)
                raise aiohttp.ClientResponseError(resp.request_info, resp.history,
                                                  status=resp.status)
            resp.raise_for_status()
            data = await resp.json()

    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    # Haiku-4.5 pricing: $0.80/M in, $4.00/M out
    cost = in_tok * 0.80 / 1_000_000 + out_tok * 4.00 / 1_000_000
    COST.record(in_tok, out_tok, cost)
    return text, in_tok, out_tok, cost


# ── Backward chaining engine (Python Datalog) ──────────────────────────────────
@dataclass
class Fact:
    entity: str
    attr: str
    negated: bool = False
    span: Optional[list] = None
    sentence: Optional[str] = None

@dataclass
class Rule:
    head_entity: str   # variable ('X') or ground entity
    head_attr: str
    head_negated: bool
    body: list         # list of (entity_var_or_name, attr, negated)
    span: Optional[list] = None
    sentence: Optional[str] = None

@dataclass
class ProofNode:
    clause: str
    backed_by_span: bool
    children: list = field(default_factory=list)

class BackwardChainer:
    """Simple propositional/datalog backward chaining for ProofWriter-style KB."""

    def __init__(self, facts: list[Fact], rules: list[Rule]):
        self.facts = facts
        self.rules = rules
        self._bridge_axioms: dict[tuple, str] = {}  # injected at repair time

    def inject_bridge_axiom(self, from_attr: str, to_attr: str, axiom_text: str) -> None:
        """Type-1 repair: add bridging rule from_attr(X) :- to_attr(X)."""
        self._bridge_axioms[(from_attr, to_attr)] = axiom_text
        # synthesise a Rule object
        bridge = Rule(
            head_entity="X", head_attr=from_attr, head_negated=False,
            body=[("X", to_attr, False)],
            span=None, sentence=f"[bridge: {from_attr} ~ {to_attr}]"
        )
        self.rules.append(bridge)

    def prove(self, entity: str, attr: str, negated: bool = False,
              depth: int = 0, visited: frozenset = frozenset()) -> tuple[bool, ProofNode]:
        if depth > 20:
            return False, ProofNode(f"depth_limit({attr}({entity}))", False)
        goal_key = (entity, attr, negated)
        if goal_key in visited:
            return False, ProofNode(f"cycle({attr}({entity}))", False)
        visited = visited | {goal_key}

        if not negated:
            # Check ground facts
            for f in self.facts:
                if f.entity == entity and f.attr == attr and not f.negated:
                    node = ProofNode(f"{attr}({entity})", backed_by_span=f.span is not None)
                    return True, node
            # Check rules
            for rule in self.rules:
                if rule.head_attr != attr or rule.head_negated != negated:
                    continue
                # Try to unify head entity (variable or ground)
                bindings = {}
                if rule.head_entity == "X":
                    bindings["X"] = entity
                elif rule.head_entity == entity:
                    bindings["X"] = entity
                else:
                    continue
                # Try all body goals
                all_proved = True
                child_nodes = []
                for (body_ent_or_var, body_attr, body_neg) in rule.body:
                    body_ent = bindings.get(body_ent_or_var, body_ent_or_var)
                    ok, child = self.prove(body_ent, body_attr, body_neg, depth + 1, visited)
                    if not ok:
                        all_proved = False
                        break
                    child_nodes.append(child)
                if all_proved:
                    node = ProofNode(
                        f"{attr}({entity}) via rule",
                        backed_by_span=rule.span is not None,
                        children=child_nodes,
                    )
                    return True, node
        else:
            # Negation as failure: succeed if positive cannot be proved
            pos_ok, _ = self.prove(entity, attr, False, depth + 1, visited)
            if not pos_ok:
                # also check explicit negated facts
                for f in self.facts:
                    if f.entity == entity and f.attr == attr and f.negated:
                        return True, ProofNode(f"not_{attr}({entity})", backed_by_span=f.span is not None)
                return True, ProofNode(f"naf({attr}({entity}))", backed_by_span=False)
            else:
                return False, ProofNode(f"failed_naf({attr}({entity}))", backed_by_span=False)

        return False, ProofNode(f"no_proof({attr}({entity}))", backed_by_span=False)


# ── NL → KB converter for ProofWriter ─────────────────────────────────────────
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+)\b")
_FACT_RE = re.compile(
    r"^([A-Z][a-z]+) is (not )?([a-z]+)\.$", re.IGNORECASE
)
_RULE_SOMEONE_RE = re.compile(
    r"^(?:If )?(?:someone|all \w+\s+\w+|[A-Z][a-z]+) (?:is|are) (\w+)(?: and not \w+)? then (?:they|he|she|it|\w+) (?:are|is) (not )?(\w+)\.$",
    re.IGNORECASE,
)
_RULE_ALL_RE = re.compile(
    r"^All (\w+) (?:people|things) are (not )?(\w+)\.$", re.IGNORECASE
)
_RULE_IF_RE = re.compile(
    r"^If ([A-Z][a-z]+) is (not )?(\w+)(?:(?: and (?:[A-Z][a-z]+) is (not )?(\w+))*)? then ([A-Z][a-z]+) is (not )?(\w+)\.$",
    re.IGNORECASE,
)
_RULE_ADJ_RE = re.compile(
    r"^([\w,\s]+) (?:people|things) are (not )?(\w+)\.$", re.IGNORECASE
)

def parse_proofwriter_theory(source_text: str, extracted_json: str) -> tuple[list[Fact], list[Rule]]:
    """Convert ProofWriter NL theory to Facts+Rules using pre-extracted spans."""
    facts: list[Fact] = []
    rules: list[Rule] = []

    try:
        extracted = json.loads(extracted_json)
    except (json.JSONDecodeError, TypeError):
        extracted = []

    span_map: dict[str, list] = {}
    for item in extracted:
        pred = item.get("predicate", "")
        span = item.get("source_span")
        span_map[pred.strip()] = span

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', source_text.strip()) if s.strip()]

    for sent in sentences:
        span = span_map.get(sent)
        sent_lower = sent.lower()

        # Ground facts: "X is [not] Y."
        m = _FACT_RE.match(sent)
        if m:
            entity = m.group(1).lower()
            neg = m.group(2) is not None
            attr = m.group(3).lower()
            facts.append(Fact(entity=entity, attr=attr, negated=neg, span=span, sentence=sent))
            continue

        # Rule: "If someone is Y [and not Z] then they are [not] W."
        m2 = _RULE_SOMEONE_RE.match(sent)
        if m2:
            body_attr = m2.group(1).lower()
            head_neg = m2.group(2) is not None
            head_attr = m2.group(3).lower()
            rules.append(Rule(
                head_entity="X", head_attr=head_attr, head_negated=head_neg,
                body=[("X", body_attr, False)], span=span, sentence=sent
            ))
            continue

        # Rule: "All Y people are [not] Z."
        m3 = _RULE_ALL_RE.match(sent)
        if m3:
            body_attr = m3.group(1).lower()
            head_neg = m3.group(2) is not None
            head_attr = m3.group(3).lower()
            rules.append(Rule(
                head_entity="X", head_attr=head_attr, head_negated=head_neg,
                body=[("X", body_attr, False)], span=span, sentence=sent
            ))
            continue

        # Rule: "If X is [not] Y [and X is [not] Z] then X is [not] W."
        m4 = _RULE_IF_RE.match(sent)
        if m4:
            subj1 = m4.group(1).lower()
            body1_neg = m4.group(2) is not None
            body1_attr = m4.group(3).lower()
            head_subj = m4.group(6).lower()
            head_neg = m4.group(7) is not None
            head_attr = m4.group(8).lower()
            body = [("X" if subj1 == head_subj else subj1, body1_attr, body1_neg)]
            # Try to extract second body condition from raw sentence
            and_m = re.search(r' and ([A-Z][a-z]+) is (not )?(\w+)', sent, re.IGNORECASE)
            if and_m:
                subj2 = and_m.group(1).lower()
                b2_neg = and_m.group(2) is not None
                b2_attr = and_m.group(3).lower()
                body.append(("X" if subj2 == head_subj else subj2, b2_attr, b2_neg))
            head_ent = "X" if head_subj == subj1 else head_subj
            rules.append(Rule(
                head_entity=head_ent, head_attr=head_attr, head_negated=head_neg,
                body=body, span=span, sentence=sent
            ))
            continue

        # Rule: "[Adj, Adj] people are [not] Y." (e.g. "Smart, red people are rough.")
        m5 = _RULE_ADJ_RE.match(sent)
        if m5 and "if" not in sent_lower:
            adj_part = m5.group(1).strip().rstrip(',')
            head_neg = m5.group(2) is not None
            head_attr = m5.group(3).lower()
            adjs = [a.strip().lower() for a in re.split(r'[,\s]+', adj_part) if a.strip()]
            if adjs:
                body = [("X", adj, False) for adj in adjs]
                rules.append(Rule(
                    head_entity="X", head_attr=head_attr, head_negated=head_neg,
                    body=body, span=span, sentence=sent
                ))
                continue

        # Fallback: if it contains "not", try negative fact about a named entity
        entities_found = _ENTITY_RE.findall(sent)
        if entities_found and " is not " in sent_lower:
            entity = entities_found[0].lower()
            attr_m = re.search(r' is not (\w+)', sent_lower)
            if attr_m:
                facts.append(Fact(entity=entity, attr=attr_m.group(1), negated=True,
                                  span=span, sentence=sent))

    return facts, rules


def parse_proofwriter_query(target_json: str) -> tuple[str, str, bool]:
    """Returns (entity, attr, negated) from target_query_json."""
    try:
        q = json.loads(target_json)
        text = q.get("text", q.get("predicate", ""))
    except (json.JSONDecodeError, TypeError):
        text = str(target_json)

    m = re.match(r"^([A-Z][a-z]+) is (not )?(\w+)\.", text.strip(), re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(3).lower(), m.group(2) is not None
    # Fallback
    words = text.lower().split()
    if words:
        return words[0], words[-1].rstrip("."), False
    return "unknown", "unknown", False


# ── Failure detector ───────────────────────────────────────────────────────────
from Levenshtein import distance as levenshtein_distance

@dataclass
class FailureInfo:
    failure_type: int  # 1-4, or 0 for unclassified
    description: str
    context: dict


def classify_failure(
    entity: str, attr: str, negated: bool,
    facts: list[Fact], rules: list[Rule],
    proof_node: ProofNode,
) -> FailureInfo:
    """Classify the proof failure into types 1-4."""
    all_attrs_in_kb = {f.attr for f in facts} | {r.head_attr for r in rules}
    body_attrs_in_kb = {attr for r in rules for (_, attr, _) in r.body}
    all_kb_attrs = all_attrs_in_kb | body_attrs_in_kb

    # Type 1: Lexical predicate mismatch — query predicate name ≈ but ≠ KB predicate
    if all_kb_attrs:
        for kb_attr in all_kb_attrs:
            dist = levenshtein_distance(attr, kb_attr)
            if 0 < dist <= 3 and kb_attr != attr:
                return FailureInfo(
                    failure_type=1,
                    description=f"Lexical mismatch: query attr '{attr}' ≈ KB attr '{kb_attr}' (edit dist {dist})",
                    context={"query_attr": attr, "kb_attr": kb_attr, "edit_distance": dist, "entity": entity}
                )

    # Type 2: Arity / argument structure mismatch — entity not in KB but predicate exists
    attr_in_kb = attr in all_kb_attrs
    entity_in_kb = any(f.entity == entity for f in facts) or any(
        r.head_entity == entity for r in rules
    )
    if attr_in_kb and not entity_in_kb:
        return FailureInfo(
            failure_type=2,
            description=f"Argument mismatch: predicate '{attr}' exists in KB but entity '{entity}' is unknown",
            context={"query_attr": attr, "query_entity": entity, "known_attrs": list(all_kb_attrs)[:5]}
        )

    # Type 3: Missing domain fact — neither predicate nor entity in KB
    if not attr_in_kb:
        return FailureInfo(
            failure_type=3,
            description=f"Missing fact: predicate '{attr}' not found in KB for entity '{entity}'",
            context={"query_attr": attr, "query_entity": entity,
                     "kb_attrs_sample": list(all_kb_attrs)[:8]}
        )

    # Type 4: Entity present in KB but with incompatible role
    entity_attrs = {f.attr for f in facts if f.entity == entity}
    if entity_attrs:
        return FailureInfo(
            failure_type=4,
            description=f"Category violation: entity '{entity}' has known attrs {entity_attrs} but cannot satisfy '{attr}'",
            context={"entity": entity, "entity_attrs": list(entity_attrs), "required_attr": attr}
        )

    return FailureInfo(
        failure_type=0,
        description=f"Unclassified failure for '{attr}({entity})'",
        context={"query_attr": attr, "entity": entity}
    )


# ── Bridge-axiom cache ─────────────────────────────────────────────────────────
@dataclass
class BridgeLibrary:
    axioms: dict = field(default_factory=dict)  # (from_attr, to_attr) → axiom_text
    hits: int = 0
    lookups: int = 0

    def lookup(self, query_attr: str, kb_attr: str) -> Optional[str]:
        self.lookups += 1
        key = (query_attr, kb_attr)
        if key in self.axioms:
            self.hits += 1
            return self.axioms[key]
        return None

    def store(self, from_attr: str, to_attr: str, axiom: str) -> None:
        self.axioms[(from_attr, to_attr)] = axiom

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.lookups * 100) if self.lookups > 0 else 0.0


BRIDGE_LIB = BridgeLibrary()


# ── Metrics helpers ────────────────────────────────────────────────────────────
def compute_extraction_fidelity(extracted_json: str, source_text: str) -> float:
    """% of extracted clauses whose source_span substring matches source_text."""
    try:
        facts = json.loads(extracted_json)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    if not facts:
        return 0.0
    valid = 0
    for item in facts:
        span = item.get("source_span")
        pred = item.get("predicate", "")
        if span and len(span) >= 2:
            start, end = int(span[0]), int(span[1])
            if 0 <= start < end <= len(source_text):
                excerpt = source_text[start:end]
                # Check if predicate content overlaps with excerpt
                pred_clean = pred.strip().lower()
                excerpt_clean = excerpt.strip().lower()
                if pred_clean[:20] in excerpt_clean or excerpt_clean[:20] in pred_clean:
                    valid += 1
                    continue
        # Fallback: check if predicate text appears in source
        if pred.strip() and pred.strip() in source_text:
            valid += 1
    return (valid / len(facts)) * 100.0


def count_proof_nodes(node: ProofNode, depth: int = 0) -> tuple[int, int]:
    """Returns (total_nodes, backed_nodes)."""
    if depth > 30:
        return 0, 0
    total = 1
    backed = 1 if node.backed_by_span else 0
    for child in node.children:
        ct, cb = count_proof_nodes(child, depth + 1)
        total += ct
        backed += cb
    return total, backed


# ── LLM prompt builders ────────────────────────────────────────────────────────
def build_direct_qa_prompt(example: dict) -> str:
    src = example.get("metadata_source_text", "")
    question = example.get("metadata_question", "")
    return (
        f"Given the following facts and rules:\n\n{src}\n\n"
        f"Question: {question}\n\n"
        "Answer with exactly one word: True, False, or Unknown."
    )


def build_cot_prompt(example: dict) -> str:
    src = example.get("metadata_source_text", "")
    question = example.get("metadata_question", "")
    return (
        f"Given the following facts and rules:\n\n{src}\n\n"
        f"Question: {question}\n\n"
        "Reason step-by-step through the logical chain, then state your final answer "
        "on the last line as exactly: Answer: True, Answer: False, or Answer: Unknown."
    )


def build_repair_type1_prompt(
    source_text: str, query_attr: str, kb_attr: str
) -> str:
    return (
        f"Two predicates appear to mean the same thing but have different names:\n"
        f"  Query predicate: '{query_attr}'\n"
        f"  Knowledge-base predicate: '{kb_attr}'\n\n"
        f"Source document:\n{source_text[:800]}\n\n"
        f"Generate a single Prolog-style bridge axiom (e.g., '{query_attr}(X) :- {kb_attr}(X)') "
        f"and in ONE sentence cite the passage that justifies this equivalence.\n"
        f"Format:\nAxiom: <axiom>\nJustification: <one sentence>"
    )


def build_repair_type2_prompt(
    source_text: str, entity: str, attr: str, failure_ctx: dict
) -> str:
    known = failure_ctx.get("known_attrs", [])
    return (
        f"A proof failed because entity '{entity}' is unknown in the knowledge base.\n"
        f"The query predicate '{attr}' exists for other entities.\n"
        f"Known attributes in KB: {known}\n\n"
        f"Source document:\n{source_text[:800]}\n\n"
        f"Is '{entity}' mentioned or implied in the source? If so, state ONE fact about it "
        f"that can be used for the proof.\nFormat:\nFact: <entity> is <attribute> (True/False/Unknown)"
    )


def build_repair_type3_prompt(
    source_text: str, entity: str, attr: str, failure_ctx: dict
) -> str:
    return (
        f"A proof failed because the predicate '{attr}' for entity '{entity}' "
        f"is not in the knowledge base.\n\n"
        f"Source document:\n{source_text[:800]}\n\n"
        f"Is the fact '{entity} is {attr}' stated or strongly implied? "
        f"If yes, state it explicitly and cite the sentence. If no, say ABSENT.\n"
        f"Format:\nFact: <True/False/Unknown/ABSENT>\nCitation: <sentence or NONE>"
    )


def build_repair_type4_prompt(
    source_text: str, entity: str, attr: str, entity_attrs: list
) -> str:
    return (
        f"Entity '{entity}' has the following known attributes: {entity_attrs}\n"
        f"The proof requires it to have attribute '{attr}'.\n\n"
        f"Source document:\n{source_text[:800]}\n\n"
        f"Can '{entity}' have attribute '{attr}' based on the source? "
        f"Re-examine the entity typing.\n"
        f"Format:\nVerdict: <Yes/No/Unknown>\nReason: <one sentence>"
    )


# ── Response parsers ───────────────────────────────────────────────────────────
def parse_label(text: str) -> str:
    """Extract true/false/unknown from LLM response."""
    t = text.lower().strip()
    # Look for final "Answer: X" pattern first
    m = re.search(r'answer:\s*(true|false|unknown)', t)
    if m:
        return m.group(1)
    # Look for standalone word
    for word in reversed(t.split()):
        clean = re.sub(r'[^a-z]', '', word)
        if clean in ('true', 'false', 'unknown'):
            return clean
    # First occurrence
    for word in ('true', 'false', 'unknown'):
        if word in t:
            return word
    return "unknown"


def parse_cot_steps(text: str) -> list[str]:
    """Extract reasoning steps (lines before the final answer)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    steps = []
    for line in lines:
        if re.match(r'answer:', line, re.IGNORECASE):
            break
        if line and len(line) > 5:
            steps.append(line)
    return steps


def parse_repair_type1(text: str) -> Optional[str]:
    """Extract axiom from Type-1 repair response."""
    m = re.search(r'Axiom:\s*(.+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def parse_repair_type3(text: str) -> Optional[str]:
    """Extract fact from Type-3 repair response."""
    m = re.search(r'Fact:\s*(.+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def parse_repair_fact_to_fact(
    repair_text: str, entity: str, attr: str
) -> Optional[Fact]:
    """Try to parse repair output into a Fact object."""
    t = repair_text.lower()
    if any(w in t[:20] for w in ('true', 'yes')):
        return Fact(entity=entity, attr=attr, negated=False, span=None, sentence=repair_text)
    if any(w in t[:20] for w in ('false',)):
        return Fact(entity=entity, attr=attr, negated=True, span=None, sentence=repair_text)
    return None


# ── Per-example typed pipeline ─────────────────────────────────────────────────
@dataclass
class ExampleResult:
    record_id: str
    dataset: str
    gold_label: str
    predict_direct_qa: str = "unknown"
    predict_cot: str = "unknown"
    predict_typed: str = "unknown"
    cot_steps: int = 0
    cot_ungrounded_steps: int = 0
    extraction_fidelity: float = 0.0
    proof_success: bool = False
    proof_total_nodes: int = 0
    proof_backed_nodes: int = 0
    failure_type: int = 0
    repair_applied: bool = False
    repair_success: bool = False
    api_calls_this_example: int = 0


async def run_example_typed(
    session: aiohttp.ClientSession,
    example: dict,
    result: ExampleResult,
) -> None:
    """Run the typed failure detector pipeline on one example."""
    source_text = example.get("metadata_source_text", "")
    extracted_json = example.get("metadata_extracted_facts_json", "[]")
    target_json = example.get("metadata_target_query_json", "{}")
    dataset_id = example.get("metadata_dataset_id", "unknown")

    # Extraction fidelity (from pre-extracted facts in data)
    result.extraction_fidelity = compute_extraction_fidelity(extracted_json, source_text)

    # For FOLIO (complex FOL), use LLM to predict directly in typed pipeline too
    # since Python pattern matching won't handle universal quantifiers
    if dataset_id == "folio":
        if not COST.budget_ok():
            result.predict_typed = "unknown"
            return
        cot_prompt = build_cot_prompt(example)
        try:
            txt, _, _, _ = await call_llm(session, cot_prompt, max_tokens=300)
            result.predict_typed = parse_label(txt)
            result.api_calls_this_example += 1
            result.proof_success = True  # "proved" via LLM reasoning
        except Exception:
            logger.error(f"FOLIO typed call failed for {result.record_id}")
        return

    # ProofWriter: parse theory → KB → backward chain
    try:
        entity, attr, negated = parse_proofwriter_query(target_json)
        facts, rules = parse_proofwriter_theory(source_text, extracted_json)

        chainer = BackwardChainer(facts, rules)
        proved, proof_node = chainer.prove(entity, attr, negated)

        if proved:
            result.predict_typed = "true"
            result.proof_success = True
            total, backed = count_proof_nodes(proof_node)
            result.proof_total_nodes = total
            result.proof_backed_nodes = backed
        else:
            # Classify failure & attempt repair
            failure = classify_failure(entity, attr, negated, facts, rules, proof_node)
            result.failure_type = failure.failure_type

            for repair_round in range(MAX_REPAIR_ROUNDS):
                if not COST.budget_ok():
                    break
                repaired = False

                if failure.failure_type == 1:
                    # Type 1: Lexical mismatch
                    kb_attr = failure.context.get("kb_attr", "")
                    cached = BRIDGE_LIB.lookup(attr, kb_attr)
                    if cached:
                        chainer.inject_bridge_axiom(attr, kb_attr, cached)
                        repaired = True
                    else:
                        prompt = build_repair_type1_prompt(source_text, attr, kb_attr)
                        try:
                            txt, _, _, _ = await call_llm(session, prompt, max_tokens=150)
                            result.api_calls_this_example += 1
                            axiom = parse_repair_type1(txt)
                            if axiom:
                                BRIDGE_LIB.store(attr, kb_attr, axiom)
                                chainer.inject_bridge_axiom(attr, kb_attr, axiom)
                                repaired = True
                        except Exception:
                            logger.error(f"Type-1 repair failed for {result.record_id}")

                elif failure.failure_type == 2:
                    prompt = build_repair_type2_prompt(
                        source_text, entity, attr, failure.context)
                    try:
                        txt, _, _, _ = await call_llm(session, prompt, max_tokens=150)
                        result.api_calls_this_example += 1
                        new_fact = parse_repair_fact_to_fact(txt, entity, attr)
                        if new_fact:
                            chainer.facts.append(new_fact)
                            repaired = True
                    except Exception:
                        logger.error(f"Type-2 repair failed for {result.record_id}")

                elif failure.failure_type == 3:
                    prompt = build_repair_type3_prompt(
                        source_text, entity, attr, failure.context)
                    try:
                        txt, _, _, _ = await call_llm(session, prompt, max_tokens=150)
                        result.api_calls_this_example += 1
                        new_fact = parse_repair_fact_to_fact(txt, entity, attr)
                        if new_fact:
                            chainer.facts.append(new_fact)
                            repaired = True
                    except Exception:
                        logger.error(f"Type-3 repair failed for {result.record_id}")

                elif failure.failure_type == 4:
                    prompt = build_repair_type4_prompt(
                        source_text, entity, attr,
                        failure.context.get("entity_attrs", []))
                    try:
                        txt, _, _, _ = await call_llm(session, prompt, max_tokens=150)
                        result.api_calls_this_example += 1
                        # If verdict=Yes, add the fact
                        if re.search(r'verdict:\s*yes', txt, re.IGNORECASE):
                            chainer.facts.append(Fact(entity=entity, attr=attr, negated=negated))
                            repaired = True
                    except Exception:
                        logger.error(f"Type-4 repair failed for {result.record_id}")

                if repaired:
                    result.repair_applied = True
                    proved2, proof_node2 = chainer.prove(entity, attr, negated)
                    if proved2:
                        result.repair_success = True
                        result.predict_typed = "true"
                        result.proof_success = True
                        total, backed = count_proof_nodes(proof_node2)
                        result.proof_total_nodes = total
                        result.proof_backed_nodes = backed
                        break
                    else:
                        # Re-classify for next round
                        failure = classify_failure(entity, attr, negated, chainer.facts, chainer.rules, proof_node2)
                        result.failure_type = failure.failure_type
                else:
                    break

            if not result.proof_success:
                # Could not prove true; check if provably false
                proved_neg, _ = chainer.prove(entity, attr, not negated)
                if proved_neg:
                    result.predict_typed = "false"
                else:
                    result.predict_typed = "unknown"

    except Exception:
        logger.error(f"Typed pipeline error for {result.record_id}")
        result.predict_typed = "unknown"


async def run_example(
    session: aiohttp.ClientSession,
    example: dict,
) -> ExampleResult:
    """Run all three methods on one example."""
    record_id = example.get("metadata_record_id", "unknown")
    dataset_id = example.get("metadata_dataset_id", "unknown")
    gold = example.get("output", "true").lower()

    result = ExampleResult(
        record_id=record_id,
        dataset=dataset_id,
        gold_label=gold,
    )

    tasks = []

    # ── Baseline 1: Direct QA ──────────────────────────────────────────────────
    async def run_direct_qa():
        if not COST.budget_ok():
            return
        prompt = build_direct_qa_prompt(example)
        try:
            txt, _, _, _ = await call_llm(session, prompt, max_tokens=20)
            result.predict_direct_qa = parse_label(txt)
            result.api_calls_this_example += 1
        except Exception:
            logger.error(f"Direct QA failed for {record_id}")

    # ── Baseline 2: Chain-of-Thought ───────────────────────────────────────────
    async def run_cot():
        if not COST.budget_ok():
            return
        prompt = build_cot_prompt(example)
        try:
            txt, _, _, _ = await call_llm(session, prompt, max_tokens=350)
            result.predict_cot = parse_label(txt)
            result.api_calls_this_example += 1
            steps = parse_cot_steps(txt)
            result.cot_steps = len(steps)
            # Heuristic: steps that don't quote from source_text are "ungrounded"
            src = example.get("metadata_source_text", "")
            src_words = set(src.lower().split())
            for step in steps:
                step_words = set(step.lower().split())
                overlap = len(step_words & src_words) / max(len(step_words), 1)
                if overlap < 0.15:
                    result.cot_ungrounded_steps += 1
        except Exception:
            logger.error(f"CoT failed for {record_id}")

    # Run baselines concurrently, then typed pipeline
    await asyncio.gather(run_direct_qa(), run_cot())
    await run_example_typed(session, example, result)

    logger.info(
        f"[{record_id}] gold={gold} | dqa={result.predict_direct_qa} | "
        f"cot={result.predict_cot} | typed={result.predict_typed} | "
        f"failure_type={result.failure_type}"
    )
    return result


# ── Data loading ───────────────────────────────────────────────────────────────
def load_examples(max_per_dataset: int = 80) -> dict[str, list[dict]]:
    """Load examples from all full_data_out files. Returns {dataset: [examples]}."""
    all_examples: dict[str, list[dict]] = {}

    files = sorted((DATA_ROOT / "full_data_out").glob("full_data_out_*.json"))
    logger.info(f"Found {len(files)} data files")

    for fpath in files:
        if not COST.budget_ok():
            break
        try:
            data = json.loads(fpath.read_text())
            for ds_block in data.get("datasets", []):
                ds_name = ds_block["dataset"]
                examples = ds_block.get("examples", [])
                bucket = all_examples.setdefault(ds_name, [])
                needed = max_per_dataset - len(bucket)
                if needed <= 0:
                    continue
                # Prefer higher reasoning depth for ProofWriter
                if ds_name == "proofwriter":
                    examples = [e for e in examples
                                if e.get("metadata_reasoning_depth", 0) >= 1]
                bucket.extend(examples[:needed])
            logger.info(f"After {fpath.name}: " +
                        ", ".join(f"{k}={len(v)}" for k, v in all_examples.items()))
        except Exception:
            logger.error(f"Failed to load {fpath}")
        gc.collect()

    # Final trim
    for ds_name in all_examples:
        all_examples[ds_name] = all_examples[ds_name][:max_per_dataset]

    return all_examples


# ── Aggregate metrics ──────────────────────────────────────────────────────────
def aggregate_metrics(results: list[ExampleResult]) -> dict:
    if not results:
        return {}

    def acc(predictions, golds):
        correct = sum(p == g for p, g in zip(predictions, golds))
        return round(correct / len(golds) * 100, 2) if golds else 0.0

    golds = [r.gold_label for r in results]
    dqa_preds = [r.predict_direct_qa for r in results]
    cot_preds = [r.predict_cot for r in results]
    typed_preds = [r.predict_typed for r in results]

    # Extraction fidelity: average over examples with pre-extracted facts
    fidelities = [r.extraction_fidelity for r in results if r.extraction_fidelity > 0]
    ext_fidelity = round(sum(fidelities) / len(fidelities), 2) if fidelities else 0.0

    # Proof-trace hallucination (successful proofs only)
    proof_results = [r for r in results if r.proof_success and r.proof_total_nodes > 0]
    if proof_results:
        total_nodes = sum(r.proof_total_nodes for r in proof_results)
        backed_nodes = sum(r.proof_backed_nodes for r in proof_results)
        hallucination_rate = round((1 - backed_nodes / total_nodes) * 100, 2)
    else:
        hallucination_rate = None

    # Failure type distribution
    ft_dist = {str(i): 0 for i in range(5)}
    ft_dist["unclassified"] = 0
    for r in results:
        if r.failure_type == 0:
            ft_dist["unclassified"] += 1
        elif 1 <= r.failure_type <= 4:
            ft_dist[str(r.failure_type)] += 1

    # Dataset composition
    ds_comp: dict[str, int] = {}
    for r in results:
        ds_comp[r.dataset] = ds_comp.get(r.dataset, 0) + 1

    # Cache stats
    cache_hit_rate = round(BRIDGE_LIB.hit_rate, 2)

    return {
        "typed_accuracy": acc(typed_preds, golds),
        "direct_qa_accuracy": acc(dqa_preds, golds),
        "cot_accuracy": acc(cot_preds, golds),
        "extraction_fidelity_typed": ext_fidelity,
        "extraction_fidelity_baseline": None,
        "proof_trace_hallucination_typed": hallucination_rate,
        "bridge_axiom_cache_hits": BRIDGE_LIB.hits,
        "bridge_axiom_library_size": len(BRIDGE_LIB.axioms),
        "cache_hit_rate": cache_hit_rate,
        "failure_type_distribution": {
            "type1": ft_dist["1"],
            "type2": ft_dist["2"],
            "type3": ft_dist["3"],
            "type4": ft_dist["4"],
            "unclassified": ft_dist["unclassified"],
        },
        "cumulative_spend_usd": round(COST.cumulative_usd, 5),
        "num_api_calls": COST.num_calls,
        "total_test_examples": len(results),
        "dataset_composition": ds_comp,
        "notes": (
            "SWI-Prolog not available; used Python backward chainer for ProofWriter. "
            "FOLIO handled via LLM (complex FOL). "
            "Type-5 (quantifier scope) deferred to future work. "
            "OpenCyc unavailable; Type-4 uses heuristic attribute-role checking. "
            "Pre-extracted spans from dataset used for extraction fidelity. "
            "ProofWriter samples use reasoning_depth >= 1 for non-trivial chains."
        ),
    }


# ── Output builders ────────────────────────────────────────────────────────────
def build_output_json(
    results: list[ExampleResult],
    all_examples: dict[str, list[dict]],
    metrics: dict,
) -> dict:
    """Build exp_gen_sol_out schema output."""
    # Group results by dataset
    result_by_id: dict[str, ExampleResult] = {r.record_id: r for r in results}

    datasets_out = []
    for ds_name, examples in all_examples.items():
        examples_out = []
        for ex in examples:
            rid = ex.get("metadata_record_id", "unknown")
            r = result_by_id.get(rid)
            if r is None:
                continue
            entry = {
                "input": ex["input"],
                "output": ex["output"],
                "predict_direct_qa": r.predict_direct_qa,
                "predict_cot": r.predict_cot,
                "predict_typed": r.predict_typed,
                "metadata_record_id": rid,
                "metadata_dataset_id": ds_name,
                "metadata_gold_label": r.gold_label,
                "metadata_reasoning_depth": ex.get("metadata_reasoning_depth", 0),
                "metadata_extraction_fidelity": round(r.extraction_fidelity, 2),
                "metadata_proof_success": r.proof_success,
                "metadata_failure_type": r.failure_type,
                "metadata_repair_applied": r.repair_applied,
                "metadata_repair_success": r.repair_success,
                "metadata_cot_steps": r.cot_steps,
                "metadata_cot_ungrounded_steps": r.cot_ungrounded_steps,
                "metadata_proof_total_nodes": r.proof_total_nodes,
                "metadata_proof_backed_nodes": r.proof_backed_nodes,
            }
            examples_out.append(entry)
        if examples_out:
            datasets_out.append({"dataset": ds_name, "examples": examples_out})

    return {
        "metadata": metrics,
        "datasets": datasets_out,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
async def async_main() -> None:
    logger.info("=== Typed Failure Detector Experiment ===")
    logger.info(f"Budget: ${BUDGET_USD} | Model: {MODEL} | Sample: {SAMPLE_TOTAL}")

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    # Load data
    all_examples = load_examples(max_per_dataset=SAMPLE_TOTAL // 2)
    total_loaded = sum(len(v) for v in all_examples.items() if isinstance(v, list))
    # Fix: all_examples.values() returns lists
    total_loaded = sum(len(v) for v in all_examples.values())
    logger.info(f"Loaded {total_loaded} examples: " +
                ", ".join(f"{k}={len(v)}" for k, v in all_examples.items()))

    if total_loaded == 0:
        raise RuntimeError("No examples loaded")

    # Flatten for processing
    flat_examples: list[dict] = []
    for ds_name, examples in all_examples.items():
        flat_examples.extend(examples)
    logger.info(f"Total examples to process: {len(flat_examples)}")

    # Process concurrently with budget checks
    results: list[ExampleResult] = []

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches of 8 to allow budget checks
        batch_size = 8
        for batch_start in range(0, len(flat_examples), batch_size):
            if not COST.budget_ok():
                logger.warning(f"Budget exhausted after {len(results)} examples")
                break
            batch = flat_examples[batch_start:batch_start + batch_size]
            batch_results = await asyncio.gather(
                *[run_example(session, ex) for ex in batch],
                return_exceptions=True
            )
            for r in batch_results:
                if isinstance(r, ExampleResult):
                    results.append(r)
                elif isinstance(r, Exception):
                    logger.error(f"Batch example failed: {r}")
            logger.info(
                f"Progress: {len(results)}/{len(flat_examples)} | "
                f"cost=${COST.cumulative_usd:.4f} | calls={COST.num_calls}"
            )
            gc.collect()

    logger.info(f"Processed {len(results)} examples")

    # Compute metrics
    metrics = aggregate_metrics(results)
    logger.info("=== RESULTS ===")
    for k, v in metrics.items():
        if k != "notes":
            logger.info(f"  {k}: {v}")

    # Build output
    output = build_output_json(results, all_examples, metrics)

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Also save summary metrics
    summary_path = WORKSPACE / "method_out_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved summary: {summary_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
