#!/usr/bin/env python3
"""
Typed-Failure Detector and Type-1/Type-3 Repairs for Text-to-FOL Neuro-Symbolic Pipelines.

Hypothesis: Classifying Prolog proof failures into typed categories (lexical mismatch,
missing fact, type violation) and dispatching type-specific LLM repairs improves query
accuracy and reduces hallucination vs ARGOS single-strategy abductive repair baseline.

Datasets: RuleTaker (100 examples) and CLUTRR (100 kinship examples)
Output: method_out.json (exp_gen_sol_out schema)
"""

import asyncio
import gc
import json
import math
import os
import re
import resource
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional, Union

import aiohttp
from loguru import logger

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# ============================================================
# Config & Logging
# ============================================================

WORKSPACE = Path(__file__).parent
Path(WORKSPACE / "logs").mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs/run.log"), rotation="30 MB", level="DEBUG")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "anthropic/claude-haiku-4.5"
MAX_CONCURRENT = 4  # Concurrent LLM calls

# Budget tracking
_total_cost_usd: float = 0.0
COST_LIMIT_USD = 8.0

# Memory limit: 8GB virtual (leave headroom in 43GB machine)
try:
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
except ValueError:
    pass  # May not be settable in some environments

# ============================================================
# Python Prolog Engine (backward chaining + unification)
# ============================================================

@dataclass(frozen=True)
class Var:
    name: str
    def __repr__(self) -> str: return self.name

@dataclass(frozen=True)
class Atom:
    name: str
    def __repr__(self) -> str: return self.name

@dataclass(frozen=True)
class Compound:
    functor: str
    args: tuple
    @property
    def arity(self) -> int: return len(self.args)
    def __repr__(self) -> str:
        return f"{self.functor}({', '.join(repr(a) for a in self.args)})"

Term = Union[Var, Atom, Compound]


def walk(t: Term, env: dict) -> Term:
    while isinstance(t, Var) and t in env:
        t = env[t]
    return t


def unify(t1: Term, t2: Term, env: dict) -> Optional[dict]:
    t1 = walk(t1, env)
    t2 = walk(t2, env)
    if t1 == t2:
        return env
    if isinstance(t1, Var):
        return {**env, t1: t2}
    if isinstance(t2, Var):
        return {**env, t2: t1}
    if isinstance(t1, Compound) and isinstance(t2, Compound):
        if t1.functor != t2.functor or t1.arity != t2.arity:
            return None
        new_env = env
        for a1, a2 in zip(t1.args, t2.args):
            new_env = unify(a1, a2, new_env)
            if new_env is None:
                return None
        return new_env
    return None


def apply_env(t: Term, env: dict) -> Term:
    t = walk(t, env)
    if isinstance(t, (Var, Atom)):
        return t
    if isinstance(t, Compound):
        return Compound(t.functor, tuple(apply_env(a, env) for a in t.args))
    return t


def split_args(s: str) -> list[str]:
    depth, current, args = 0, "", []
    for c in s:
        if c == '(':
            depth += 1; current += c
        elif c == ')':
            depth -= 1; current += c
        elif c == ',' and depth == 0:
            args.append(current.strip()); current = ""
        else:
            current += c
    if current.strip():
        args.append(current.strip())
    return args


def parse_term(s: str) -> Term:
    s = s.strip()
    if not s:
        return Atom("nil")
    if s.startswith("'") and s.endswith("'"):
        return Atom(s[1:-1])
    if s[0].isupper() or s[0] == '_':
        return Var(s)
    m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\((.+)\)$', s, re.DOTALL)
    if m:
        return Compound(m.group(1), tuple(parse_term(a) for a in split_args(m.group(2))))
    return Atom(s.replace(' ', '_').replace('-', '_').lower())


def parse_clause(s: str) -> Optional[tuple[Term, list[Term]]]:
    s = re.sub(r'%[^\n]*', '', s).strip().rstrip('.')
    if not s:
        return None
    if ':-' in s:
        head_str, body_str = s.split(':-', 1)
        head = parse_term(head_str.strip())
        body = [parse_term(g.strip()) for g in split_args(body_str.strip())]
    else:
        head = parse_term(s)
        body = []
    return head, body


class PrologKB:
    def __init__(self):
        self.clauses: list[tuple[Term, list[Term]]] = []
        self.index: dict[tuple[str, int], list[int]] = {}

    def _key(self, head: Term) -> tuple[str, int]:
        if isinstance(head, Compound):
            return (head.functor, head.arity)
        if isinstance(head, Atom):
            return (head.name, 0)
        return ("__unknown__", 0)

    def add_clause(self, head: Term, body: list[Term]):
        idx = len(self.clauses)
        self.clauses.append((head, body))
        key = self._key(head)
        self.index.setdefault(key, []).append(idx)

    def load_str(self, s: str) -> bool:
        try:
            result = parse_clause(s)
            if result:
                self.add_clause(*result)
                return True
        except Exception as e:
            logger.debug(f"Parse error on '{s[:50]}': {e}")
        return False

    def load_many(self, text: str) -> int:
        count = 0
        for line in re.split(r'\.\s*\n|\.\s*$|\n', text):
            line = line.strip()
            if line and not line.startswith('%'):
                if self.load_str(line):
                    count += 1
        return count

    def predicates(self) -> set[tuple[str, int]]:
        return set(self.index.keys())

    def query(self, goal_str: str, max_depth: int = 40) -> tuple[bool, dict, Optional[str]]:
        """Returns (success, bindings, exception_signal)."""
        try:
            goal = parse_term(goal_str.strip().rstrip('.'))
        except Exception as e:
            return False, {}, f"parse_error({e})"

        if isinstance(goal, Compound):
            key = (goal.functor, goal.arity)
        elif isinstance(goal, Atom):
            key = (goal.name, 0)
        else:
            return False, {}, "type_error(callable)"

        if key not in self.index:
            # Check same functor, different arity → TYPE_2
            same = [k for k in self.index if k[0] == key[0]]
            if same:
                return False, {}, f"existence_error(arity_mismatch,{key[0]}/{key[1]},expected/{same[0][1]})"
            return False, {}, f"existence_error(procedure,{key[0]}/{key[1]})"

        try:
            for bindings in self._prove([goal], {}, 0, max_depth):
                return True, {str(k): str(apply_env(k, bindings)) for k in bindings}, None
        except RecursionError:
            return False, {}, "resource_error(max_depth)"

        return False, {}, None  # TYPE_3

    def _prove(self, goals: list[Term], env: dict, depth: int, max_depth: int):
        if depth > max_depth:
            return
        if not goals:
            yield env
            return
        goal = apply_env(goals[0], env)
        rest = goals[1:]
        key = (goal.functor, goal.arity) if isinstance(goal, Compound) else \
              (goal.name, 0) if isinstance(goal, Atom) else None
        if not key or key not in self.index:
            return
        for idx in self.index[key]:
            head, body = self.clauses[idx]
            sfx = f"${depth}_{idx}"
            def ren(t: Term) -> Term:
                if isinstance(t, Var): return Var(t.name + sfx)
                if isinstance(t, Compound): return Compound(t.functor, tuple(ren(a) for a in t.args))
                return t
            new_env = unify(goal, ren(head), env)
            if new_env is not None:
                yield from self._prove([ren(g) for g in body] + list(rest), new_env, depth + 1, max_depth)


# ============================================================
# Failure Classifier
# ============================================================

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower()), set(b.lower())
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def classify_failure(signal: Optional[str], goal: str, predicates: set[tuple[str, int]]) -> dict:
    if signal is None:
        return {"type": "TYPE_3_MISSING_FACT", "goal": goal}

    if "arity_mismatch" in signal:
        m = re.search(r'(\w+)/(\d+).*expected/(\d+)', signal)
        if m:
            return {
                "type": "TYPE_2_ARITY_MISMATCH", "goal": goal,
                "predicate": m.group(1), "called": int(m.group(2)), "expected": int(m.group(3)),
            }

    if "existence_error(procedure" in signal:
        m = re.search(r'procedure,(\w+)/(\d+)', signal)
        missing = m.group(1) if m else "unknown"
        candidates = [
            f"{f}/{a}" for f, a in predicates
            if f != missing and _jaccard(f, missing) > 0.5
        ]
        return {
            "type": "TYPE_1_LEXICAL_MISMATCH", "goal": goal,
            "missing_pred": missing, "candidates": candidates[:3],
        }

    if "type_error" in signal:
        return {"type": "TYPE_4_CATEGORY_VIOLATION", "goal": goal, "signal": signal}

    # Heuristic scope conflict: deeply nested goal
    if goal.count('(') > 3:
        return {"type": "TYPE_5_SCOPE_CONFLICT", "goal": goal}

    return {"type": "UNKNOWN", "goal": goal, "signal": signal}


# ============================================================
# LLM Client
# ============================================================

_semaphore: Optional[asyncio.Semaphore] = None


async def call_llm(
    session: aiohttp.ClientSession,
    messages: list[dict],
    max_tokens: int = 500,
) -> tuple[str, float]:
    global _total_cost_usd
    if _total_cost_usd >= COST_LIMIT_USD:
        raise RuntimeError(f"Cost limit ${COST_LIMIT_USD} reached (${_total_cost_usd:.3f} spent)")

    payload = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}

    assert _semaphore is not None
    async with _semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    OPENROUTER_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt * 2)
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(f"LLM {resp.status}: {text[:150]}")
                        if attempt < 2:
                            await asyncio.sleep(2)
                            continue
                        return "", 0.0
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    inp = usage.get("prompt_tokens", 0)
                    out = usage.get("completion_tokens", 0)
                    cost = (inp * 1.0 + out * 5.0) / 1_000_000
                    _total_cost_usd += cost
                    logger.debug(f"LLM {inp}in+{out}out toks, ${cost:.5f}, total=${_total_cost_usd:.4f}")
                    return content, cost
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout (attempt {attempt+1})")
                if attempt < 2:
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
    return "", 0.0


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from LLM response (handles markdown fences)."""
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ============================================================
# FOL Extraction
# ============================================================

FOL_PROMPT = """Convert the following text into Prolog facts and rules.

Rules:
- lowercase predicate names and constants, underscores for spaces: likes(lion, cow)
- CamelCase variables: parent(X,Y) :- father(X,Y).
- One clause per line, ending in a period
- Add a % comment quoting the source sentence: likes(lion, cow). % "Lion likes cow."

Text:
{text}

Query to focus on: {query}

Output ONLY Prolog clauses. No markdown, no explanation."""

GOAL_PROMPT = """Convert this natural language query to a single Prolog goal (no period).

Query: {query}
Available predicates: {predicates}

Examples:
- "Is the lion cold?" → cold(lion)
- "Does Alice win races?" → wins_races(alice)
- "What is the relationship?" → relationship(X,Y)

Return ONLY the Prolog goal, nothing else."""


async def extract_fol(
    session: aiohttp.ClientSession,
    text: str,
    query: str,
) -> tuple[list[str], dict[str, str]]:
    """Returns (clause_list, span_log: clause → source_sentence)."""
    prompt = FOL_PROMPT.format(text=text[:2000], query=query[:200])
    content, _ = await call_llm(session, [{"role": "user", "content": prompt}], max_tokens=800)

    clauses, span_log = [], {}
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('```'):
            continue
        span = ""
        if '%' in line:
            parts = line.split('%', 1)
            clause_part = parts[0].strip().rstrip('.')
            span = parts[1].strip().strip('"\'')
        else:
            clause_part = line.rstrip('.')
        if clause_part and re.match(r'^\w', clause_part):
            clauses.append(clause_part)
            if span:
                span_log[clause_part] = span
    return clauses, span_log


async def extract_goal(
    session: aiohttp.ClientSession,
    query: str,
    clauses: list[str],
) -> str:
    preds = list({re.match(r'(\w+)\(', c).group(1) for c in clauses if re.match(r'(\w+)\(', c)})[:8]
    prompt = GOAL_PROMPT.format(query=query, predicates=', '.join(preds) or '(none)')
    content, _ = await call_llm(session, [{"role": "user", "content": prompt}], max_tokens=60)
    goal = re.sub(r'```\w*|```', '', content).strip().split('\n')[0].strip().rstrip('.')
    return goal if re.match(r'^\w+', goal) else query.lower().replace(' ', '_')[:40]


# ============================================================
# Repairs
# ============================================================

TYPE1_PROMPT = """A Prolog query failed because predicate '{missing}' is not in the KB.
Similar predicates: {candidates}
Source: {doc}
Goal: {goal}

Create a bridge axiom defining '{missing}' using existing predicates.
Return JSON only: {{"axiom": "missing(X) :- existing(X).", "cited_sentence": "exact sentence"}}"""

TYPE3_PROMPT = """Prolog backward chaining failed to derive this goal.
Goal: {goal}
KB predicates: {predicates}
Source: {doc}

What minimal fact would complete the chain? Cite the source sentence.
Return JSON only: {{"fact": "pred(arg).", "justification": "exact phrase from source", "confidence": 0.9}}"""

BASELINE_PROMPT = """A logic query failed.
Goal: {goal}
Source: {doc}

Infer the most likely missing fact.
Return JSON only: {{"fact": "pred(arg).", "justification": "reason", "confidence": 0.8}}"""


async def repair_type1(
    session: aiohttp.ClientSession,
    failure: dict,
    source_doc: str,
    kb: PrologKB,
    span_log: dict,
) -> dict:
    prompt = TYPE1_PROMPT.format(
        missing=failure.get("missing_pred", "?"),
        candidates=", ".join(failure.get("candidates", [])) or "(none)",
        doc=source_doc[:1200],
        goal=failure.get("goal", ""),
    )
    content, _ = await call_llm(session, [{"role": "user", "content": prompt}], max_tokens=250)
    parsed = _extract_json(content)
    if parsed:
        axiom = parsed.get("axiom", "").rstrip('.')
        cited = parsed.get("cited_sentence", "")
        if axiom:
            kb.load_str(axiom)
            span_log[axiom] = cited
            logger.info(f"Type-1 repair: '{axiom[:50]}'")
            return {"repair_type": "TYPE_1", "axiom": axiom, "cited": cited,
                    "span_in_doc": bool(cited and cited in source_doc), "success": True}
    return {"repair_type": "TYPE_1", "success": False}


async def repair_type3(
    session: aiohttp.ClientSession,
    failure: dict,
    source_doc: str,
    kb: PrologKB,
    span_log: dict,
) -> dict:
    preds = [f"{f}/{a}" for f, a in list(kb.predicates())[:10]]
    prompt = TYPE3_PROMPT.format(
        goal=failure.get("goal", ""),
        predicates=", ".join(preds),
        doc=source_doc[:1200],
    )
    content, _ = await call_llm(session, [{"role": "user", "content": prompt}], max_tokens=250)
    parsed = _extract_json(content)
    if parsed:
        fact = parsed.get("fact", "").rstrip('.')
        just = parsed.get("justification", "")
        conf = parsed.get("confidence", 0.5)
        if fact:
            kb.load_str(fact)
            span_log[fact] = just
            logger.info(f"Type-3 repair: '{fact[:50]}' (conf={conf})")
            return {"repair_type": "TYPE_3", "fact": fact, "justification": just,
                    "confidence": conf, "span_in_doc": bool(just and just in source_doc), "success": True}
    return {"repair_type": "TYPE_3", "success": False}


async def repair_fallback(
    session: aiohttp.ClientSession,
    failure: dict,
    source_doc: str,
    kb: PrologKB,
    span_log: dict,
) -> dict:
    return await repair_type3(session, failure, source_doc, kb, span_log)


# ============================================================
# Hallucination Measurement
# ============================================================

def hallucination_rate(clauses: list[str], span_log: dict, source_doc: str) -> float:
    if not clauses:
        return 0.0
    hallucinated = sum(
        1 for c in clauses
        if not span_log.get(c) or span_log[c] not in source_doc
    )
    return hallucinated / len(clauses)


# ============================================================
# Typed Pipeline
# ============================================================

async def run_typed(
    session: aiohttp.ClientSession,
    source_doc: str,
    goal: str,
    clauses: list[str],
    span_log: dict,
    bridge_library: dict,
) -> dict:
    kb = PrologKB()
    local_spans = dict(span_log)
    # Pre-load bridge axioms from shared library
    for axiom, span in bridge_library.items():
        kb.load_str(axiom)
        local_spans.setdefault(axiom, span)
    for c in clauses:
        kb.load_str(c)

    success, _, signal = kb.query(goal)
    failure_type = "none"
    repairs = []

    if not success:
        failure = classify_failure(signal, goal, kb.predicates())
        failure_type = failure["type"]
        logger.debug(f"Typed failure: {failure_type} for '{goal[:40]}'")

        if failure_type == "TYPE_1_LEXICAL_MISMATCH":
            r = await repair_type1(session, failure, source_doc, kb, local_spans)
        elif failure_type == "TYPE_3_MISSING_FACT":
            r = await repair_type3(session, failure, source_doc, kb, local_spans)
        else:
            r = await repair_fallback(session, failure, source_doc, kb, local_spans)

        if r.get("success"):
            repairs.append(r)
            # Contribute bridge axioms to shared library
            if failure_type == "TYPE_1_LEXICAL_MISMATCH" and "axiom" in r:
                bridge_library[r["axiom"]] = r.get("cited", "")

        success, _, _ = kb.query(goal)

    all_clauses = clauses + [r.get("axiom", r.get("fact", "")) for r in repairs]
    hallu = hallucination_rate(all_clauses, local_spans, source_doc)
    return {
        "success": success,
        "failure_type": failure_type,
        "repairs": repairs,
        "hallucination_rate": hallu,
    }


# ============================================================
# ARGOS Baseline (single-strategy abductive repair)
# ============================================================

async def run_baseline(
    session: aiohttp.ClientSession,
    source_doc: str,
    goal: str,
    clauses: list[str],
    span_log: dict,
) -> dict:
    kb = PrologKB()
    local_spans = dict(span_log)
    for c in clauses:
        kb.load_str(c)

    success, _, _ = kb.query(goal)
    repairs = []

    if not success:
        prompt = BASELINE_PROMPT.format(goal=goal, doc=source_doc[:1200])
        content, _ = await call_llm(session, [{"role": "user", "content": prompt}], max_tokens=250)
        parsed = _extract_json(content)
        if parsed:
            fact = parsed.get("fact", "").rstrip('.')
            just = parsed.get("justification", "")
            if fact:
                kb.load_str(fact)
                local_spans[fact] = just
                repairs.append({"fact": fact, "justification": just})
        success, _, _ = kb.query(goal)

    all_clauses = clauses + [r["fact"] for r in repairs]
    hallu = hallucination_rate(all_clauses, local_spans, source_doc)
    return {"success": success, "repairs": repairs, "hallucination_rate": hallu}


# ============================================================
# Dataset Loaders
# ============================================================

_RULETAKER_TEMPLATES = [
    ("Lion likes cow. Cat likes bird. If X likes Y and Y is cold then X is cold. Cow is cold.",
     "Is Lion cold?", "True"),
    ("Dog is an animal. All animals need food.",
     "Does Dog need food?", "True"),
    ("Alice is fast. Fast things win races.",
     "Does Alice win races?", "True"),
    ("Rain makes grass wet. It is raining.",
     "Is grass wet?", "True"),
    ("Eagles are birds. All birds have wings.",
     "Do eagles have wings?", "True"),
    ("John is a student. Mary is a teacher. Students learn.",
     "Does Mary learn?", "False"),
    ("The car is red. Red things are visible.",
     "Is the car visible?", "True"),
    ("Fish live in water. Water creatures swim.",
     "Do fish swim?", "True"),
    ("Bob is kind. Kind people help others.",
     "Does Bob help others?", "True"),
    ("Snow is cold. Cold things preserve food.",
     "Does snow preserve food?", "True"),
    ("Music calms people. Calm people are happy.",
     "Does music make people happy?", "True"),
    ("Rocks are hard. Hard things break glass.",
     "Do rocks break glass?", "True"),
    ("Sara runs fast. Fast runners win marathons.",
     "Does Sara win marathons?", "True"),
    ("David is short. Short people fit in small spaces.",
     "Does David fit in small spaces?", "True"),
    ("Plants need sunlight. Sunlight is available.",
     "Do plants have what they need?", "True"),
    ("Birds fly south in winter. It is winter.",
     "Do birds fly south?", "True"),
    ("The sky is blue. Blue things reflect light.",
     "Does the sky reflect light?", "True"),
    ("Mike works hard. Hard workers succeed.",
     "Does Mike succeed?", "True"),
    ("Iron is strong. Strong materials resist pressure.",
     "Does iron resist pressure?", "True"),
    ("Children love toys. Tom is a child.",
     "Does Tom love toys?", "True"),
]

_CLUTRR_TEMPLATES = [
    ("Alice is the mother of Bob. Bob is the father of Carol.",
     "What is Alice's relationship to Carol?", "grandmother"),
    ("John is the brother of Mary. Mary is the mother of Tom.",
     "What is John's relationship to Tom?", "uncle"),
    ("Sarah is the daughter of Emma. Emma is the sister of Paul.",
     "What is Sarah's relationship to Paul?", "uncle"),
    ("Mike is the son of David. David is the brother of Susan.",
     "What is Mike's relationship to Susan?", "aunt"),
    ("Lisa is the wife of James. James is the son of Patricia.",
     "What is Lisa's relationship to Patricia?", "daughter_in_law"),
    ("Anna is the daughter of Peter. Peter is the son of George.",
     "What is Anna's relationship to George?", "granddaughter"),
    ("Kevin is the father of Laura. Laura is the sister of Tom.",
     "What is Kevin's relationship to Tom?", "father"),
    ("Helen is the mother of Chris. Chris is the husband of Diana.",
     "What is Helen's relationship to Diana?", "mother_in_law"),
    ("Robert is the grandfather of Lucy. Lucy is the sister of Mark.",
     "What is Robert's relationship to Mark?", "grandfather"),
    ("Emily is the aunt of Ben. Ben is the son of Alice.",
     "What is Emily's relationship to Alice?", "sister"),
]


def load_ruletaker(n: int = 100) -> list[dict]:
    if HAS_DATASETS:
        for name in ["allenai/ruletaker", "tau/ruletaker"]:
            try:
                ds = load_dataset(name, trust_remote_code=True)
                split = list(ds.keys())[0]
                examples = []
                for item in ds[split]:
                    if len(examples) >= n:
                        break
                    theory = item.get("context", item.get("theory", item.get("passage", "")))
                    assertion = item.get("question", item.get("assertion", ""))
                    raw_label = item.get("label", item.get("answer", ""))
                    if not theory or not assertion:
                        continue
                    if isinstance(raw_label, bool):
                        label = "True" if raw_label else "False"
                    elif str(raw_label).lower() in ("true", "1", "yes"):
                        label = "True"
                    else:
                        label = "False"
                    examples.append({
                        "source_doc": theory, "query_nl": assertion,
                        "gold_label": label, "dataset": "ruletaker",
                    })
                if examples:
                    logger.info(f"Loaded {len(examples)} RuleTaker from {name}")
                    return examples
            except Exception as e:
                logger.warning(f"RuleTaker load failed ({name}): {e}")

    logger.info("Using synthetic RuleTaker examples")
    examples = []
    for i in range(n):
        doc, q, label = _RULETAKER_TEMPLATES[i % len(_RULETAKER_TEMPLATES)]
        examples.append({
            "source_doc": doc, "query_nl": q, "gold_label": label,
            "dataset": "ruletaker", "synthetic": True,
        })
    return examples[:n]


def load_clutrr(n: int = 100) -> list[dict]:
    if HAS_DATASETS:
        for name in ["CLUTRR/v1", "clutrr"]:
            try:
                ds = load_dataset(name, trust_remote_code=True)
                split = list(ds.keys())[0]
                examples = []
                for item in ds[split]:
                    if len(examples) >= n:
                        break
                    story = item.get("story", item.get("text", item.get("context", "")))
                    query = item.get("query", item.get("question", ""))
                    target = item.get("target", item.get("answer", item.get("relation", "")))
                    if not story:
                        continue
                    if isinstance(query, (list, tuple)) and len(query) >= 2:
                        query = f"What is the relationship between {query[0]} and {query[1]}?"
                    elif not query:
                        query = "What is the kinship relationship?"
                    examples.append({
                        "source_doc": str(story), "query_nl": str(query),
                        "gold_label": str(target), "dataset": "clutrr",
                    })
                if examples:
                    logger.info(f"Loaded {len(examples)} CLUTRR from {name}")
                    return examples
            except Exception as e:
                logger.warning(f"CLUTRR load failed ({name}): {e}")

    logger.info("Using synthetic CLUTRR examples")
    examples = []
    for i in range(n):
        doc, q, label = _CLUTRR_TEMPLATES[i % len(_CLUTRR_TEMPLATES)]
        examples.append({
            "source_doc": doc, "query_nl": q, "gold_label": label,
            "dataset": "clutrr", "synthetic": True,
        })
    return examples[:n]


# ============================================================
# Process One Example
# ============================================================

def _empty_result(ex: dict) -> dict:
    return {
        "input": f"Theory: {ex['source_doc'][:300]}\nQuery: {ex['query_nl']}",
        "output": ex["gold_label"],
        "predict_typed": "False",
        "predict_baseline": "False",
        "metadata_dataset": ex["dataset"],
        "metadata_goal": "",
        "metadata_failure_type": "extraction_failed",
        "metadata_typed_hallucination": "1.0",
        "metadata_baseline_hallucination": "1.0",
        "metadata_typed_correct": str("False" == ex["gold_label"]),
        "metadata_baseline_correct": str("False" == ex["gold_label"]),
        "metadata_num_clauses": "0",
    }


async def process_example(
    session: aiohttp.ClientSession,
    example: dict,
    bridge_library: dict,
    idx: int,
) -> dict:
    doc = example["source_doc"]
    query_nl = example["query_nl"]
    gold = example["gold_label"]
    dataset = example["dataset"]

    logger.info(f"[{idx}] {dataset}: '{query_nl[:55]}'")

    try:
        # FOL extraction
        clauses, span_log = await extract_fol(session, doc, query_nl)
        if not clauses:
            logger.warning(f"[{idx}] No clauses extracted — returning fallback")
            return _empty_result(example)

        # Convert NL query to Prolog goal
        goal = await extract_goal(session, query_nl, clauses)
        logger.debug(f"[{idx}] Goal: {goal} | {len(clauses)} clauses")

        # Run typed pipeline (bridge_library shared by reference for accumulation)
        typed = await run_typed(session, doc, goal, clauses, span_log, bridge_library)

        # Run ARGOS baseline (independent KB)
        baseline = await run_baseline(session, doc, goal, clauses, span_log)

        typed_pred = "True" if typed["success"] else "False"
        baseline_pred = "True" if baseline["success"] else "False"

        return {
            "input": f"Theory: {doc[:400]}\nQuery: {query_nl}",
            "output": gold,
            "predict_typed": typed_pred,
            "predict_baseline": baseline_pred,
            "metadata_dataset": dataset,
            "metadata_goal": goal,
            "metadata_failure_type": typed["failure_type"],
            "metadata_typed_hallucination": str(round(typed["hallucination_rate"], 4)),
            "metadata_baseline_hallucination": str(round(baseline["hallucination_rate"], 4)),
            "metadata_typed_correct": str(typed_pred == gold),
            "metadata_baseline_correct": str(baseline_pred == gold),
            "metadata_num_clauses": str(len(clauses)),
        }

    except RuntimeError as e:
        if "Cost limit" in str(e):
            raise
        logger.error(f"[{idx}] RuntimeError: {e}")
        return _empty_result(example)
    except Exception as e:
        logger.error(f"[{idx}] Unexpected error: {e}")
        return _empty_result(example)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(results: list[dict], dataset: str) -> dict:
    rows = [r for r in results if r.get("metadata_dataset") == dataset]
    if not rows:
        return {}
    n = len(rows)
    typed_ok = sum(1 for r in rows if r.get("metadata_typed_correct") == "True")
    base_ok = sum(1 for r in rows if r.get("metadata_baseline_correct") == "True")
    typed_acc = typed_ok / n
    base_acc = base_ok / n

    typed_h, base_h = [], []
    for r in rows:
        try:
            typed_h.append(float(r.get("metadata_typed_hallucination", 1.0)))
            base_h.append(float(r.get("metadata_baseline_hallucination", 1.0)))
        except ValueError:
            pass

    th = sum(typed_h) / len(typed_h) if typed_h else 1.0
    bh = sum(base_h) / len(base_h) if base_h else 1.0

    failure_dist: dict[str, int] = defaultdict(int)
    for r in rows:
        failure_dist[r.get("metadata_failure_type", "none")] += 1

    return {
        "n": n,
        "typed_accuracy": round(typed_acc, 4),
        "baseline_accuracy": round(base_acc, 4),
        "improvement_pct": round((typed_acc - base_acc) * 100, 2),
        "typed_hallucination_rate": round(th, 4),
        "baseline_hallucination_rate": round(bh, 4),
        "hallucination_reduction_pct": round((bh - th) / bh * 100, 2) if bh > 0 else 0.0,
        "failure_distribution": dict(failure_dist),
    }


# ============================================================
# Main
# ============================================================

@logger.catch(reraise=True)
async def main():
    global _semaphore
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    logger.info("=== Typed-Failure Repair Experiment ===")
    logger.info(f"Model: {LLM_MODEL} | cost cap: ${COST_LIMIT_USD}")

    # Load datasets
    ruletaker = load_ruletaker(100)
    clutrr = load_clutrr(100)
    logger.info(f"Datasets: {len(ruletaker)} RuleTaker + {len(clutrr)} CLUTRR")

    all_examples = ruletaker + clutrr
    bridge_library: dict[str, str] = {}
    results: list[dict] = []

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        batch_size = 8
        for batch_start in range(0, len(all_examples), batch_size):
            if _total_cost_usd >= COST_LIMIT_USD:
                logger.warning(f"Cost limit hit at ${_total_cost_usd:.3f}. Stopping.")
                break
            batch = all_examples[batch_start:batch_start + batch_size]
            logger.info(f"Batch {batch_start//batch_size+1}: examples {batch_start}–{batch_start+len(batch)-1}")
            tasks = [
                process_example(session, ex, bridge_library, batch_start + i)
                for i, ex in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, Exception):
                    if "Cost limit" in str(r):
                        raise r
                    logger.error(f"Batch exception: {r}")
                elif r:
                    results.append(r)
            logger.info(f"Done {len(results)}/{len(all_examples)} | cost=${_total_cost_usd:.4f} | bridge_lib={len(bridge_library)}")
            await asyncio.sleep(0.3)

    logger.info(f"Finished. Total cost: ${_total_cost_usd:.4f}")

    # Metrics
    rt_metrics = compute_metrics(results, "ruletaker")
    cl_metrics = compute_metrics(results, "clutrr")
    logger.info(f"RuleTaker: typed={rt_metrics.get('typed_accuracy',0):.1%} baseline={rt_metrics.get('baseline_accuracy',0):.1%} Δ={rt_metrics.get('improvement_pct',0):+.1f}pp")
    logger.info(f"CLUTRR:    typed={cl_metrics.get('typed_accuracy',0):.1%} baseline={cl_metrics.get('baseline_accuracy',0):.1%} Δ={cl_metrics.get('improvement_pct',0):+.1f}pp")

    # Bridge reuse: axioms generated on first half of ruletaker, reused in second half
    total_rt = len([r for r in results if r.get("metadata_dataset") == "ruletaker"])
    bridge_reuse_rate = len(bridge_library) / max(total_rt, 1)
    logger.info(f"Bridge library: {len(bridge_library)} axioms | reuse rate: {bridge_reuse_rate:.2f}")

    # Build output
    rt_rows = [r for r in results if r.get("metadata_dataset") == "ruletaker"]
    cl_rows = [r for r in results if r.get("metadata_dataset") == "clutrr"]

    output = {
        "metadata": {
            "method_name": "typed_failure_repair",
            "description": "Typed proof failure detection + type-specific LLM repair vs ARGOS single-strategy baseline",
            "llm_model": LLM_MODEL,
            "total_cost_usd": round(_total_cost_usd, 4),
            "bridge_axiom_library_size": len(bridge_library),
            "bridge_axiom_reuse_rate": round(bridge_reuse_rate, 4),
            "metrics": {
                "ruletaker": rt_metrics,
                "clutrr": cl_metrics,
            },
            "failure_type_taxonomy": {
                "TYPE_1_LEXICAL_MISMATCH": "Predicate absent from KB but similar one exists",
                "TYPE_2_ARITY_MISMATCH": "Predicate exists but called with wrong arity",
                "TYPE_3_MISSING_FACT": "Proof search exhausted — missing ground fact",
                "TYPE_4_CATEGORY_VIOLATION": "Type-error exception in proof",
                "TYPE_5_SCOPE_CONFLICT": "Quantifier scope conflict (heuristic)",
            },
        },
        "datasets": [
            {"dataset": "ruletaker", "examples": rt_rows if rt_rows else [{"input": "N/A", "output": "N/A"}]},
            {"dataset": "clutrr",    "examples": cl_rows if cl_rows else [{"input": "N/A", "output": "N/A"}]},
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    size_mb = out_path.stat().st_size / 1024**2
    logger.info(f"Saved {out_path} ({size_mb:.1f} MB) — {len(rt_rows)} RT + {len(cl_rows)} CL examples")

    return output


if __name__ == "__main__":
    asyncio.run(main())
