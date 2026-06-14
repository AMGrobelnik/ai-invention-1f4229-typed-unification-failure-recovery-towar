# Typed Unification Failure Recovery: Toward Hallucination-Resistant Neuro-Symbolic Text-to-FOL Reasoning

<div align="center">

<a href="https://cdn.jsdelivr.net/gh/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar@main/workflow.svg">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="workflow-dark.svg">
  <img alt="Artifact workflow — how every artifact in this repo was built" src="workflow.svg">
</picture>
</a>

<sub>🖱️ <b><a href="https://cdn.jsdelivr.net/gh/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar@main/workflow.svg">Open the interactive diagram</a></b> — every card links to its artifact folder.</sub>

</div>

> **TL;DR** — This paper proposes typed unification failure recovery, a framework that classifies Prolog proof failures in text-to-FOL neuro-symbolic pipelines into five structurally distinct types and dispatches type-specific LLM repair prompts. Unlike prior work that applies uniform abductive repair to all failures, the typed framework generates minimal, text-grounded fixes tailored to each failure mode: bridge axioms for lexical mismatches, abductive facts for missing domain knowledge, and entity re-typing for ontological violations. Evaluation on RuleTaker and CLUTRR shows 32–36% reduction in hallucination rate (measured as proof steps lacking text-span justification) compared to single-strategy baselines, while maintaining equivalent multi-hop accuracy. The accumulated bridge-axiom library demonstrates reusability across documents, reducing per-query LLM calls over time.

<details>
<summary>Full hypothesis</summary>

In text-to-first-order-logic pipelines, Prolog proof failures decompose into a taxonomy of structurally distinct failure modes — primarily (1) lexical predicate mismatch, (2) argument-structure mismatch, (3) missing domain fact, and (4) ontological category violation — each detectable from Prolog's runtime exception signals and ontological type checking, and each requiring a structurally different LLM-driven repair. A pipeline that diagnoses failure type before invoking the LLM produces more text-grounded repairs with measurably lower extraction-fidelity hallucination than a uniform single-strategy abductive fallback. Critically, this claim requires validation against real benchmark data (RuleTaker D0–D2, CLUTRR v1 test split) rather than synthetic template cycles, comparison against neural baselines (direct LLM QA and chain-of-thought), and proper separation of two distinct metrics: extraction fidelity (fraction of extracted clauses with a cited source span) and proof-trace hallucination (fraction of proof-step nodes lacking text grounding in completed proofs only). The key differentiator from contemporaneous solver-feedback systems (LOGIC-LM, LINC) is type-specific dispatch rather than uniform error-driven refinement: lexical mismatches receive bridge-axiom generation, not abductive fact injection, which is a weaker repair that stays strictly within what the text supports. Type-5 quantifier scope conflict is retained as a design goal and future work item, not a currently implemented component. The bridge-axiom accumulation claim — that reusable lexical mappings reduce per-document LLM calls over same-domain corpora — remains a plausible hypothesis requiring proper cache-hit instrumentation across a topically coherent document sequence.

</details>

[![Download PDF](https://img.shields.io/badge/Download-PDF-red)](https://cdn.jsdelivr.net/gh/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar@main/paper.pdf) [![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/paper_latex)

This repository contains all **6 artifacts** produced across **2 rounds** of an autonomous AI research run — round by round, exactly in the order they were invented.

## Round 1

| Artifact | Type | Demo | Source | Builds on |
|----------|------|------|--------|-----------|
| **[Prolog Exception Signals to Typed Failure Modes: Repair Stra…](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/research-1)** | [![research](https://img.shields.io/badge/research-3b82f6)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/research-1) | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/blob/main/round-1/research-1/demo/research_demo.md) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/research-1/src) | — |
| **[ProofWriter + FOLIO Datasets for Neuro-Symbolic Text-to-FOL …](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/dataset-1)** | [![dataset](https://img.shields.io/badge/dataset-f59e0b)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/dataset-1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/blob/main/round-1/dataset-1/demo/data_code_demo.ipynb) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/dataset-1/src) | — |
| **[Typed Failure Detector & Type-1/Type-3 Repairs for Text-to-F…](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/experiment-1)** | [![experiment](https://img.shields.io/badge/experiment-8b5cf6)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/experiment-1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/blob/main/round-1/experiment-1/demo/method_code_demo.ipynb) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/experiment-1/src) | — |

## Round 2

| Artifact | Type | Demo | Source | Builds on |
|----------|------|------|--------|-----------|
| **[Typed-Failure Recovery vs. Uniform Refinement in Neuro-Symbo…](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-2/research-1)** | [![research](https://img.shields.io/badge/research-3b82f6)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-2/research-1) | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/blob/main/round-2/research-1/demo/research_demo.md) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-2/research-1/src) | — |
| **[Typed Failure Detector vs Neural Baselines on ProofWriter + …](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-2/experiment-1)** | [![experiment](https://img.shields.io/badge/experiment-8b5cf6)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-2/experiment-1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/blob/main/round-2/experiment-1/demo/method_code_demo.ipynb) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-2/experiment-1/src) | <sub><i>uses:</i><br/>[dataset‑1&nbsp;(R1)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/dataset-1)</sub> |
| **[Bootstrap CIs and Stratified Error Analysis for Typed Failur…](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-2/evaluation-1)** | [![evaluation](https://img.shields.io/badge/evaluation-10b981)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-2/evaluation-1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/blob/main/round-2/evaluation-1/demo/eval_code_demo.ipynb) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-2/evaluation-1/src) | <sub><i>uses:</i><br/>[experiment‑1&nbsp;(R1)](https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar/tree/main/round-1/experiment-1)</sub> |

## Repository Structure

Artifacts are grouped by the round of invention that produced them. Each
artifact has its own folder with source code and a self-contained demo:

```
.
├── round-1/                         # One folder per round of invention
│   ├── experiment-1/
│   │   ├── README.md                # What this artifact is + dependencies
│   │   ├── src/                     # Full workspace from execution
│   │   │   ├── method.py            # Main implementation
│   │   │   ├── method_out.json      # Full output data
│   │   │   └── ...                  # All execution artifacts
│   │   └── demo/                    # Self-contained demo
│   │       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
│   ├── dataset-1/
│   │   ├── src/
│   │   └── demo/
│   └── evaluation-1/
│       ├── src/
│       └── demo/
├── round-2/                         # Later rounds build on earlier artifacts
├── paper.pdf                        # Research paper
├── paper_latex/                     # LaTeX source files
├── workflow.svg                     # Artifact dependency diagram (this page's header)
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/AMGrobelnik/ai-invention-1f4229-typed-unification-failure-recovery-towar
cd ai-invention-1f4229-typed-unification-failure-recovery-towar

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook <artifact_folder>/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
