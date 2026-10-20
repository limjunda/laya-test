# Laya (Jev-Compatible System 1 Decision Model) — Insurance Showcase & Benchmark

This repository showcases **[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)** (latest `laya` SDK), an open-source, **Jev-compatible non-autoregressive System 1 decision engine** trained with **Reinforcement Learning for Calibrated Decisions (RLCD)**.

## Overview: What is Laya / Jev?

* **Jev** (introduced by TypeSafe AI) and **Laya** (`convaiinnovations/laya`, Apache 2.0) are **System 1 non-autoregressive decision models**.
* Instead of generating open-ended tokens like traditional LLMs—which can hallucinate or break JSON schemas—Laya takes a **state** (text, email, FNOL report, or JSON payload) and **typed questions**, returning calibrated probabilities and structured decisions in a **single forward pass**:
  * **`choice`**: Pick one option from a defined set of labels/criteria with calibrated probabilities.
  * **`score`**: Evaluate an ordered numerical scale (`[0, K-1]`) with expected value and level probabilities.
  * **`noul`**: Evaluate a binary (`Yes`/`No`) condition returning calibrated $P(\text{true})$.

| Checkpoint | Backbone | Parameters | Context | Best For |
|---|---|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large | 421M | 512 | English routing, claims triage, guardrails |
| `convaiinnovations/laya-multilingual` | mmBERT-base | 322M | 1,024 (up to 8,192) | 100+ languages, long documents |
| `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 421M | 1,024 | Fine-tuned typed-decisions workflows |

---

## Repository Structure

```text
laya-test/
├── laya_insurance_showcase.ipynb   # Self-contained Jupyter Notebook (model download, showcase, & benchmarks)
├── data/
│   ├── insurance_test_cases.json   # 75 realistic insurance test cases with ground-truth labels
│   └── benchmark_results.png       # Generated benchmark comparison chart (Row-by-Row vs. Batch)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

---

## Insurance Benchmark Dataset (`75 Test Cases / 225 Decisions`)

The benchmark dataset ([`data/insurance_test_cases.json`](data/insurance_test_cases.json) and embedded directly inside [`laya_insurance_showcase.ipynb`](laya_insurance_showcase.ipynb)) contains **75 realistic insurance claims and policyholder inquiries** across 6 core departments:

1. **`auto_claims`**: Vehicle collisions, windshield damage, catalytic converter theft, roadside towing, total loss disputes.
2. **`property_claims`**: Burst pipes, kitchen fires, hail/tornado roof damage, sewer backups, apartment burglary.
3. **`health_medical`**: Prior authorizations, emergency room claim appeals, insulin refill overrides, dental/vision benefits.
4. **`life_disability`**: Life insurance beneficiary death claims, terminal illness accelerated benefits, short/long-term disability, AD&D.
5. **`billing_policy`**: Duplicate premium drafts, escrow billing collisions, rate hike complaints, policy discounts, vehicle/mortgagee changes.
6. **`fraud_siu`**: Special Investigations Unit (SIU) referrals—staged accidents, phantom medical billing, doctored tow invoices, arson indicators, chameleon carriers.

Each test case evaluates **3 typed questions simultaneously** (`225` total decisions per test pass):
- `department` (`choice` across the 6 insurance queues)
- `urgency` (`score` on a `0` = Routine, `1` = Standard, `2` = Critical scale)
- `churn_or_legal_risk` (`noul` evaluating whether the sender threatens policy cancellation, switching carriers, legal action, or regulatory complaints)

---

## Benchmark Results (`75 Insurance Cases / 225 Typed Decisions`)

![Benchmark Results](data/benchmark_results.png)

### 1. Execution Speed: Row-by-Row vs. Batch Processing

| Test Run | Method | Total Time (s) | Latency / Case (ms) | Latency / Decision (ms) | Throughput (Decisions/s) | Speedup |
|---|---|---:|---:|---:|---:|---:|
| **Test #1** | `agent.predict()` (Row-by-Row) | `42.294 s` | `563.91 ms` | `187.97 ms` | `5.32 dec/s` | `1.00x` |
| **Test #2a** | `agent.predict_batch(batch_size=8, sort_by_length=False)` | `39.181 s` | `522.41 ms` | `174.14 ms` | `5.74 dec/s` | `1.08x` |
| **Test #2b** | `agent.predict_batch(batch_size=8, sort_by_length=True)` | **`36.947 s`** | **`492.62 ms`** | **`164.21 ms`** | **`6.09 dec/s`** | **`1.14x`** |
| **Test #2c** | `agent.predict_batch(batch_size=16, sort_by_length=True)` | `37.000 s` | `493.33 ms` | `164.44 ms` | `6.08 dec/s` | `1.14x` |

* **100.0% Decision Parity**: Row-by-row and batched execution produced **75 / 75 (100.0%)** identical department choices, with negligible floating-point drift (`<= 0.0174`).

### 2. Zero-Shot Insurance Decision Accuracy (`convaiinnovations/laya`)

* **Overall 6-Class Department Routing Accuracy**: **76.00% (57 / 75)** (and **87.5% [56 / 64]** across the 5 standard lines of business: Auto `100.0%`, Property `100.0%`, Life/Disability `90.9%`, Billing/Policy `83.3%`, Health/Medical `61.5%`).
* **Churn / Legal Escalation (`noul`) Accuracy**: **81.33% (61 / 75)**.
* **Urgency Score (`score`, 0–2 scale)**: **MAE = 0.591**, Pearson correlation $r = 0.548$ with ground-truth urgency tiers.

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Jupyter Notebook

```bash
jupyter notebook laya_insurance_showcase.ipynb
```

The notebook is completely self-contained:
- Installs required packages if missing
- Downloads `convaiinnovations/laya` from Hugging Face Hub
- Demonstrates single-pass `agent.predict()`, schema-driven `agent.decide()`, confidence gating (`min_confidence`), and multilingual `Router`
- Runs **Test #1 (Row-by-Row Processing)** via `agent.predict()`
- Runs **Test #2 (Batch Processing)** via `agent.predict_batch(..., sort_by_length=True)`
- Compares wall-clock latency, throughput speedup, decision parity, and zero-shot accuracy against ground truth.
