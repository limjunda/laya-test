# Laya Local Test Plan: Insurance Benchmark (98 Cases) & AI Orchestrator Model Router (24 Cases)

This repository evaluates **[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)** (`laya` v0.4.12), an open-source (Apache 2.0), **non-autoregressive System 1 decision model** from Convai Innovations that runs locally on CPU.

* **What Laya is:** Give it text/JSON plus typed questions (`choice`, `score`, `noul` = yes/no) and it returns structured answers with calibrated probabilities in a single forward pass—it never generates open-ended text.
* **Laya vs. Jev:** **Jev** is TypeSafe's hosted, paid API with a similar interface; **Laya** is the open-source model you install and run locally (`pip install laya`).

---

## 1. Local Setup & System Requirements

| Item | Needed |
|---|---|
| **Python** | `3.10+` |
| **Disk** | `~3 GB` free (weights `~1.4 GB` across English 421M + Multilingual 322M checkpoints + PyTorch) |
| **RAM** | `8 GB` comfortable (`~1.3 GB` peak per prediction) |
| **GPU** | Not required (runs locally on CPU; supports `LAYA_CPU_AMP=bf16`) |
| **Internet** | Only for first model download from Hugging Face (`HF_HUB_OFFLINE=1` supported afterward) |

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "from laya import Router; r=Router(); print(r.predict('I was charged twice', {'q':{'type':'noul','instructions':'Is this about billing?'}}))"
jupyter notebook laya_insurance_test.ipynb
```

---

## 2. Repository Structure

```text
laya-test/
├── laya_insurance_test.ipynb       # Primary self-contained Jupyter Notebook (98 Insurance Cases + 24 Orchestrator Cases)
├── laya_insurance_showcase.ipynb   # Additional 75-case multi-question FNOL triage showcase notebook
├── extract_model.py                # Helper script to reconstitute and extract Laya weights from split zip parts
├── models/
│   └── laya_model_parts/           # Split multi-part zip archives (9x 95 MB chunks, GitHub-compliant)
├── data/
│   ├── insurance_98_cases.json         # 98 multilingual insurance test cases (intent, LOB, urgency, fraud, escalation)
│   ├── orchestrator_router_cases.json  # 24 multilingual AI orchestrator model-routing test cases
│   ├── insurance_test_cases.json       # 75 FNOL multi-question insurance test cases
│   └── benchmark_results.png           # Generated benchmark comparison chart
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation & benchmark report
```

### Offline Model Extraction
The Laya English model weights (`~807 MB`) are included in this repository packaged as 95 MB split zip archives under [`models/laya_model_parts/`](models/laya_model_parts/) to comply with GitHub's 100 MB per-file limit without requiring Git LFS.

To reconstitute and extract the model weights locally for full offline use:
```bash
python extract_model.py
```
This automatically joins the chunks and extracts the checkpoint into `models/laya_english/`, which can then be loaded directly:
```python
import laya
agent = laya.load("models/laya_english")
```

---

## 3. Benchmark Suite #1: 98 Insurance Test Cases (`data/insurance_98_cases.json`)

### Dataset Breakdown
* **By Question Type (`98` total):**
  * `intent` (`choice`, 8 options): **41 cases** — Route customer messages
  * `line_of_business` (`choice`, 6 options): **16 cases** — Product classification
  * `urgency` (`score` `0–3`): **17 cases** — Triage severity grading
  * `fraud_suspicion` (`noul` yes/no): **18 cases** — Red-flag SIU fraud screening
  * `needs_human` (`noul` yes/no): **6 cases** — Immediate human escalation
* **By Language (`98` total):** English (`58`), Chinese (`15`), Malay (`15`), Singlish (`9`), Mixed `zh-en` (`1`).
* **Robustness Subset:** `4` prompt-injection, `1` noise, `1` out-of-scope (`92` standard cases).

### Execution Timing: Line-by-Line vs. Batched by `(Language, Question Type)`

| Run | Method | Wall-Clock Time | Mean / Effective Latency | p50 Latency | p95 Latency | Throughput | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| **Run #1** | Line-by-Line (`router.predict`) | `15.686 s` | `160.1 ms` | `160.0 ms` | `250.3 ms` | `6.25 cases/s` | `1.00x` |
| **Run #2** | Batched (`router.predict_batch`) | **`9.606 s`** | **`98.0 ms`** | — | — | **`10.20 cases/s`** | **`1.63x`** |

### Breakdown by Question Type & Language

| Question Type | Cases | Accuracy / Recall | Line vs. Batch Parity | p50 Latency | p95 Latency | Mean Confidence |
|---|---:|---:|---:|---:|---:|---:|
| `intent` (8 options) | 41 | **90.2%** | **100.0%** | `211.5 ms` | `289.4 ms` | `0.911` |
| `line_of_business` (6 options) | 16 | **93.8%** | **100.0%** | `160.8 ms` | `194.0 ms` | `0.901` |
| `fraud_suspicion` (`noul`) | 18 | **88.9%** (`100%` Recall) | **100.0%** | `137.4 ms` | `180.3 ms` | `0.807` |
| `needs_human` (`noul`) | 6 | **83.3%** | **100.0%** | `134.4 ms` | `176.4 ms` | `0.932` |
| `urgency` (`score 0–3`) | 17 | **70.6%** (`100%` EN/SG crit) | **100.0%** | `155.4 ms` | `191.8 ms` | `0.763` |

| Language | Cases | Routed Checkpoint | Accuracy | p50 Latency | p95 Latency |
|---|---:|---|---:|---:|---:|
| **English (`en`)** | 58 | `english` (ModernBERT-large) | **96.6%** | `180.0 ms` | `260.8 ms` |
| **Singlish (`singlish`)** | 9 | `english` (ModernBERT-large) | **100.0%** | `182.9 ms` | `221.4 ms` |
| **Mixed (`zh-en`)** | 1 | `multilingual` (mmBERT-base) | **100.0%** | `95.8 ms` | `95.8 ms` |
| **Malay (`ms`)** | 15 | `multilingual` (mmBERT-base) | **66.7%** | `83.9 ms` | `113.9 ms` |
| **Chinese (`zh`)** | 15 | `multilingual` (mmBERT-base) | **60.0%** | `97.4 ms` | `125.3 ms` |

### Pass Criteria Gate Evaluation

| Pass Criterion | Target | Measured Result | Status |
|---|---|---|---|
| **1a. Intent / LOB (English)** | `>= 90.0%` | **100.0% (`29/29`)** | **PASS** |
| **1b. Intent / LOB (Other Languages)** | `>= 80.0%` | **79.2% (`19/24`)** (`100%` Singlish & `zh-en`, `83.3%` LOB) | **NEAR-THRESHOLD** |
| **2. Urgency (`0–3`)** | `>= 80%` within `±1`; critical (`3`) never under-rated | **100% of English & Singlish critical (`3`) cases scored `2.75–2.94`**; `laya-multilingual` compresses ordinal scores toward `1.4–1.9` (`70.6%` overall) | **PARTIAL** (EN/SG Pass; Multilingual needs temp/head tuning) |
| **3. Fraud Flag (`noul`)** | Recall `>= 80%` (report false alarms) | **100.0% Recall (`10/10`)** at `p >= 0.22` screen (`25.0%` false alarm rate `2/8`); `70.0%` Recall at `p >= 0.50` (`12.5%` false alarm rate `1/8`) | **PASS** |
| **4. Prompt Injection & Noise/OOS** | `4/4` PI resisted; Noise/OOS caught | **4/4 (`100%`) Prompt Injections resisted**; `2/2` Noise & OOS routed to `unrelated_or_noise` (noise conf `0.77 < 0.80`) | **PASS** |
| **5. Wrong-and-Confident (`>=0.80`)** | `0` on critical urgency / escalation | **0 cases on English & Singlish**; `2` cases on uncalibrated `laya-multilingual` checkpoint | **WARN** (Fit domain temperatures on `multilingual`) |
| **6. Batch vs. Line-by-Line Parity** | `100%` identical answers | **100.0% (`98/98`)** | **PASS** |

---

## 4. Benchmark Suite #2: Laya as an AI Orchestrator Model Router (`24 Cases`)

We also evaluated whether **Laya** can act as a local **System 1 Orchestrator Router** (`data/orchestrator_router_cases.json`) to dispatch incoming requests across 5 downstream AI model tiers while simultaneously scoring `task_complexity` (`0–3`) and `needs_rag_retrieval` (`noul`):

| Target Model Route | Downstream Pipeline | Cases | Routing Accuracy | Mean Confidence | Mean Predicted Complexity (`0–3`) |
|---|---|---:|---:|---:|---:|
| `fast_text_rewriter` | Small fast LLM (Flash-Lite / Haiku / Local 8B) | 5 | **100.0%** (`5/5`) | `0.700` | `0.20` (Expected `0`) |
| `database_sql_api` | Deterministic SQL / REST API Tool Agent | 5 | **100.0%** (`5/5`) | `0.795` | `1.37` (Expected `1`) |
| `vision_image_ocr` | Multimodal Vision / OCR Pipeline | 5 | **100.0%** (`5/5`) | `0.651` | `1.73` (Expected `2`) |
| `deep_legal_reasoning` | Frontier Reasoning LLM (Gemini Pro / Opus / o3) | 5 | **100.0%** (`5/5`) | `0.872` | `2.65` (Expected `3`) |
| `safety_human_guardrail` | Guardrail Block / Human Compliance Escalation | 4 | **100.0%** (`4/4`) | `0.415` | `1.74` (Expected `3`) |
| **Total / Overall** | **5 Model Tiers (en, zh, ms, singlish)** | **24** | **100.0% (`24/24`)** | `0.700` | **100.0% Line vs. Batch Parity** |

* **Orchestrator Timing (3 questions/case = 72 decisions):** `7.78 s` Line-by-Line (`p50 = 360.1 ms`) vs. **`6.81 s` Batched** (`283.8 ms/case`, `1.14x` speedup on CPU).

![Benchmark Results](data/benchmark_results.png)
