![ProMoAI logo](https://i.imgur.com/cu5OtpK.png)

# ProMoAI & PMAx

**ProMoAI** is an AI-powered suite for Process Mining that leverages Large Language Models (LLMs) to bridge the gap between natural language and technical process analysis. The framework now includes **PMAx**, an autonomous agentic system for data-driven insights.

The suite consists of two primary modules:

1. **ProMoAI (Model Generation):** Transforms text or event logs into formal process models (BPMN/Petri nets).
2. **PMAx (Agentic Analytics):** An autonomous multi-agent framework that functions as a virtual process analyst to query event logs and generate data-grounded reports.

---

## Features

### 1. ProMoAI: Model Generation & Refinement

- **Text-to-Model:** Generate BPMN or PNML models from natural language descriptions.
- **Model Refinement:** Upload an existing BPMN or Petri net and use AI to modify or extend it via chat.
- **Discovery Baseline:** Start with an XES event log to discover an initial model, then refine it using the LLM.
- **Rule-Based POWL Discovery:** Extract DECLARE constraints from a process description or enter them manually, inspect their empirical support and confidence, select rules, and discover a rule-constrained POWL model using event-, trace-, or edit-distance repair.

### 2. PMAx: Agentic Process Mining (New!)

- **Autonomous Reasoning:** Uses a "divide-and-conquer" architecture with specialized **Engineer** and **Analyst** agents.
- **Privacy-Preserving:** Only lightweight metadata (column names/types) is sent to the LLM. Raw event data never leaves your local environment.
- **Deterministic Accuracy:** The system generates and executes local Python code (using whitelisted data preprocessing libraries) to compute exact metrics, avoiding LLM hallucinations.
- **Comprehensive Reporting:** Automatically generates tables, statistical charts, and narrative insights from high-level business questions.

---

## Launching the App

### On the Cloud

Access the unified suite directly at: [https://promoai.streamlit.app/](https://promoai.streamlit.app/)

### Locally

1. Clone this repository.
2. Install the required environment and packages (see [Requirements](#requirements)).
3. Run the application:

- **Unified Suite (ProMoAI + PMAx):**
  ```bash
  streamlit run app.py
  ```
- **Standalone ProMoAI (Legacy Interface):**
  ```bash
  streamlit run promoai_standalone.py
  ```

### Rule-based discovery

Open **ProMoAI → Rules** in the unified app (or select **Rules** in the standalone app). Upload an XES, compressed XES, or CSV event log and check the case ID, activity, and optional timestamp columns. Describe the constraints in natural language and click **Extract rules from description**, or paste rule assignments directly into the editor. Rule extraction requires an LLM connection configured on the Setup page; manually supplied rules do not.

After applying rules, inspect per-rule support, confidence, activation counts, and violations, select the rules to enforce, and choose a repair strategy and noise threshold. **Discover rule-constrained model** calls the rule-aware POWL miner with final rule verification enabled. The resulting model can be viewed and exported through the existing BPMN, Petri net, and POWL tabs. LLM edits to rule-constrained models are checked against the selected rules before replacing the existing model.

The supported templates are `AtMostOnce`, `Init`, `End`, `Existence`, `CoExistence`, `NotCoExistence`, `NotSuccession`, `Precedence`, `ChainPrecedence`, `RespondedExistence`, `Response`, and `ChainResponse`.

```python
import promoai
from powl.discovery.rule_based import InitializationRule, ResponseRule

event_log = [["Register", "Close"], ["Register", "Close"]]
rules = [InitializationRule("Register"), ResponseRule("Register", "Close")]
metrics = promoai.validate_event_log_rules(event_log, rules)
model = promoai.generate_model_from_event_log_with_rules(
    event_log,
    rules,
    repair_mode="event_level",
    noise_threshold=0.0,
)
powl_model = model.get_powl()
```

To extract rules from text in Python, use `promoai.generate_rules_from_text(description, activities, api_key, ai_model, ai_provider)`. You can also parse supplied rule assignments without an LLM using `promoai.prompting.declarative_rules.parse_declare_rules(code, activities)`.

Rule-based discovery requires `powl>=2.3.8` and Python 3.11 or newer.

### Installation as Python Library:

You can install the core ProMoAI components via pip:

```bash
pip install promoai
```

## Requirements

- _Environment:_ Python 3.11 or newer is required by POWL.
- _Dependencies:_ all required dependencies are listed in the file 'requirements.txt'.
- _Packages:_ all required packages are listed in the file 'packages.txt'.

## Citation

If you use this suite in your research, please cite the relevant papers:

### ProMoAI (Process Modeling)

```bibtex
@inproceedings{DBLP:conf/ijcai/KouraniB0A24,
  author       = {Humam Kourani and
                  Alessandro Berti and
                  Daniel Schuster and
                  Wil M. P. van der Aalst},
  title        = {ProMoAI: Process Modeling with Generative {AI}},
  booktitle    = {Proceedings of the Thirty-Third International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2024},
  pages        = {8708--8712},
  publisher    = {ijcai.org},
  year         = {2024},
  url          = {https://www.ijcai.org/proceedings/2024/1014}
}
```
