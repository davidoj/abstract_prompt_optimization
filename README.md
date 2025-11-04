# Abstract Prompt Optimization for OTT-QA

This repository sets up an experiment that compares **direct prompt optimization** with
**abstract prompt optimization** (via a prompt interpreter) on the
[OTT-QA](https://ott-qa.github.io/) benchmark. The pipeline is built on top of
[DSPy](https://dspy.ai) and GEPA.

The code prepares the following pieces so you can plug in your OpenRouter-backed models
and run the optimization loops locally:

* dataset loaders for OTT-QA style JSONL splits;
* metrics for exact match, evidence coverage/precision, JSON validity, and latency;
* two optimization strategies (direct prompt vs. abstract prompt + interpreter);
* a CLI wrapper that invokes DSPy, GEPA, and writes experiment reports.

## Prerequisites

1. **Python 3.10+**
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or export OTT-QA splits (`train.jsonl`, `val.jsonl`, `test.jsonl`).
4. Export your OpenRouter API key (and optional headers required by OpenRouter):
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   export OPENROUTER_HTTP_REFERER="https://yourdomain.example"
   export OPENROUTER_APP_NAME="ott-qa-dspy"
   ```

The OpenRouter headers are optional but recommended for higher rate limits.

## Configuration

The CLI expects a YAML configuration file (see [`configs/ottqa_example.yaml`](configs/ottqa_example.yaml))
that describes:

* dataset paths for `train`, `val`, `test` splits;
* model configuration (OpenRouter model name, temperature, etc.);
* the list of strategies to evaluate, including prompts, structured strategy templates, and GEPA hyperparameters.

### Strategy Types

* `direct`: GEPA mutates the prompt that is fed straight into the QA module.
* `abstract`: GEPA jointly optimizes the prompt interpreter **and** the downstream QA prompt.
  The interpreter receives a rich strategy template (steps, prompt construction tactics such as
  few-shot placement, repetition cues, execution guidance, failure modes) which can be edited or
  swapped per run.

Both strategies ask the QA model to return JSON with fields `answer`, `evidence_cells`, and
`evidence_sentences`. Metrics validate this JSON and compute EM and evidence coverage.

The abstract strategy now passes the full template into the interpreter so it can preserve
critical tactics while translating the high-level plan into executable instructions.

### GEPA Judge

The GEPA teleprompter is configured with an OTT-QA specific judge that

* parses model JSON,
* scores with the selected primary metric, and
* returns targeted feedback (e.g., missing evidence, malformed JSON, wrong answer).

This feedback is surfaced to GEPA during search so that candidate prompts receive actionable
critiques instead of bare scalar scores.

## Running the Experiment

```bash
python -m abstract_prompt_optimization.cli run configs/ottqa_example.yaml
```

After completion, the runner writes a timestamped JSON report under `results/` containing
aggregate metrics and per-example predictions.

## Reports

Each report captures:

* per-strategy aggregate metrics (exact match, evidence precision/recall, JSON validity rate,
  average latency);
* raw model outputs and parsed predictions for each evaluated example.

You can further analyze the JSON or integrate it into your own dashboards.

## Extending

* Modify the YAML config to change GEPA hyperparameters, prompts, or add new strategies.
* Use DSPy's trace tooling to inspect model calls during optimization.
* Add additional metrics by extending `abstract_prompt_optimization/evaluation.py`.
* Reference additional GEPA examples in the DSPy repository for inspiration:
  * https://github.com/stanfordnlp/dspy/blob/main/examples/teleprompters/gepa.ipynb
  * https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa.py
