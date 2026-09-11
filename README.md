# Hiver SDE Intern Take-Home Assignment

## AppleSupport AI Customer-Support Agent

A small, reproducible support-agent prototype built from the **Customer Support on Twitter** dataset. The system classifies a customer message, retrieves similar historical AppleSupport resolutions, drafts a grounded reply, and decides whether to `AUTO_HANDLE` or `ESCALATE` to a human.

> **Important:** the current evaluation used the historical-response fallback because the OpenAI API account had no available credits. No LLM-generated results or LLM-judge scores are claimed where the API could not run.

## 1. Architecture

```text
Customer-Support on Twitter dataset
              |
              v
       AppleSupport filter
              |
              v
   Conversation reconstruction
              |
              v
       Customer messages
              |
       +------+------+
       |             |
       v             v
 Intent classifier  Historical retriever
       |             |
       |             v
       |       Similar solved cases
       |             |
       +------+------+
              |
              v
        Reply generator
              |
              v
       Safety / decision engine
          /           \
         v             v
 AUTO_HANDLE       ESCALATE
```

## 2. Dataset and Sampling

Primary dataset: `thoughtvector/customer-support-on-twitter`.

For this project I selected **AppleSupport** because it has enough volume and varied support issues for classification and retrieval experiments.

Key preprocessing results:

- 106,860 AppleSupport tweets
- 36,658 directly connected customer tweets
- 143,518 filtered rows
- 74,879 reconstructed conversation rows
- 23,674 reconstructed conversations
- 30,550 customer messages used for intent work

The reconstruction follows the available `response_tweet_id` / `in_response_to_tweet_id` relationships. Some real conversations may still be incomplete because the source dataset does not contain every possible follow-up.

## 3. Intent Taxonomy

The manually labeled Golden Set uses 12 practical intents:

- `ios_update`
- `battery_power`
- `app_issue`
- `device_performance`
- `audio_call_issue`
- `connectivity`
- `apple_id_account`
- `app_store_purchase`
- `music_media`
- `feature_how_to`
- `service_support`
- `other_unclear`

Escalation is deliberately a **decision**, not an intent.

## 4. Golden Evaluation Set

A **200-example manually labeled Golden Set** was created with a fixed random seed. All 12 intents are represented.

Distribution:

| Intent | Count |
|---|---:|
| `other_unclear` | 48 |
| `ios_update` | 47 |
| `service_support` | 29 |
| `app_issue` | 14 |
| `feature_how_to` | 12 |
| `connectivity` | 10 |
| `device_performance` | 8 |
| `apple_id_account` | 8 |
| `battery_power` | 8 |
| `music_media` | 7 |
| `audio_call_issue` | 5 |
| `app_store_purchase` | 4 |

The Golden Set is intentionally untouched during classifier training and is the main evaluation set.

## 5. Classifier and Baselines

The initial classifier is **TF-IDF + Logistic Regression**. Because the large training set uses deterministic silver labels, those labels are treated as noisy training data rather than ground truth.

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| Majority class | 24.00% | 3.23% |
| TF-IDF + Logistic Regression | **36.50%** | **28.42%** |

The improvement over majority is useful, but the classifier is not production-ready. Macro F1 is reported because the Golden Set is imbalanced.

## 6. Silver Labels

30,350 non-Golden customer messages were assigned deterministic rule-based silver labels for training.

The main limitation is severe imbalance: approximately **77.7%** are labeled `other_unclear`. This is explicitly treated as a weakness of the current training setup.

## 7. Historical Retrieval

The retriever searches historical customer → AppleSupport response pairs while excluding the 200 Golden conversations.

- 74,261 historical conversations available after Golden exclusion
- 20,492 historical customer → AppleSupport response pairs
- Mean top-1 TF-IDF similarity: **0.4674**

Similarity is not treated as retrieval accuracy. Failure analysis found cases with similarity `1.0` that were still irrelevant, especially for short/version-only messages.

### Retrieval intent-match experiment

| Metric | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Intent match | 29.50% | 34.00% | 36.00% |

For actionable messages only: 7.89% / 13.16% / 15.79%.

Two additional experiments were run:

- Context-aware retrieval: 26.0% / 34.5% / 36.5% for Top-1/3/5.
- Intent-aware reranking: improved Top-1 from 29.5% to 33.0%, but reduced Top-3/Top-5, so it was not adopted as the main strategy.

## 8. Reply Generation

The generator is designed to ground replies in historical AppleSupport responses. When an LLM is available, the prompt explicitly prohibits unsupported policies, actions, guarantees, and invented troubleshooting steps.

When the LLM is unavailable, the system uses the strongest intent-compatible historical response; if no sufficiently strong evidence exists, it returns a safe escalation message.

During final evaluation the OpenAI API returned `credit_balance_exhausted`, so the evaluated replies came from the fallback path. The repository includes the LLM judge harness, but no fabricated judge scores are reported.

## 9. Decision Engine

`AUTO_HANDLE` requires all of the following:

- sufficiently high intent confidence
- message is not obviously short/context-dependent
- compatible historical evidence exists
- compatible evidence has sufficient similarity
- a usable customer-facing reply exists

Otherwise the system escalates.

On the 200-example Golden Set:

- `AUTO_HANDLE`: 17 (8.5%)
- `ESCALATE`: 183 (91.5%)

The design intentionally favors safe escalation over confident but unsupported replies.

## 10. Agent Evaluation

Headline intent result:

**36.50% intent accuracy on the 200-example Golden Set.**

This is an intent-classification metric, not an overall measure of support-agent quality.

Largest classifier failure groups:

1. `ios_update` → `other_unclear`: 35
2. `service_support` → `other_unclear`: 20
3. `feature_how_to` → `other_unclear`: 11
4. `app_issue` → `device_performance`: 6
5. `connectivity` → `other_unclear`: 6

A particularly important failure was a version-only message that received high classifier confidence and similarity yet was still context-dependent. This demonstrated that numerical confidence alone is not sufficient for safe automation.

## 11. Human Evaluation

A 30-example decision-stratified human audit was completed: all 17 auto-handled cases plus 13 randomly selected escalations.

| Dimension | Mean |
|---|---:|
| Groundedness | **4.40 / 5** |
| Relevance | **4.17 / 5** |
| Helpfulness | **3.63 / 5** |
| Safety | **4.80 / 5** |
| Overall | **3.90 / 5** |

This is a small audit, not a representative estimate of all 200 examples. Auto-handled examples had higher mean overall quality (4.41/5) than the sampled escalations (3.23/5).

## 12. LLM-as-Judge

The repository contains:

- `src/evaluation/llm_judge.py`
- `src/evaluation/compare_judge_human.py`

The intended judge rubric scores groundedness, relevance, helpfulness, safety, and overall quality from 1–5 and compares those scores with human ratings.

The judge could not be executed during this run because the OpenAI API account had no credits. Therefore, **LLM-vs-human agreement is not claimed**.

## 13. What Is Misleading About the Headline Number?

The 36.50% number is only intent accuracy on a small, manually labeled Golden Set. It does not mean that 36.5% of complete support interactions are successfully solved.

It is affected by:

- a small 200-example evaluation set
- class imbalance
- noisy silver-label training data
- context-dependent Twitter messages
- an incomplete conversation reconstruction
- LLM generation being unavailable during the final run

The result is therefore best interpreted as a baseline measurement that exposes where the system still needs work.

## 14. What I Chose Not to Build

To keep the first version focused, I did not build:

- a multi-brand support agent
- autonomous actions on behalf of AppleSupport
- automatic ticket closure/resolution
- a production CRM/customer database
- a complex multi-agent architecture
- a custom deep-learning classifier
- an evaluation pipeline that assumes LLM APIs are always available

The goal was to first build and measure a complete **classify → retrieve → draft → decide** pipeline.

## 15. Next Week

The highest-value improvements would be:

1. Replace noisy silver labels with a larger stratified labeled training set.
2. Add conversation-aware intent classification rather than target-message-only classification.
3. Use semantic embeddings for retrieval instead of only TF-IDF.
4. Add reranking that considers intent, conversation state, and response quality.
5. Tune escalation thresholds against a larger human-labeled safety set.
6. Run the LLM generator and LLM-as-judge with available API credits and measure judge-human agreement.

## 16. Reproducibility

### Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### Download dataset

```bash
python src/data/download.py
```

The download script uses `kagglehub` to fetch the dataset.

### Build the processed data

```bash
python src/data/extract_apple_threads.py
python src/data/reconstruct_threads.py
python src/intent/discover.py
python src/intent/sample_messages.py
python src/intent/create_golden_sample.py
python src/intent/add_context_to_golden.py
```

### Label the Golden Set

```bash
python src/intent/label_golden_set.py
```

This is the only manual labeling step. The script saves progress after each example.

### Create silver labels and run evaluation

```bash
python src/intent/create_silver_labels.py
python src/evaluation/majority_baseline.py
python src/evaluation/tfidf_baseline_proper.py
python src/evaluation/evaluate_classifier.py
python src/evaluation/inspect_failures.py
python src/evaluation/evaluate_retrieval.py
python src/evaluation/evaluate_retrieval_intent.py
python src/evaluation/evaluate_context_retrieval.py
python src/evaluation/evaluate_hybrid_retrieval.py
python src/evaluation/evaluate_intent_reranking.py
python src/evaluation/evaluate_agent.py
```

### Human evaluation

```bash
python src/evaluation/create_human_eval.py
python src/evaluation/fill_human_scores.py
python src/evaluation/evaluate_human.py
```

### Optional LLM evaluation

Create `.env` locally from `.env.example` and add an API key. Never commit `.env`.

```bash
python src/evaluation/llm_judge.py
python src/evaluation/compare_judge_human.py
```

If API credits are unavailable, the judge will not produce scores; do not fabricate them.

## 17. Repository Structure

```text
configs/
  intents.json
  labeling_guide.md

src/
  agent.py
  data/
  intent/
  retrieval/
  generation/
  decision/
  evaluation/

DECISIONS.md
REPORT.md
README.md
requirements.txt
.env.example
```

Generated datasets and evaluation CSVs are intentionally excluded from normal Git tracking because the source dataset is large. For submission, the **200-example Golden Set must be included or otherwise delivered with the repository**, as required by the assignment.

## 18. Key Takeaway

The prototype demonstrates the full support-agent workflow and, more importantly, measures its weaknesses honestly. The current strongest evidence is that simple TF-IDF classification beats the majority baseline, while retrieval and safe decision-making remain limited by short, context-dependent support messages and noisy training labels.