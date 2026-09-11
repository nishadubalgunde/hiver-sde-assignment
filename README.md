# Hiver SDE Intern Take-Home Assignment

## AI Support Agent for AppleSupport

An end-to-end AI customer-support agent built using historical AppleSupport Twitter conversations.

The system:

1. Classifies a customer's message into a small set of support intents.
2. Retrieves historically similar AppleSupport conversations and their responses.
3. Generates a grounded customer-facing reply.
4. Decides whether the message can be **AUTO_HANDLE** or should be **ESCALATED to a human**.
5. Evaluates intent classification, retrieval, reply quality, and decision behavior.

The project focuses on building a measurable and reproducible support-agent pipeline rather than an autonomous system that takes actions on behalf of customers.

---

## 1. Problem Statement

Customer-support conversations contain repeated patterns.

For a new customer message, a useful support agent should be able to:

* understand what the customer is asking about,
* identify the type of issue,
* find similar historical support cases,
* draft a response consistent with how the brand previously handled similar cases,
* avoid unsupported or risky responses,
* escalate cases where confidence or evidence is insufficient.

This project explores that workflow using real customer-support conversations involving **AppleSupport**.

---

## 2. Approach

The system follows this pipeline:

```text
                    Customer Message
                           |
                           v
                  +------------------+
                  | Intent Classifier|
                  +--------+---------+
                           |
                           v
                  Predicted Intent
                           |
                           v
                  +------------------+
                  | Historical       |
                  | Case Retrieval   |
                  +--------+---------+
                           |
                           v
               Similar AppleSupport Cases
                           |
                           v
                  +------------------+
                  | Reply Generator  |
                  +--------+---------+
                           |
                           v
                   Drafted Reply
                           |
                           v
                  +------------------+
                  | Decision Engine  |
                  +--------+---------+
                           |
                    +------+------+
                    |             |
                    v             v
                AUTO_HANDLE    ESCALATE
```

The system deliberately separates:

* **intent classification**
* **historical retrieval**
* **reply generation**
* **human-escalation decision**

This makes the system easier to evaluate and debug.

---

# 3. Dataset

The primary dataset is:

**Customer Support on Twitter**

Kaggle dataset:

`thoughtvector/customer-support-on-twitter`

The dataset contains customer-support conversations between customers and brands on Twitter.

The project focuses only on the **AppleSupport** brand.

### AppleSupport filtering

After filtering the dataset:

| Data                               |   Count |
| ---------------------------------- | ------: |
| AppleSupport tweets                | 106,860 |
| Directly connected customer tweets |  36,658 |
| Total filtered rows                | 143,518 |
| Unique authors                     |  23,669 |

The project does not require processing the complete ~3M-row dataset during every experiment.

Instead, the dataset is filtered to the selected brand and then reconstructed into conversations.

---

# 4. Conversation Reconstruction

The original dataset contains tweet-level relationships such as:

* `response_tweet_id`
* `in_response_to_tweet_id`

These relationships are used to reconstruct support conversations.

After reconstruction:

| Metric                      |  Value |
| --------------------------- | -----: |
| Conversation rows           | 74,879 |
| Conversations               | 23,674 |
| Mean conversation length    |   3.16 |
| Median conversation length  |      3 |
| Maximum conversation length |     25 |

The reconstruction follows available tweet-response relationships.

Because the dataset does not necessarily contain every customer-to-customer follow-up, reconstructed conversations should not be interpreted as perfect complete threads.

---

# 5. Intent Taxonomy

A compact taxonomy of 12 intents was manually defined from the AppleSupport data.

| Intent               | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `ios_update`         | iOS/software update problems and update-related issues    |
| `battery_power`      | Battery drain, charging, overheating and power problems   |
| `app_issue`          | Specific application crashing, freezing or failing        |
| `device_performance` | General device lag, freezing or system instability        |
| `audio_call_issue`   | Calls, microphone, speaker and audio problems             |
| `connectivity`       | Wi-Fi, Bluetooth and wireless connectivity                |
| `apple_id_account`   | Apple ID, account access and authentication               |
| `app_store_purchase` | App Store, purchases, payments and refunds                |
| `music_media`        | Apple Music, iTunes, iBooks and media-related problems    |
| `feature_how_to`     | How to use a feature, setting or control                  |
| `service_support`    | Repair, service, callbacks, advisors and support requests |
| `other_unclear`      | Messages that are ambiguous or lack enough context        |

### Labeling principles

Specific causes are preferred over generic symptoms.

For example:

* A battery problem after an update → `battery_power`
* A problem specifically occurring inside an app → `app_issue`
* Wi-Fi/Bluetooth problems → `connectivity`
* Apple ID access problems → `apple_id_account`
* Refund/payment/App Store problems → `app_store_purchase`
* Explicit how-to questions → `feature_how_to`
* Repair/callback/DM/service requests → `service_support`

`Escalation` is deliberately **not** an intent. It is a separate decision made after classification and retrieval.

---

# 6. Golden Evaluation Set

A manually labeled Golden Set was created to provide an independent evaluation set.

### Golden Set

* 200 examples
* Random seed: `42`
* One initial customer message per selected conversation
* Conversation context retained
* All 12 intents represented
* Labels manually reviewed

The Golden Set is kept separate from the silver-labeled training data.

Current distribution:

```text
other_unclear          48
ios_update             47
service_support        29
app_issue              14
feature_how_to         12
connectivity           10
device_performance      8
apple_id_account        8
battery_power           8
music_media              7
audio_call_issue         5
app_store_purchase       4
```

The imbalance is intentional because the sample was designed to reflect the naturally difficult support-message distribution rather than force equal class counts.

---

# 7. Silver Labels

The remaining AppleSupport customer messages were automatically labeled using deterministic keyword/rule-based logic.

After excluding the 200 Golden Set examples:

**30,350 silver-labeled examples** were available for training.

The silver-label distribution is highly imbalanced:

```text
other_unclear          23,568
battery_power           1,245
device_performance      1,108
connectivity              891
audio_call_issue          736
music_media               604
service_support           599
ios_update                528
apple_id_account          455
app_store_purchase        291
feature_how_to             248
app_issue                   77
```

Approximately 77.7% of the silver labels are `other_unclear`.

This is an important limitation of the current experiment.

The silver labels are used as a practical training source, but the manually labeled Golden Set remains the main evaluation reference.

---

# 8. Intent Classifier

The main classifier uses:

* TF-IDF features
* unigram + bigram features
* Logistic Regression
* balanced class weights

Configuration:

```text
ngram_range = (1, 2)
max_features = 50,000
min_df = 2
class_weight = balanced
random_state = 42
```

The classifier returns:

* predicted intent
* confidence
* top three predictions

Example output:

```text
intent: battery_power
confidence: 0.9945

top_predictions:
1. battery_power
2. ios_update
3. device_performance
```

---

# 9. Baseline Evaluation

Two classification baselines were evaluated on the manually labeled Golden Set.

## Majority-Class Baseline

The majority baseline always predicts:

```text
other_unclear
```

Results:

| Metric   |  Score |
| -------- | -----: |
| Accuracy | 24.00% |
| Macro F1 |  3.23% |

Accuracy is relatively high because the Golden Set is imbalanced, but macro F1 shows that the baseline performs poorly across the intent categories.

---

## TF-IDF + Logistic Regression

The classifier was trained on the silver-labeled training data and evaluated on the untouched 200-example Golden Set.

Results:

| Metric   |  Score |
| -------- | -----: |
| Accuracy | 36.50% |
| Macro F1 | 28.42% |

Therefore:

```text
Majority Accuracy       = 24.00%
TF-IDF Accuracy         = 36.50%

Majority Macro F1       =  3.23%
TF-IDF Macro F1         = 28.42%
```

The classifier improves substantially over the majority baseline, particularly when considering macro F1.

However, 36.5% accuracy is not strong enough to claim production-ready intent classification.

---

# 10. Historical Retrieval

The retrieval component searches historical AppleSupport customer messages and their corresponding AppleSupport responses.

After excluding Golden Set conversations:

| Metric                                 |  Value |
| -------------------------------------- | -----: |
| Historical conversations               | 74,261 |
| Customer → AppleSupport response pairs | 20,492 |

TF-IDF cosine similarity is used for initial retrieval.

For each retrieved case, the system retains:

* historical customer message
* AppleSupport response
* similarity score
* silver intent label

Example:

```text
Historical customer message:
My battery is draining very quickly.

Historical AppleSupport response:
Let's take a closer look at your battery issue...
```

The historical response is important because the generator should not invent a completely new support policy.

---

# 11. Retrieval Evaluation

Retrieval was evaluated against the Golden Set.

Top-1 similarity statistics:

| Metric                  | Result |
| ----------------------- | -----: |
| Mean top-1 similarity   | 0.4674 |
| Median top-1 similarity | 0.3731 |
| Similarity >= 0.20      | 97.50% |
| Similarity >= 0.30      | 66.00% |
| Similarity >= 0.40      | 46.50% |
| Similarity >= 0.50      | 32.00% |

However, similarity alone is **not retrieval accuracy**.

A high lexical similarity can still produce an irrelevant historical case.

For example, short messages containing only an iOS version number can achieve very high similarity while providing little useful evidence about the customer's actual issue.

---

# 12. Retrieval Intent Matching

As an additional diagnostic, the retrieved case's silver intent was compared with the manually labeled Golden Set intent.

Results:

| Retrieval           |     @1 |     @3 |     @5 |
| ------------------- | -----: | -----: | -----: |
| All Golden examples | 29.50% | 34.00% | 36.00% |
| Actionable examples |  7.89% | 13.16% | 15.79% |

This metric is only a proxy because the retrieved cases use noisy silver labels.

It should therefore not be interpreted as definitive retrieval correctness.

---

# 13. Context Retrieval Experiment

Short support messages are often dependent on earlier conversation context.

A context-aware retrieval experiment therefore included up to two previous messages.

Results:

| Dataset    |     @1 |     @3 |     @5 |
| ---------- | -----: | -----: | -----: |
| All        | 26.00% | 34.50% | 36.50% |
| Actionable |  5.26% | 13.82% | 16.45% |

The context-aware approach did not consistently improve retrieval.

A hybrid strategy was also tested:

```text
Weak/context-dependent message
        ↓
Use conversation context

Normal message
        ↓
Use target message
```

Hybrid results:

| Dataset    |     @1 |     @3 |     @5 |
| ---------- | -----: | -----: | -----: |
| All        | 26.50% | 34.00% | 36.00% |
| Actionable |  7.24% | 13.16% | 15.79% |

The experiment improved some weak messages but did not provide a strong overall improvement.

---

# 14. Intent-Aware Retrieval Experiment

An additional reranking experiment increased the score of historical cases whose silver intent matched the predicted intent.

Results:

| Method                 |     @1 |     @3 |     @5 |
| ---------------------- | -----: | -----: | -----: |
| Original               | 29.50% | 34.00% | 36.00% |
| Intent-aware reranking | 33.00% | 33.50% | 34.00% |

The reranker improved top-1 matching but reduced top-3 and top-5 performance.

Because the reranker depends on noisy silver labels, it was not adopted as the unquestioned final solution.

---

# 15. Reply Generation

The reply generator uses historical AppleSupport responses as grounding evidence.

The intended flow is:

```text
Customer message
       ↓
Predicted intent
       ↓
Historical AppleSupport cases
       ↓
LLM
       ↓
Grounded customer-facing reply
```

The generation prompt instructs the model to:

* use only information supported by historical AppleSupport responses,
* avoid inventing policies,
* avoid unsupported refunds or guarantees,
* avoid claiming Apple performed an action,
* avoid mentioning the predicted intent,
* avoid mentioning that the response was generated by AI,
* keep the response concise,
* ask the customer to contact Apple Support when historical evidence is weak.

If the LLM is unavailable, the system falls back to a safe historical response when sufficiently compatible evidence exists.

Otherwise:

```text
Please contact Apple Support for further assistance with this issue.
```

---

# 16. Decision Engine

The decision engine determines:

```text
AUTO_HANDLE
```

or

```text
ESCALATE
```

The current system considers:

* intent confidence,
* message length,
* context dependence,
* presence of intent-compatible historical evidence,
* historical similarity,
* reply quality,
* fallback/generic responses.

Current thresholds:

```text
Minimum intent confidence: 0.75
Minimum retrieval similarity: 0.55
```

The system intentionally favors escalation when evidence is weak.

---

# 17. Full Agent Evaluation

The complete agent was evaluated on the 200-example Golden Set.

### Intent accuracy

```text
36.50%
```

### Decision distribution

```text
AUTO_HANDLE    17 / 200  = 8.5%
ESCALATE      183 / 200  = 91.5%
```

The high escalation rate is intentional.

The system is designed to avoid confidently auto-handling messages when intent, evidence, or the generated reply is unreliable.

---

# 18. Human Evaluation

A decision-stratified subset of 30 examples was manually evaluated.

The sample contained:

```text
17 AUTO_HANDLE
13 ESCALATE
```

Each response was scored from 1–5 on:

* Groundedness
* Relevance
* Helpfulness
* Safety
* Overall quality

### Human evaluation results

| Dimension    |  Average |
| ------------ | -------: |
| Groundedness | 4.40 / 5 |
| Relevance    | 4.17 / 5 |
| Helpfulness  | 3.63 / 5 |
| Safety       | 4.80 / 5 |
| Overall      | 3.90 / 5 |

The strongest dimension was safety.

Helpfulness was the weakest dimension, suggesting that the system is often safe and reasonably grounded but does not always provide a useful next step.

---

# 19. Human Evaluation by Decision

| Decision    | Groundedness | Relevance | Helpfulness | Safety | Overall |
| ----------- | -----------: | --------: | ----------: | -----: | ------: |
| AUTO_HANDLE |         4.82 |      4.53 |        4.41 |   4.88 |    4.41 |
| ESCALATE    |         3.85 |      3.69 |        2.62 |   4.69 |    3.23 |

This supports the conservative behavior of the decision engine.

The examples that were auto-handled generally received higher human quality scores.

However, this was a small, decision-stratified audit rather than a statistically representative sample.

---

# 20. LLM-as-Judge

An automated LLM-as-judge harness has been implemented.

The judge evaluates:

1. Groundedness
2. Relevance
3. Helpfulness
4. Safety
5. Overall quality

Each dimension is scored from 1–5.

A separate comparison script also calculates:

* exact agreement
* agreement within one point
* mean absolute error
* quadratic weighted Cohen's kappa

The intended workflow is:

```text
Agent outputs
      ↓
LLM Judge
      ↓
1–5 scores
      ↓
Compare with human scores
      ↓
Agreement metrics
```

### Current limitation

The LLM judge could not be executed during the final experiment because the OpenAI API account used for testing had exhausted its available API credit balance.

Therefore:

**No fabricated LLM-judge scores are reported.**

The harness is included in the repository and can be run after a valid API account with available credits is configured.

---

# 21. Top Failure Patterns

The largest classification failure patterns were:

| True Intent       | Predicted Intent     | Count |
| ----------------- | -------------------- | ----: |
| `ios_update`      | `other_unclear`      |    35 |
| `service_support` | `other_unclear`      |    20 |
| `feature_how_to`  | `other_unclear`      |    11 |
| `app_issue`       | `device_performance` |     6 |
| `connectivity`    | `other_unclear`      |     6 |

### Failure 1 — iOS update vs unclear

Many AppleSupport messages contain only:

* iOS versions
* update confirmations
* short responses after a previous support message

Without conversation context, the classifier cannot reliably determine whether the user is reporting an update failure, a device issue after an update, or simply acknowledging a previous response.

---

### Failure 2 — Service support vs unclear

Messages involving:

* DM requests
* advisors
* callbacks
* service appointments
* waiting for support

often contain little technical vocabulary.

The classifier therefore frequently predicts `other_unclear`.

---

### Failure 3 — Feature/how-to vs unclear

Some feature questions are implicit rather than explicit.

For example, a user may describe a desired behavior without using words such as:

```text
how
setting
enable
disable
```

This makes the intent dependent on context.

---

### Failure 4 — App issue vs device performance

Words such as:

```text
crashing
freezing
restarting
not working
```

appear in both application-specific and device-level issues.

The classifier sometimes treats an application problem as general device instability.

---

### Failure 5 — Connectivity vs unclear

Short messages such as:

```text
Wi-Fi
Bluetooth
connected
not connecting
```

can be difficult to interpret without the previous support conversation.

---

# 22. Particularly Important Failure

One particularly dangerous failure involved:

```text
iPhone 6S, iOS 11.0.3
```

The model produced:

```text
confidence = 0.8367
similarity = 1.0
decision = AUTO_HANDLE
```

The manually assigned intent was:

```text
device_performance
```

The problem is not only that the classifier was wrong.

The more important issue is that:

**high model confidence + high lexical similarity did not guarantee that the system actually understood the customer's problem.**

This is an important reason for keeping human escalation in the system.

---

# 23. What Is Misleading About My Headline Number?

The headline number of:

```text
36.50% intent accuracy
```

is useful but incomplete.

It does **not** mean that the complete support agent successfully solves 36.5% of customer issues.

Several factors make this number potentially misleading:

1. The classifier is trained using noisy silver labels.
2. The Golden Set is only 200 examples.
3. The Golden Set is naturally imbalanced.
4. Accuracy hides class-level differences.
5. The complete agent also depends on retrieval and reply generation.
6. LLM generation was unavailable during the final API-constrained run.
7. The decision engine intentionally escalates many uncertain cases.
8. A high retrieval similarity score does not guarantee a useful historical case.
9. Short customer messages can require conversation context that the classifier does not currently use.

For this reason, macro F1, failure analysis, retrieval diagnostics, human evaluation and escalation behavior are reported alongside accuracy.

---

# 24. What I Chose Not to Build

The first version deliberately focuses on the core support-agent workflow rather than trying to build a complete production support platform.

I did not build:

* a multi-brand support agent
* a fully autonomous agent that takes actions on behalf of AppleSupport
* automatic ticket closure or resolution
* a production CRM/customer database
* a complex multi-agent architecture
* a custom deep-learning intent model
* an evaluation system that assumes LLM judges are always available

The goal was to build and measure a complete pipeline:

```text
Classify → Retrieve → Draft → Decide
```

This kept the system understandable and made it easier to identify weaknesses instead of hiding them behind additional architecture.

---

# 25. What I Would Improve Next Week

### 1. Improve the Golden Set

Increase the manually labeled set from 200 to approximately 500–1,000 examples and deliberately include:

* short messages
* context-dependent messages
* ambiguous messages
* difficult intent pairs
* rare intents

### 2. Improve Intent Classification

Move beyond purely lexical TF-IDF classification.

Potential approaches:

* sentence embeddings
* semantic retrieval
* lightweight transformer-based classification
* better context-aware features

### 3. Improve Retrieval

The largest retrieval problem is that lexical similarity can retrieve an apparently similar but operationally irrelevant case.

I would test:

* semantic embeddings
* hybrid lexical + semantic retrieval
* intent-aware filtering
* conversation-level retrieval
* reranking using a cross-encoder/LLM

### 4. Better Context Handling

Instead of classifying only the target message, use:

```text
previous customer message
+
previous AppleSupport response
+
current customer message
```

when the current message is clearly context-dependent.

### 5. Improve Auto-Handle Safety

The next version should require multiple independent signals:

```text
Intent confidence
+
Evidence quality
+
Historical relevance
+
Reply quality
+
No ambiguity
```

before allowing `AUTO_HANDLE`.

### 6. Complete LLM-as-Judge Agreement

Run the implemented LLM judge with valid API credits and compare:

```text
LLM judge scores
vs
Human scores
```

using:

* exact agreement
* within-one agreement
* MAE
* weighted Cohen's kappa

### 7. Better Evaluation

Evaluate the complete system separately on:

* intent classification
* retrieval relevance
* reply groundedness
* reply helpfulness
* safety
* escalation precision

rather than using one number for the entire agent.

---

# 26. Project Structure

```text
hiver-sde-assignment/
│
├── configs/
│   ├── intents.json
│   └── labeling_guide.md
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── extract_apple_threads.py
│   │   └── reconstruct_threads.py
│   │
│   ├── intent/
│   │   ├── discover.py
│   │   ├── sample_messages.py
│   │   ├── create_golden_sample.py
│   │   ├── add_context_to_golden.py
│   │   ├── label_golden_set.py
│   │   ├── create_silver_labels.py
│   │   └── classifier.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   ├── generation/
│   │   └── reply_generator.py
│   │
│   ├── decision/
│   │   └── decision_engine.py
│   │
│   ├── agent.py
│   │
│   └── evaluation/
│       ├── majority_baseline.py
│       ├── tfidf_baseline_proper.py
│       ├── evaluate_classifier.py
│       ├── inspect_failures.py
│       ├── evaluate_retrieval.py
│       ├── evaluate_retrieval_intent.py
│       ├── inspect_retrieval_failures.py
│       ├── evaluate_context_retrieval.py
│       ├── evaluate_hybrid_retrieval.py
│       ├── evaluate_intent_reranking.py
│       ├── evaluate_agent.py
│       ├── llm_judge.py
│       ├── create_human_eval.py
│       ├── fill_human_scores.py
│       ├── evaluate_human.py
│       └── compare_judge_human.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
│
├── golden_set/
├── results/
│
├── DECISIONS.md
├── REPORT.md
├── requirements.txt
├── .env.example
└── README.md
```

Large datasets, generated CSV files, virtual environments and API keys are intentionally excluded from Git.

---

# 27. Setup

## Requirements

Recommended environment:

```text
Python 3.13+
pip
Git
```

Clone the repository:

```bash
git clone https://github.com/nishadubalgunde/hiver-sde-assignment.git
cd hiver-sde-assignment
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 28. Dataset Download

The project uses the Kaggle dataset:

```text
thoughtvector/customer-support-on-twitter
```

The download script uses `kagglehub`.

Run:

```bash
python src/data/download.py
```

The dataset should be placed under the local project data/cache location expected by the preprocessing scripts.

Large raw and processed datasets are intentionally not committed to GitHub.

---

# 29. Reproduce the Data Pipeline

After downloading the dataset:

```bash
python src/data/extract_apple_threads.py
```

Then reconstruct conversations:

```bash
python src/data/reconstruct_threads.py
```

Discover customer messages:

```bash
python src/intent/discover.py
```

Create the 200-example sample:

```bash
python src/intent/create_golden_sample.py
```

Add conversation context:

```bash
python src/intent/add_context_to_golden.py
```

Generate silver labels:

```bash
python src/intent/create_silver_labels.py
```

---

# 30. Golden Set Labeling

The Golden Set is manually labeled.

Run:

```bash
python src/intent/label_golden_set.py
```

The script:

* displays one customer message at a time,
* presents the 12 available intents,
* saves progress after every example,
* can resume from the first unlabeled example.

The final Golden Set should contain:

```text
200 examples
0 missing labels
12 represented intents
```

---

# 31. Run the Baselines

Majority baseline:

```bash
python src/evaluation/majority_baseline.py
```

TF-IDF baseline:

```bash
python src/evaluation/tfidf_baseline_proper.py
```

Classifier evaluation:

```bash
python src/evaluation/evaluate_classifier.py
```

Inspect classification failures:

```bash
python src/evaluation/inspect_failures.py
```

---

# 32. Run Retrieval Evaluation

Basic retrieval:

```bash
python src/evaluation/evaluate_retrieval.py
```

Intent-based retrieval evaluation:

```bash
python src/evaluation/evaluate_retrieval_intent.py
```

Inspect retrieval failures:

```bash
python src/evaluation/inspect_retrieval_failures.py
```

Context retrieval:

```bash
python src/evaluation/evaluate_context_retrieval.py
```

Hybrid retrieval:

```bash
python src/evaluation/evaluate_hybrid_retrieval.py
```

Intent-aware reranking:

```bash
python src/evaluation/evaluate_intent_reranking.py
```

---

# 33. Run the Full Agent

The complete agent can be evaluated with:

```bash
python src/evaluation/evaluate_agent.py
```

The agent performs:

```text
Customer message
       ↓
Intent classification
       ↓
Historical retrieval
       ↓
Reply generation
       ↓
Safety / decision engine
       ↓
AUTO_HANDLE or ESCALATE
```

---

# 34. Optional OpenAI Configuration

Copy:

```text
.env.example
```

to:

```text
.env
```

Configure:

```text
OPENAI_API_KEY=your_api_key
OPENAI_GENERATION_MODEL=your_available_model
OPENAI_JUDGE_MODEL=your_available_model
```

Never commit `.env` or an API key to GitHub.

The API key is only read from the environment.

If the API is unavailable, the system uses the implemented safe fallback behavior.

---

# 35. LLM Judge Evaluation

After configuring an available LLM API:

```bash
python src/evaluation/llm_judge.py
```

Then compare the LLM judge against human scores:

```bash
python src/evaluation/compare_judge_human.py
```

The comparison script calculates agreement metrics when both human and LLM-judge results are available.

---

# 36. Evaluation Philosophy

The project intentionally avoids treating a single headline metric as proof that the support agent works.

Evaluation is separated into:

```text
                    System Evaluation
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
 Intent              Retrieval           Reply
Classification       Quality             Quality
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                    Decision Quality
```

This makes it possible to identify where errors originate.

For example:

```text
Wrong intent
    ↓
Wrong historical cases
    ↓
Weak reply
    ↓
Incorrect auto-handle decision
```

or:

```text
Correct intent
    ↓
Poor retrieval
    ↓
Weak historical evidence
    ↓
Escalation
```

---

# 37. Key Results

### Intent classification

```text
Majority Accuracy:       24.00%
Majority Macro F1:        3.23%

TF-IDF Accuracy:         36.50%
TF-IDF Macro F1:         28.42%
```

### Full agent

```text
Intent Accuracy:         36.50%

AUTO_HANDLE:              8.50%
ESCALATE:                91.50%
```

### Human evaluation

```text
Groundedness:             4.40 / 5
Relevance:                4.17 / 5
Helpfulness:              3.63 / 5
Safety:                   4.80 / 5
Overall:                  3.90 / 5
```

These numbers should be interpreted together rather than as a single overall success score.

---

# 38. Limitations

The current system has several important limitations:

* The Golden Set contains only 200 examples.
* The Golden Set is naturally imbalanced.
* Silver labels are noisy and heavily dominated by `other_unclear`.
* TF-IDF is mainly lexical and does not fully capture semantic similarity.
* Short messages often require conversation context.
* Historical similarity can be high even when the retrieved case is operationally irrelevant.
* The current intent model is not production-grade.
* LLM generation was unavailable during the final API-constrained evaluation.
* LLM-as-judge results are therefore not reported.
* Human evaluation was performed on only 30 decision-stratified examples.
* The dataset reconstruction cannot guarantee complete multi-turn conversations.

---

# 39. Safety Philosophy

The system is designed as a **human-in-the-loop support assistant**, not a fully autonomous customer-support agent.

The AI should:

```text
Recommend
Draft
Retrieve
Explain
Escalate
```

It should not:

```text
Automatically issue refunds
Automatically modify customer accounts
Automatically close tickets
Make unsupported policy claims
Claim that Apple performed an action when it did not
```

When evidence is weak, the safer behavior is escalation.

---

# 40. Reproducibility

The repository contains the code required to reproduce the experiments.

Large generated files and datasets are excluded from Git to keep the repository lightweight.

The main reproducibility flow is:

```bash
# Install
pip install -r requirements.txt

# Download dataset
python src/data/download.py

# Prepare data
python src/data/extract_apple_threads.py
python src/data/reconstruct_threads.py
python src/intent/discover.py

# Prepare labels
python src/intent/create_silver_labels.py

# Evaluate
python src/evaluation/majority_baseline.py
python src/evaluation/tfidf_baseline_proper.py
python src/evaluation/evaluate_classifier.py

# Retrieval
python src/evaluation/evaluate_retrieval.py

# Full agent
python src/evaluation/evaluate_agent.py
```

---

# 41. Documentation

Additional project documentation:

* `REPORT.md` — detailed assignment report and experimental findings
* `DECISIONS.md` — non-obvious engineering and modeling decisions
* `configs/intents.json` — intent taxonomy
* `configs/labeling_guide.md` — Golden Set labeling rules

---

# 42. Final Takeaway

The main finding from this project is that building a support agent is not just an intent-classification problem.

A useful system needs:

```text
Intent understanding
        +
Historical evidence
        +
Grounded response generation
        +
Safety-aware escalation
```

The current system demonstrates the complete workflow and establishes measurable baselines.

The strongest current result is the improvement from the majority baseline to the TF-IDF classifier:

```text
24.00% → 36.50% accuracy
3.23%  → 28.42% macro F1
```

At the same time, the failure analysis shows that short, context-dependent customer messages remain difficult.

The most important next step is therefore not simply increasing model complexity, but improving:

**context understanding + semantic retrieval + evaluation quality + safe decision-making.**
