# Hiver SDE Intern Take-Home Assignment
## AppleSupport AI Customer-Support Agent

### 1. Problem Framing

The goal is to build a small support agent from historical customer-support conversations on Twitter. Given a new customer message, the system should (1) classify the issue into a compact intent taxonomy, (2) retrieve similar historical AppleSupport cases and resolutions, (3) draft a grounded reply, and (4) decide whether to auto-handle or escalate to a human.

I chose **AppleSupport** because it has high interaction volume and diverse technical/support issues. I intentionally kept the system **human-in-the-loop**: weak intent confidence, weak evidence, short/context-dependent messages, or unusable replies should lead to escalation rather than confident unsupported answers.

### What I Chose Not to Build

I did not build a multi-brand agent, autonomous account actions, automatic ticket closure, a production CRM, a complex multi-agent architecture, or a custom deep-learning intent model. The focus was the measurable pipeline: **classify → retrieve → draft → decide**.

---

## 2. Dataset, Sampling and Golden Set

The primary source is the Customer Support on Twitter dataset. I filtered AppleSupport tweets plus directly connected customer tweets and reconstructed conversations using the available response relationships.

- AppleSupport tweets: **106,860**
- Directly connected customer tweets: **36,658**
- Filtered rows: **143,518**
- Reconstructed conversations: **23,674**
- Customer messages used for intent work: **30,550**

I created a **200-example manually labeled Golden Set** using a fixed random seed and excluded those examples from the historical retrieval/training pool. All 12 intents are represented.

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

The Golden Set is naturally imbalanced, so I report **macro F1 as well as accuracy**. Short acknowledgements and context-only messages are intentionally labeled `other_unclear` when they cannot be reliably interpreted from the available context.

---

## 3. Intent Taxonomy

I defined 12 practical support intents: `ios_update`, `battery_power`, `app_issue`, `device_performance`, `audio_call_issue`, `connectivity`, `apple_id_account`, `app_store_purchase`, `music_media`, `feature_how_to`, `service_support`, and `other_unclear`.

Escalation is deliberately **not** an intent. It is a handling decision that can apply to any intent when confidence or evidence is insufficient.

---

## 4. System Architecture

```text
Twitter Dataset
     |
     v
AppleSupport Filter
     |
     v
Conversation Reconstruction
     |
     v
Customer Message
     |
     +-------------------+
     |                   |
     v                   v
Intent Classifier    Historical Retriever
     |                   |
     +---------+---------+
               v
        Reply Generator
               |
               v
        Decision Engine
          /         \
 AUTO_HANDLE       ESCALATE
```

The intent classifier is **TF-IDF + Logistic Regression**. The retriever uses TF-IDF similarity over historical customer→AppleSupport response pairs. The reply generator supports an external LLM with strict grounding instructions and a safe historical-response fallback. The decision engine uses confidence, message length/context, compatible historical evidence, similarity, and reply quality checks.

---

## 5. Baselines and Intent Results

I used two baselines on the untouched 200-example Golden Set.

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| Majority class | 24.00% | 3.23% |
| TF-IDF + Logistic Regression | **36.50%** | **28.42%** |

The majority baseline predicts `other_unclear`. The classifier improves both metrics, but the result is not production-ready.

Because labeling all 30,550 customer messages manually was impractical, the classifier was trained on **30,350 deterministic silver-labeled examples** after excluding the Golden Set. The silver labels are noisy; approximately **77.7%** are `other_unclear`. I therefore treat silver labels as training data, not ground truth.

The largest classification errors were:

| True → Predicted | Errors |
|---|---:|
| `ios_update` → `other_unclear` | 35 |
| `service_support` → `other_unclear` | 20 |
| `feature_how_to` → `other_unclear` | 11 |
| `app_issue` → `device_performance` | 6 |
| `connectivity` → `other_unclear` | 6 |

These errors are concentrated around short, contextual, or semantically overlapping Twitter messages rather than being uniformly distributed.

---

## 6. Retrieval Experiments

After excluding the 200 Golden conversations, the historical pool contained **74,261 conversations** and **20,492 customer→AppleSupport response pairs**.

Mean top-1 TF-IDF similarity was **0.4674**, but similarity is not retrieval accuracy. Several very short messages achieved similarity **1.0** while retrieving an unhelpful case.

Using the silver intent of retrieved cases as a proxy for retrieval correctness:

| Retrieval | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Original | 29.50% | 34.00% | 36.00% |
| Intent-aware reranking | **33.00%** | 33.50% | 34.00% |

For actionable messages only, original retrieval intent match was **7.89% / 13.16% / 15.79%** at Top-1/3/5. These figures use noisy silver labels and are therefore a diagnostic proxy, not ground truth retrieval accuracy.

I also tested adding up to two previous messages as retrieval context. Results were **26.00% / 34.50% / 36.50%** at Top-1/3/5. A hybrid strategy for weak messages produced **26.50% / 34.00% / 36.00%**. Context helped some weak examples but did not improve the overall approach enough to replace the simpler strategy.

Intent-aware reranking improved Top-1 but reduced Top-3/Top-5, and it depends on noisy silver labels, so I did not make it the main strategy.

---

## 7. Reply Generation and Safety

The intended generation flow is:

```text
Customer message → predicted intent → historical cases →
AppleSupport responses → grounded draft reply
```

The LLM prompt requires historical evidence, prohibits invented policies/actions/guarantees, avoids exposing internal intent or AI identity, and asks for a support escalation when evidence is weak.

During final evaluation, the OpenAI API returned **`credit_balance_exhausted`**. I therefore did **not** claim that the evaluated replies were LLM-generated. The system used its historical-response fallback instead. This is an important limitation of the reported agent results, and no fabricated LLM-judge scores are included.

---

## 8. Auto-Handle vs Escalate

The decision engine is deliberately conservative. It checks:

1. Intent confidence.
2. Short/context-dependent messages.
3. Presence of intent-compatible historical cases.
4. Historical similarity.
5. Presence and usability of a reply.
6. Signs that a reply is incomplete or unsafe.

On the 200-example Golden Set:

| Decision | Count | Share |
|---|---:|---:|
| `AUTO_HANDLE` | 17 | **8.5%** |
| `ESCALATE` | 183 | **91.5%** |

The high escalation rate is intentional for this prototype: it is preferable to escalate an uncertain message than confidently send unsupported guidance.

---

## 9. Top Failure Patterns

**1. iOS update → `other_unclear` (35 errors).** Version-only or “already updated” messages can be meaningful only with earlier conversation context.

**2. Service workflow → `other_unclear` (20 errors).** Messages such as “DM sent” or “waiting for response” describe support state rather than a technical issue.

**3. Feature/how-to → `other_unclear` (11 errors).** Customers often describe a desired behavior without explicitly naming it as a how-to question.

**4. App issue → device performance (6 errors).** “Crashing”, “freezing”, and “not working” overlap across app-specific and device-wide problems.

**5. Connectivity → `other_unclear` (6 errors).** Very short messages containing Wi-Fi/Bluetooth terms are difficult to interpret without context.

A particularly important failure was **“Iphone 6S, IOS 11.0.3”**. The classifier gave approximately **0.84 confidence**, retrieval similarity was **1.0**, and the system selected `AUTO_HANDLE`, while the human label was `device_performance`. This shows why confidence and lexical similarity cannot be treated as proof that the system understands the user's actual problem.

---

## 10. Human Evaluation

I manually audited **30 examples**: all 17 AUTO_HANDLE examples plus 13 randomly selected ESCALATE examples. This is a decision-stratified audit, not a representative estimate of the full Golden Set.

| Dimension | Average |
|---|---:|
| Groundedness | **4.40 / 5** |
| Relevance | **4.17 / 5** |
| Helpfulness | **3.63 / 5** |
| Safety | **4.80 / 5** |
| Overall | **3.90 / 5** |

AUTO_HANDLE examples averaged **4.41/5 overall**, while ESCALATE examples averaged **3.23/5**. Helpfulness was the weakest dimension, mainly because conservative fallbacks can be safe but generic.

The LLM-as-judge harness is implemented with a five-dimension rubric (groundedness, relevance, helpfulness, safety, overall), and a separate script computes exact agreement, within-one agreement, MAE and weighted kappa against human scores. However, because the API had no credits during evaluation, **judge-human agreement could not be measured** and is intentionally not reported as a result.

---

## 11. What Is Misleading About My Headline Number?

The headline **36.50% intent accuracy** is easy to misread.

First, the model was trained on noisy rule-based silver labels rather than fully human-labeled training data. Second, the Golden Set is imbalanced. Third, many Twitter messages are incomplete without conversation context. Finally, the full agent's reply generation used the historical fallback during final evaluation because the LLM API was unavailable.

Therefore, 36.50% is best interpreted as a transparent prototype benchmark, not as a production-quality support-agent accuracy number. The more useful conclusion is where the system fails and which safeguards prevent those failures from becoming automatic customer responses.

---

## 12. What I Would Improve Next Week

1. Replace rule-based silver labels with a smaller but higher-quality human-labeled training set and active learning around failure cases.
2. Use conversation-aware intent classification rather than classifying the target tweet in isolation.
3. Replace lexical retrieval with dense embeddings and rerank retrieved cases using intent, issue entities and resolution relevance.
4. Evaluate retrieval using human judgments of **resolution usefulness**, not only similarity or silver-intent agreement.
5. Add explicit conversation-state handling for acknowledgements, version-only messages and support-workflow replies.
6. Run the LLM generation and judge evaluation with available API credits, then report judge-human agreement rather than leaving it unmeasured.
7. Expand the safety gate so AUTO_HANDLE requires evidence that is both semantically relevant and resolution-compatible.

---

## Final Takeaway

The prototype demonstrates the complete support-agent workflow and, more importantly, exposes its weaknesses honestly. The strongest lesson from the experiments is that **retrieval similarity and classifier confidence are not sufficient evidence of understanding**. A practical support agent therefore needs conversation context, better retrieval, stronger evaluation of resolution usefulness, and conservative human escalation.