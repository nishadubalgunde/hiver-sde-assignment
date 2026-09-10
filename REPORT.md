# Hiver SDE Intern Take-Home Assignment

## AppleSupport AI Customer-Support Agent

### 1. Problem Framing

The goal of this project is to build a small AI support agent using historical customer-support conversations from Twitter.

The agent should be able to:

1. Understand what type of issue the customer is facing.
2. Find similar cases from the brand's previous support conversations.
3. Draft a reply based on how the brand handled similar cases in the past.
4. Decide whether the reply is safe enough to handle automatically or whether it should be escalated to a human.

I chose **AppleSupport** as the brand because the dataset contains a large number of AppleSupport interactions and a wide variety of technical and support-related problems.

I intentionally designed the system as a **human-in-the-loop support agent**. The system should not blindly answer every customer. When confidence or historical evidence is weak, it should prefer escalation.

---

## What I Chose Not to Build

I deliberately kept the first version focused on the core support-agent workflow rather than trying to build a complete production support platform.

I did not build:

* a multi-brand support agent
* a fully autonomous agent that can take actions on behalf of AppleSupport
* automatic ticket closure or resolution
* a production-grade customer database or CRM
* a complex multi-agent architecture
* a custom deep-learning intent model
* a fully automated evaluation system that assumes LLM scores are always available

The goal was to first build and measure a complete, understandable pipeline: **classify → retrieve → draft → decide whether to handle or escalate**.

This also made it easier to identify where the current approach fails instead of hiding the weaknesses behind a more complicated architecture.


# 2. Dataset and Sampling

The primary dataset is the Customer Support on Twitter dataset.

The dataset contains:

* `tweet_id`
* `author_id`
* `inbound`
* `created_at`
* `text`
* `response_tweet_id`
* `in_response_to_tweet_id`

I first filtered the dataset to AppleSupport conversations.

The AppleSupport subset contained:

* **106,860 AppleSupport tweets**
* **36,658 directly connected customer tweets**
* **143,518 total filtered rows**

I then reconstructed conversations using the response relationships available in the dataset.

This resulted in approximately:

* **74,879 conversation rows**
* **23,674 conversations**

From these conversations, I extracted customer messages for intent classification.

---

# 3. Intent Taxonomy

Instead of trying to create a very large number of categories, I defined 12 practical support intents.

| Intent               | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `ios_update`         | iOS/software update problems                         |
| `battery_power`      | Battery drain, charging and power issues             |
| `app_issue`          | Problems with a specific application                 |
| `device_performance` | General lag, freezing or device instability          |
| `audio_call_issue`   | Calls, microphone, speaker and audio problems        |
| `connectivity`       | Wi-Fi, Bluetooth and wireless connectivity           |
| `apple_id_account`   | Apple ID and account access problems                 |
| `app_store_purchase` | App Store, payments, purchases and refunds           |
| `music_media`        | Apple Music, iTunes and other media issues           |
| `feature_how_to`     | Questions about using Apple features/settings        |
| `service_support`    | Repair, appointments, callbacks and support requests |
| `other_unclear`      | Messages that cannot be reliably classified          |

I kept escalation separate from these intents because escalation describes **how the system should handle the message**, rather than what the message is about.

---

# 4. Golden Evaluation Set

I created a manually labeled Golden Set containing **200 examples**.

The examples were sampled using a fixed random seed so that the evaluation can be reproduced.

All 200 examples were manually labeled and all 12 intents are represented.

The distribution was:

| Intent               | Examples |
| -------------------- | -------: |
| `other_unclear`      |       48 |
| `ios_update`         |       47 |
| `service_support`    |       29 |
| `app_issue`          |       14 |
| `feature_how_to`     |       12 |
| `connectivity`       |       10 |
| `device_performance` |        8 |
| `apple_id_account`   |        8 |
| `battery_power`      |        8 |
| `music_media`        |        7 |
| `audio_call_issue`   |        5 |
| `app_store_purchase` |        4 |

One important observation is that the Golden Set is not balanced. In particular, `other_unclear` and `ios_update` occur much more often than some of the smaller categories.

Because of this, I report **macro F1 alongside accuracy** instead of relying on accuracy alone.

---

# 5. System Architecture

The overall pipeline is:

```text
Twitter Customer-Support Dataset
              |
              v
        AppleSupport Filter
              |
              v
     Conversation Reconstruction
              |
              v
        Customer Messages
              |
       +------+------+
       |             |
       v             v
 Intent Classifier   Historical Retriever
       |             |
       |             v
       |       Similar AppleSupport Cases
       |             |
       +------+------+
              |
              v
        Reply Generator
              |
              v
       Safety / Decision Engine
              |
       +------+------+
       |             |
       v             v
 AUTO_HANDLE      ESCALATE
```

The intent classifier currently uses **TF-IDF + Logistic Regression**.

The historical retriever uses TF-IDF similarity to find previous customer-support cases and their corresponding AppleSupport responses.

The reply generation layer supports an external LLM, but also has a safe historical-response fallback when the LLM is unavailable.

---

# 6. Baselines

I used two baselines for intent classification.

## 6.1 Majority-Class Baseline

The majority baseline always predicts:

`other_unclear`

Results on the 200-example Golden Set:

| Metric   | Majority Baseline |
| -------- | ----------------: |
| Accuracy |        **24.00%** |
| Macro F1 |         **3.23%** |

The accuracy is relatively high compared with the macro F1 because the Golden Set is imbalanced.

---

## 6.2 TF-IDF + Logistic Regression

The second baseline uses TF-IDF features with a Logistic Regression classifier.

The model was trained on the larger silver-labeled dataset and evaluated only on the manually labeled Golden Set.

Results:

| Metric   | TF-IDF + Logistic Regression |
| -------- | ---------------------------: |
| Accuracy |                   **36.50%** |
| Macro F1 |                   **28.42%** |

This is an improvement over the majority baseline:

* Accuracy: **24.00% → 36.50%**
* Macro F1: **3.23% → 28.42%**

However, the result is still far from what I would consider production-ready.

---

# 7. Silver Labels

Since manually labeling the full customer-message dataset was not practical, I created deterministic rule-based silver labels for the remaining messages.

The silver dataset contained approximately **30,350 training examples** after excluding the Golden Set.

The biggest limitation is the class imbalance:

* `other_unclear`: approximately **77.7%**

This means the silver labels are noisy and biased.

Therefore, I treat the silver labels as **training data only**, not as ground truth.

This is also one reason why the final classifier results should be interpreted carefully.

---

# 8. Retrieval Experiments

The agent retrieves historical AppleSupport customer-support cases before generating a reply.

For the historical retrieval dataset:

* **74,261 historical conversations** were available after excluding Golden Set conversations.
* **20,492 historical customer → AppleSupport response pairs** were available.

The average top-1 similarity on the Golden Set was:

**0.4674**

However, similarity alone is not retrieval accuracy.

For example, some very short messages received a similarity score of **1.0** even though the retrieved case was not actually useful.

This was an important finding during failure analysis.

## Retrieval intent matching

I also checked whether the retrieved case's silver intent matched the human-labeled intent.

Results:

| Metric                 |  Top-1 |  Top-3 |  Top-5 |
| ---------------------- | -----: | -----: | -----: |
| Retrieval intent match | 29.50% | 34.00% | 36.00% |

For actionable messages only:

| Metric                 | Top-1 |  Top-3 |  Top-5 |
| ---------------------- | ----: | -----: | -----: |
| Retrieval intent match | 7.89% | 13.16% | 15.79% |

These numbers show that lexical similarity is not enough to reliably retrieve the correct historical resolution.

---

# 9. Retrieval Experiments with Context

I also tested whether adding previous conversation messages would improve retrieval.

The idea was that a message such as:

> "Done"

or

> "11.0.2"

contains very little information by itself.

Using previous conversation context produced:

* Top-1: **26.00%**
* Top-3: **34.50%**
* Top-5: **36.50%**

A hybrid approach was also tested, where context was used only for weak/context-dependent messages.

The hybrid approach produced:

* Top-1: **26.50%**
* Top-3: **34.00%**
* Top-5: **36.00%**

It helped some weak examples but did not provide enough overall improvement to replace the simpler retrieval approach.

---

# 10. Intent-Aware Retrieval Experiment

I tested a second retrieval idea where a historical case received an additional score when its silver intent matched the predicted intent.

Results:

| Method                 |      Top-1 |  Top-3 |  Top-5 |
| ---------------------- | ---------: | -----: | -----: |
| Original retrieval     |     29.50% | 34.00% | 36.00% |
| Intent-aware reranking | **33.00%** | 33.50% | 34.00% |

The reranking improved Top-1 retrieval, but Top-3 and Top-5 became worse.

Since the additional intent signal is itself based on noisy silver labels, I decided not to make this the main retrieval strategy.

---

# 11. Reply Generation

The reply generation layer is designed around historical AppleSupport responses.

The intended flow is:

```text
Customer message
       |
       v
Predicted intent
       |
       v
Similar historical cases
       |
       v
Historical AppleSupport responses
       |
       v
LLM
       |
       v
Customer-facing draft reply
```

The LLM prompt instructs the generator to:

* use historical responses as evidence
* avoid inventing policies
* avoid unsupported actions or guarantees
* avoid mentioning the internal intent
* avoid mentioning that it is an AI
* keep the reply concise
* ask the customer to contact Apple Support when the evidence is weak

---

# 12. Important LLM Limitation

The OpenAI API could not be used during the final evaluation because the API account had no available credits.

The API returned:

`credit_balance_exhausted`

Because of this, I did **not** pretend that the 200 evaluated replies were generated by the LLM.

Instead, the system used the historical-response fallback.

The fallback was tested and worked successfully.

This means the current agent evaluation measures the full pipeline and safety/decision logic, but its replies should not be described as LLM-generated replies.

---

# 13. Decision Engine

The decision engine determines whether the system should automatically handle the message or escalate it.

The current system checks:

1. Intent confidence.
2. Whether the message is very short or context-dependent.
3. Whether compatible historical cases exist.
4. Historical similarity.
5. Whether a usable reply exists.
6. Whether the reply appears incomplete or unsafe.

The current thresholds are deliberately conservative.

The final 200-example evaluation produced:

| Decision      | Examples | Percentage |
| ------------- | -------: | ---------: |
| `AUTO_HANDLE` |       17 |   **8.5%** |
| `ESCALATE`    |      183 |  **91.5%** |

I consider the high escalation rate acceptable for this prototype because the goal is to avoid confidently sending unsupported replies.

---

# 14. Intent Evaluation

The final agent evaluation on the 200-example Golden Set produced:

**Intent Accuracy: 36.50%**

The largest confusion patterns were:

| True Intent       | Predicted Intent     | Errors |
| ----------------- | -------------------- | -----: |
| `ios_update`      | `other_unclear`      |     35 |
| `service_support` | `other_unclear`      |     20 |
| `feature_how_to`  | `other_unclear`      |     11 |
| `app_issue`       | `device_performance` |      6 |
| `connectivity`    | `other_unclear`      |      6 |

The most common problem is therefore not random classification. It is the difficulty of understanding short, contextual Twitter messages.

---

# 15. Top 5 Failure Patterns

## Failure 1: iOS update → other_unclear

This was the largest error group.

Many messages mention an iOS version or say that they have already updated the device.

For example, a message such as:

> "I just updated it to software version 11.0.3."

contains useful information for a human, but the classifier can treat it as unclear because it does not explicitly describe the problem.

### Hypothesis

The classifier needs conversation context and better handling of version/update language.

---

## Failure 2: service_support → other_unclear

Messages such as:

* "DM sent"
* "Waiting for response"
* "Spoke with an advisor"
* "No satisfactory resolution"

are support-workflow messages rather than technical descriptions.

### Hypothesis

A separate conversation-state or support-workflow model could help identify these messages.

---

## Failure 3: feature_how_to → other_unclear

Some customers ask about features indirectly instead of explicitly saying "How do I...?"

These messages often require the previous conversation to understand what feature they are referring to.

### Hypothesis

Feature names and conversation context should be included more strongly in the classifier.

---

## Failure 4: app_issue → device_performance

Both categories contain words such as:

* crashing
* freezing
* restarting
* not working

The difference is whether the problem is specific to an application or affects the device more generally.

### Hypothesis

The classifier needs stronger entity-level understanding of the application involved.

---

## Failure 5: connectivity → other_unclear

Very short messages such as:

> "Wi-Fi"

or

> "I did try to update via Wi-Fi"

can be difficult to classify correctly.

### Hypothesis

The system needs better context handling and should distinguish the actual problem from incidental words such as "Wi-Fi."

---

# 16. A Particularly Important Failure

One example was:

> "Iphone 6S, IOS 11.0.3"

The model predicted `other_unclear` with approximately **0.84 confidence**.

The retrieved similarity was **1.0**, and the decision engine allowed `AUTO_HANDLE`.

However, the manually assigned intent was `device_performance`.

This is exactly the type of example that influenced the conservative design of the decision engine.

It demonstrates that:

**high classifier confidence + high retrieval similarity does not automatically mean that the system understands the customer's problem.**

---

# 17. Human Evaluation

I manually evaluated a 30-example audit set.

The sample contained:

* all 17 `AUTO_HANDLE` examples
* 13 randomly selected `ESCALATE` examples

This was intentionally a decision-focused audit rather than a representative estimate of the full 200-example Golden Set.

The average scores were:

| Dimension    |        Score |
| ------------ | -----------: |
| Groundedness | **4.40 / 5** |
| Relevance    | **4.17 / 5** |
| Helpfulness  | **3.63 / 5** |
| Safety       | **4.80 / 5** |
| Overall      | **3.90 / 5** |

The weakest dimension was **helpfulness**.

This makes sense because many escalated examples use safe but generic replies such as asking the customer to contact Apple Support.

Those replies are safe, but they do not necessarily solve the customer's problem.

---

# 18. Human Evaluation by Decision

The difference between automatic and escalated cases was also useful:

| Decision    | Groundedness | Relevance | Helpfulness | Safety |  Overall |
| ----------- | -----------: | --------: | ----------: | -----: | -------: |
| AUTO_HANDLE |         4.82 |      4.53 |        4.41 |   4.88 | **4.41** |
| ESCALATE    |         3.85 |      3.69 |        2.62 |   4.69 | **3.23** |

This suggests that the decision engine is behaving conservatively in the right direction: the examples that it allowed through as `AUTO_HANDLE` generally received better human ratings.

However, because the human audit contains only 30 examples and was intentionally stratified by decision, I would not treat these numbers as a statistically representative estimate of overall system quality.

---

# 19. LLM-as-Judge

I implemented an LLM-as-judge evaluation harness using five dimensions:

1. Groundedness
2. Relevance
3. Helpfulness
4. Safety
5. Overall quality

The judge is designed to score each dimension from 1–5 and provide a short reason.

I also implemented a comparison script that will compare LLM and human scores using:

* exact agreement
* agreement within one point
* mean absolute error
* quadratic weighted Cohen's kappa

However, the LLM judge could not be executed because the OpenAI API account had no available credits.

I intentionally did not fabricate judge scores or judge-human agreement numbers.

The harness is ready to run when API access is available.

---

# 20. What Is Misleading About My Headline Number?

The headline number is:

**36.50% intent accuracy.**

It is useful, but it can also be misleading if presented without context.

First, the classifier is trained using noisy silver labels rather than a fully human-labeled training set.

Second, the Golden Set is imbalanced.

Third, many Twitter messages are extremely short and depend on previous conversation context.

Fourth, the current LLM was not available during evaluation, so the reply-generation results are based on the historical fallback rather than actual LLM generation.

Finally, retrieval similarity can be high even when the retrieved case is not actually useful.

For this reason, I would not describe the system as a production-ready 36.5%-accurate support agent.

A more honest summary is:

> The initial TF-IDF classifier improves substantially over the majority baseline, but the system still struggles with context-dependent customer messages. The strongest current result is the conservative safety behavior and the ability to ground fallback responses in historical AppleSupport interactions, rather than the raw classification accuracy alone.

---

# 21. What I Would Improve Next Week

If I had another week, I would focus on the following improvements.

## 1. Improve the training labels

The current silver-label approach is the biggest data-quality limitation.

I would manually label a larger and more balanced training set, especially for:

* `ios_update`
* `service_support`
* `feature_how_to`
* `app_issue`
* `connectivity`

---

## 2. Use conversation context for classification

Instead of classifying only the target tweet, I would include the previous one or two customer/support messages.

This should particularly help with:

* "Done"
* "Yes"
* version numbers
* "Still happening"
* "I did"
* "Thanks"

---

## 3. Improve retrieval beyond TF-IDF

The current retrieval is mainly lexical.

I would test semantic embeddings so that messages with different wording but the same problem can retrieve each other.

I would also evaluate retrieval using manually labeled relevance rather than relying mainly on silver intent labels.

---

## 4. Improve the decision model

The current decision engine is rule-based.

A next version could combine:

* intent confidence
* retrieval relevance
* historical response quality
* message completeness
* context availability
* reply safety checks

into a calibrated confidence score.

---

## 5. Improve reply usefulness

The human evaluation showed that helpfulness was the weakest dimension.

The system currently prefers safety over aggressive troubleshooting.

The next version should provide more useful next steps when there is strong historical evidence, while still preventing unsupported claims.

---

## 6. Run the LLM judge

Once API access is available, I would run the existing judge harness and compare its scores with the human audit.

This would complete the judge-human agreement part of the evaluation.

---

# 22. Final Takeaway

This project started as a simple intent-classification problem, but the evaluation showed that the harder problem is actually **understanding support conversations with incomplete context**.

The system currently has a working end-to-end pipeline:

```text
Customer message
      ↓
Intent classification
      ↓
Historical case retrieval
      ↓
Grounded reply generation / fallback
      ↓
Safety checks
      ↓
AUTO_HANDLE or ESCALATE
```

The initial classifier achieved **36.50% accuracy and 28.42% macro F1**, improving over the majority baseline of **24.00% accuracy and 3.23% macro F1**.

The human audit produced an overall quality score of **3.90/5**, with particularly strong safety at **4.80/5**.

The biggest remaining weaknesses are context-dependent messages, noisy silver labels, lexical retrieval, and reply helpfulness.

I would therefore treat this implementation as a **working prototype and evaluation framework**, rather than a production-ready customer-support system.
