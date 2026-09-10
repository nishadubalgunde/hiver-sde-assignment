# Decision Log

This file records the main decisions I made while building the AppleSupport support agent, including decisions that affected the design, evaluation, and limitations of the system.

## 1. Why I chose AppleSupport

I chose AppleSupport as the brand for this project because it has a large number of customer-support conversations and covers many different types of issues.

This gave me enough data to experiment with intent classification, historical case retrieval, and reply generation without mixing multiple brands.

---

## 2. Why I did not use the complete dataset for everything

The original Twitter dataset contains millions of tweets. Running every experiment on the entire dataset would add a lot of processing time without being necessary for this assignment.

I therefore filtered the dataset to AppleSupport and worked with the relevant customer-support conversations.

---

## 3. How I reconstructed conversations

The dataset contains individual tweets along with response relationships. I used `response_tweet_id` and `in_response_to_tweet_id` to connect the messages and reconstruct conversations.

This was important because many customer messages are very short. A message such as "Done" or a software version number is difficult to understand without seeing the previous messages.

One limitation is that the dataset does not always contain every message from a real-world conversation, so some reconstructed conversations may still be incomplete.

---

## 4. Why I used 12 intents

I wanted the intent set to be small enough to be practical but broad enough to cover the main AppleSupport issues in the sample.

The final 12 intents are:

* `ios_update`
* `battery_power`
* `app_issue`
* `device_performance`
* `audio_call_issue`
* `connectivity`
* `apple_id_account`
* `app_store_purchase`
* `music_media`
* `feature_how_to`
* `service_support`
* `other_unclear`

I also kept escalation separate from the intent because escalation is a decision about how the agent should handle the message, not what the message is about.

---

## 5. How I created the Golden Set

I created a manually labeled Golden Set of 200 customer examples.

I used a fixed random seed so that the sample can be reproduced.

The Golden Set was kept separate from the larger training data so that it could be used as an untouched evaluation set.

---

## 6. Why I kept conversation context in the Golden Set

I stored the messages leading up to each target message along with the target itself.

This helped during manual labeling and failure analysis because some Twitter messages cannot be understood on their own.

For example, "Done" may have completely different meanings depending on what AppleSupport asked in the previous message.

---

## 7. Why I created silver labels

Manually labeling tens of thousands of messages would not have been practical for this assignment.

Instead, I created deterministic rule-based silver labels for the remaining customer messages and used them to train the initial classifier.

I treated these labels as **noisy training data**, not ground truth.

This distinction is important because the silver dataset is heavily imbalanced, with roughly 77.7% of examples falling into `other_unclear`.

---

## 8. Why I started with TF-IDF + Logistic Regression

I deliberately started with a simple model instead of immediately using a large language model for intent classification.

TF-IDF + Logistic Regression is:

* fast to train
* easy to reproduce
* easy to inspect
* a useful baseline for comparison

It also gave me a clear starting point before adding retrieval and generation.

---

## 9. How I handled historical support knowledge

For reply generation, I retrieve similar historical customer-support cases and use the corresponding AppleSupport responses as evidence.

The idea is that the agent should learn from how AppleSupport actually responded in the past instead of inventing a completely new support policy.

---

## 10. Why similarity alone was not enough

During testing, I found that a high similarity score does not necessarily mean that the retrieved case is actually useful.

Some examples had a similarity score of `1.0` but were still irrelevant because the customer message was extremely short or contained only a version number.

This was an important finding because it showed that retrieval confidence should not be treated as proof of semantic correctness.

---

## 11. Why I experimented with intent-aware retrieval

I tested whether retrieved cases should receive an additional score when their silver intent matched the predicted intent.

This improved top-1 retrieval from 29.5% to 33.0%, but top-3 and top-5 performance became slightly worse.

Because the intent signal itself comes from noisy silver labels, I decided not to rely on this reranking approach as the main retrieval strategy.

---

## 12. Why I tested conversation-aware retrieval

I also tested retrieval using previous conversation messages instead of only the current customer message.

The reasoning was straightforward: more context should help with short or ambiguous messages.

However, the experiment did not produce a meaningful overall improvement. A hybrid approach helped some weak/context-dependent examples but did not improve the overall result enough to replace the simpler approach.

---

## 13. Why the agent is conservative about AUTO_HANDLE

I did not want the agent to automatically respond just because the classifier had a high confidence score.

For `AUTO_HANDLE`, the system also checks whether:

* the intent confidence is high enough
* the message is not obviously context-dependent
* relevant historical evidence exists
* the historical evidence is sufficiently similar
* a usable reply is available

If these checks fail, the system escalates instead.

This makes the system more conservative, which is preferable for a customer-support setting where an incorrect confident response can be worse than asking a human to take over.

---

## 14. Why I added a fallback when the LLM is unavailable

The reply generator supports an external LLM, but I also wanted the project to remain runnable when the API is unavailable.

If the LLM cannot be called, the system falls back to a relevant historical AppleSupport response when possible. If that is also unavailable, it uses a safe escalation message.

During my evaluation, the OpenAI API returned a `credit_balance_exhausted` error because there were no API credits available.

Therefore, the 200-example evaluation used the fallback path rather than pretending that the replies were generated by the LLM.

---

## 15. Why I did not fabricate LLM-as-judge results

The assignment asks for an LLM-as-judge evaluation and agreement with human ratings.

I implemented the judge and the comparison script, but the API could not execute because the account had no available API credits.

I therefore did **not** create or report fake judge scores.

Instead, I completed a 30-example human audit and recorded the results separately. The judge comparison can be run later when API access is available.

---

## 16. Why I report macro F1 along with accuracy

The Golden Set is not evenly distributed across intents.

Because of this, accuracy alone can be misleading. A model can perform reasonably on the majority class while performing poorly on smaller intents.

For that reason, I report both accuracy and macro F1.

The current classifier achieved:

* **Accuracy: 36.50%**
* **Macro F1: 28.42%**

The majority baseline achieved:

* **Accuracy: 24.00%**
* **Macro F1: 3.23%**

The result is better than the majority baseline, but it also shows that the classifier still needs significant improvement.

---

## 17. What I learned from the evaluation

The biggest problem was not simply the classifier itself.

Many AppleSupport conversations contain very short, context-dependent messages such as version numbers, "Done", "Thanks", or follow-ups to an earlier question.

These messages are difficult to classify from the target message alone.

The failure analysis also showed confusion between:

* `ios_update` and `other_unclear`
* `service_support` and `other_unclear`
* `feature_how_to` and `other_unclear`
* `app_issue` and `device_performance`
* `connectivity` and `other_unclear`

These observations will guide the next iteration of the system.
