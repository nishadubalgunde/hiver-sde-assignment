# Decision Log

This file records the main non-obvious decisions made while building the AppleSupport support agent.

## 1. Why I chose AppleSupport

I chose AppleSupport because it has enough support volume and issue diversity for classification, retrieval, and reply-generation experiments without mixing multiple brands.

## 2. Why I did not use the complete dataset for everything

The source contains millions of tweets. I filtered to AppleSupport and the directly connected customer-support data so the experiments remained practical and reproducible.

## 3. How I reconstructed conversations

I used `response_tweet_id` and `in_response_to_tweet_id` to reconstruct conversations. This matters because many support messages are short and require context. The reconstruction can still be incomplete when source messages are missing.

## 4. Why I used 12 intents

I wanted a small, practical taxonomy that covered the main issues in the manually labeled sample. The 12 intents are `ios_update`, `battery_power`, `app_issue`, `device_performance`, `audio_call_issue`, `connectivity`, `apple_id_account`, `app_store_purchase`, `music_media`, `feature_how_to`, `service_support`, and `other_unclear`.

Escalation remains separate because it describes handling policy, not issue type.

## 5. How I created the Golden Set

I created a 200-example manually labeled Golden Set with a fixed random seed. Golden examples were kept out of classifier training and historical retrieval so evaluation remained untouched.

## 6. Why I kept conversation context in the Golden Set

The Golden Set stores the target message together with preceding conversation context. This improves manual labeling and makes context-dependent failures visible instead of silently treating them as standalone text.

## 7. Why I created silver labels

Manual labeling of tens of thousands of messages was not practical. I therefore used deterministic rules to create silver labels for the remaining messages. They are explicitly treated as noisy training data, not ground truth.

## 8. Why I started with TF-IDF + Logistic Regression

I chose a simple, fast, inspectable model as the first classifier. It gives a meaningful baseline before adding more complex semantic methods.

## 9. How I handled historical support knowledge

Replies are grounded in historical AppleSupport customer → support-response pairs. The goal is to reuse observed support behavior rather than inventing unsupported policies.

## 10. Why similarity alone was not enough

Failure analysis showed that a similarity score of `1.0` can still correspond to an irrelevant case when messages are extremely short or contain only a version number. Similarity is therefore evidence, not proof of correctness.

## 11. Why I tested intent-aware retrieval

I added a reranking bonus when the retrieved case's silver intent matched the predicted intent. Top-1 improved from 29.5% to 33.0%, but Top-3/Top-5 decreased. Because the extra signal is noisy, I did not adopt it as the main strategy.

## 12. Why I tested conversation-aware retrieval

I tested retrieval using previous conversation messages because short messages need context. Context-aware retrieval did not improve the overall result enough, although a hybrid variant helped some weak examples.

## 13. Why the agent is conservative about AUTO_HANDLE

Automatic handling requires high intent confidence, a non-obviously-context-dependent message, compatible historical evidence, sufficient similarity, and a usable reply. Otherwise the system escalates. This favors safety over unsupported confidence.

## 14. Why I added an LLM fallback

The generator supports an external LLM but remains runnable when the API is unavailable. It falls back to an intent-compatible historical response and, if necessary, a safe escalation message. During final evaluation the API returned `credit_balance_exhausted`, so I used the fallback and did not describe those replies as LLM-generated.

## 15. Why I did not fabricate LLM-as-judge results

The LLM judge and human-agreement scripts are implemented, but the judge could not execute because the API account had no credits. I therefore reported the completed human audit separately and made no unsupported claim about judge-human agreement.