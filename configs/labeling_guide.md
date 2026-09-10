\# AppleSupport Intent Labeling Guide



\## Purpose



This guide defines the rules used to manually label the Golden Evaluation Set.



Each example receives exactly one intent.



The label should represent the customer's primary support problem, using the surrounding conversation context when necessary.



\---



\## 1. ios\_update



Use when the customer's problem involves:



\- iOS updates

\- failed updates

\- unavailable updates

\- update installation

\- problems caused by an iOS update

\- questions about iOS versions



Examples:



\- "My iPhone won't update to the latest iOS"

\- "After updating iOS my phone stopped working"

\- "It says I'm up to date but 11.1.1 is available"



Do NOT use for general device problems when the update is unrelated.



\---



\## 2. battery\_power



Use for:



\- battery draining quickly

\- charging problems

\- battery percentage problems

\- unexpected shutdowns

\- device not powering on

\- power-related issues



Examples:



\- "My battery only lasts half a day"

\- "My iPhone won't turn on"

\- "Battery drains overnight"



\---



\## 3. app\_issue



Use when a specific application is:



\- crashing

\- freezing

\- failing to open

\- failing to load

\- behaving incorrectly



Examples:



\- "Facebook keeps freezing"

\- "The App Store crashes whenever I open it"

\- "Snapchat won't load"



If the problem is with the entire operating system rather than an individual app, prefer `device\_performance`.



\---



\## 4. device\_performance



Use for general device/system problems such as:



\- freezing

\- touchscreen problems

\- random behavior

\- system instability

\- device becoming unresponsive

\- unexpected screen/app switching



Examples:



\- "My phone keeps opening apps by itself"

\- "The touchscreen randomly stops responding"

\- "My Mac keeps freezing"



If the problem is specifically caused by an iOS update, prefer `ios\_update`.



\---



\## 5. audio\_call\_issue



Use for:



\- phone call problems

\- microphone problems

\- speaker problems

\- static/crackling

\- inability to hear someone

\- other person cannot hear the customer



Examples:



\- "I can't hear the person I'm calling"

\- "People can't hear me"

\- "My speakers are crackling"



\---



\## 6. connectivity



Use for:



\- Wi-Fi

\- Bluetooth

\- wireless connectivity

\- accessory connection problems



Examples:



\- "My Wi-Fi keeps disconnecting"

\- "My Beats won't connect"

\- "Bluetooth stopped working"



\---



\## 7. apple\_id\_account



Use for:



\- Apple ID

\- account login

\- account access

\- authentication

\- Apple account problems



Examples:



\- "I can't sign into my Apple ID"

\- "My Apple ID password isn't working"



\---



\## 8. app\_store\_purchase



Use for:



\- App Store

\- application downloads

\- purchases

\- payments

\- store credit



Examples:



\- "My App Store purchase isn't showing"

\- "I can't download an app"

\- "What happened to my App Store credit?"



\---



\## 9. music\_media



Use for:



\- Apple Music

\- iTunes

\- iBooks

\- music syncing

\- media syncing

\- media playback/library problems



Examples:



\- "My iTunes purchases won't sync"

\- "My Apple Music library disappeared"



\---



\## 10. feature\_how\_to



Use for questions asking how to use an Apple feature, setting, or control.



Examples:



\- "How do I enable this setting?"

\- "How do I use Face ID?"

\- "How do I change this option?"



This is primarily a HOW-TO question, not a malfunction report.



\---



\## 11. service\_support



Use for:



\- requesting human support

\- service appointments

\- callback requests

\- contacting Apple support

\- repair/service requests

\- situations where the customer explicitly asks for support



Examples:



\- "Can someone call me?"

\- "I need to book a service appointment"

\- "Can you help me with this issue?"



\---



\## 12. other\_unclear



Use when:



\- there is not enough information to identify the intent

\- the message is only a contextual follow-up

\- the message is ambiguous

\- the message does not fit the supported intents



Examples:



\- "Yes"

\- "Last night"

\- "Still happening"

\- "That didn't work"



Use conversation context first. Only assign `other\_unclear` if the intent remains genuinely unclear.



\---



\# Label Priority Rules



When multiple labels appear possible, use these rules:



1\. Specific cause beats generic symptom.

2\. Specific application problem beats general device problem.

3\. iOS-update-related problems use `ios\_update`.

4\. Wi-Fi/Bluetooth problems use `connectivity`.

5\. Call/audio problems use `audio\_call\_issue`.

6\. Explicit Apple ID/account problems use `apple\_id\_account`.

7\. Store/payment/purchase problems use `app\_store\_purchase`.

8\. Music/iTunes/iBooks problems use `music\_media`.

9\. Explicit how-to questions use `feature\_how\_to`.

10\. Requests for appointments/callbacks/human support use `service\_support`.

11\. If context is insufficient, use `other\_unclear`.



\---



\# Golden Set Rules



The Golden Set must be kept separate from data used for retrieval, training, or development.



The 200 examples should include:



\- common intents

\- less common intents

\- ambiguous messages

\- short/context-dependent messages

\- difficult examples

\- noisy Twitter messages

\- examples requiring conversation context



The final Golden Set must be manually reviewed before evaluation.

