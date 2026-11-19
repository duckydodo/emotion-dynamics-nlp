# Application Demonstration Script: Context-Aware Conversational Emotion Analytics

**Repository**: `emotion-dynamics-nlp`  
**Target Audience**: Academic Reviewers, Course Instructors, and Project Committee  
**Total Running Time**: ~6 Minutes (5–7 Minute Range)  
**Deliverable**: `outputs/reports/application_demo_script.md`  
**Prerequisites**: Application running live at `http://localhost:8000` (`docker compose` or `emotion-api` container)  

---

## 1. Overview and Demonstration Goals

This script guides the presenter through a live, professional demonstration of the **Emotion Dynamics Analyzer** platform. The objective is to demonstrate how natural language processing models can extract subtle emotional shifts and conversational dynamics from multi-turn dialogues, while rigorously evaluating trade-offs between isolated transformer baselines, causal recurrent architectures, and relational graph neural networks.

---

## 2. Timed Demonstration Agenda (6-Minute Walkthrough)

| Minute | Phase / Target Screen | Primary Focus | Key Talking Points |
| :---: | :--- | :--- | :--- |
| **0:00 – 1:00** | **Introduction & Problem Framing** | Motivation & Research Problem | The limitation of sentence-level sentiment; the role of conversational context, speaker identity, and emotional inertia. |
| **1:00 – 2:00** | **Screen 1: Home / Import** | Data Ingestion & Normalization | Frictionless multi-format ingestion (TXT, WhatsApp, Instagram, CSV); live canonical schema preview and audit. |
| **2:00 – 3:00** | **Screen 2: Conversation Viewer** | Chat Bubbles & Inspection | Modern messaging stream, speaker alignment, predicted expressed emotion badges, uncalibrated softmax warnings, probability drawers. |
| **3:00 – 4:00** | **Screen 3: Emotion Dynamics** | Affective Trajectory & Turning Points | Chart.js probability trajectory, transition math ($T=S+P$), speaker inertia vs partner influence, interactive turning point navigation. |
| **4:00 – 5:00** | **Screen 4: Model Comparison** | Multi-Model Dissent & Agreement | Synchronized turn stream across DistilBERT, DialogueRNN, and DialogueGCN; pairwise agreement matrix; disagreement filter. |
| **5:00 – 6:00** | **Screen 5: Research & Limitations** | Verified Benchmarks & Boundary Safeguards | Architecture specifications, Cohort A verified table ($p=0.0036$), confusion matrix switcher, TV domain limits, non-clinical ethics. |

---

## 3. Step-by-Step Demonstration Walkthrough

### Minute 0:00 – 1:00 | Introduction & Problem Framing
**Presenter Action**: Open browser to `http://localhost:8000`. Point out the header showing *"3 Models Online"* and the clean dark slate theme.

**Talking Points**:
> *"Good morning, Professor. In standard NLP sentiment analysis, sentences are classified in total isolation. But in human communication, emotion is relational and context-dependent. An utterance like 'What did you do?' can be neutral surprise or explosive anger depending entirely on conversational history and who is speaking.*  
>  
> *Today, I am demonstrating our conversational emotion analytics platform. It incorporates three distinct neural model families: a fine-tuned DistilBERT baseline, a causal DialogueRNN tracking speaker states over time, and our DialogueGCN relational graph convolutional network. We will inspect how these models capture affective trajectories, identify conversational turning points, and audit their empirical benchmarks on MELD."*

---

### Minute 1:00 – 2:00 | Screen 1: Home / Import (Data Ingestion & Normalization)
**Presenter Action**:
1. Point to the dropzone and paste area.
2. Click the quick sample button: **"1. Conflict & Resolution"**.
3. Point to the automatically rendered **Conversation Ingestion Preview** card below.

**Sample Dialogue Used**:
```text
Alice: I can't believe you did that without consulting me first.
Bob: What? I didn't do anything unreasonable.
Alice: Stop lying to me! Look at the email you sent to the entire team.
Bob: I'm not lying. I had to make an urgent decision because you were away.
Alice: You completely undermined my position on the project.
Bob: I apologize if it came across that way. That was never my intention.
Alice: It's hard to trust your intentions right now.
Bob: Let's sit down with the team lead and clear up the misunderstanding.
Alice: Fine. I appreciate you taking responsibility.
```

**Talking Points**:
> *"Here on the Import screen, our system supports four real-world export formats: raw plain text, WhatsApp chats, Instagram JSON, and structured CSV files.*  
>  
> *I’ve loaded a representative conflict dialogue between Alice and Bob. As soon as text is provided, our backend parses it into a canonical schema. In the preview card below, you see that the system detected 9 messages, 2 speakers, verified timestamps, and audited 0 warnings.*  
>  
> *We have selected DialogueGCN as our active model. Now let's click 'Analyze Conversation Dynamics'."*

---

### Minute 2:00 – 3:00 | Screen 2: Conversation Viewer (Annotated Message Stream)
**Presenter Action**:
1. Click **"Analyze Conversation Dynamics"**.
2. The UI smoothly transitions to **Screen 2: Conversation Viewer**.
3. Highlight the blue scientific disclaimer banner at top.
4. Point out the left-aligned bubbles (Alice) and right-aligned bubbles (Bob).
5. On Turn #1 (Alice), point to the **Predicted: Anger** badge and **Conf: 88% ℹ️**.
6. Click **"Distribution ▾"** on Turn #1 to expand the inline 7-class probability bar chart.

**Talking Points**:
> *"The system has evaluated the dialogue and presented it in a modern conversation stream.*  
>  
> *Notice our blue scientific banner at top: we explicitly use the wording 'predicted expressed emotion', reminding users that these models predict learned statistical text patterns from conversational corpora, not internal psychological states.*  
>  
> *On Turn #1, DialogueGCN predicts 'Anger' with 88% confidence. When I expand the probability drawer, we see the full distribution across all seven Ekman classes: 88% Anger, 11% Surprise, and nearly zero for the remaining classes. Furthermore, hovering over the info icon clarifies that softmax confidence represents relative logit margins rather than calibrated Bayesian truth.*  
>  
> *Now, let's explore how emotion evolves across the conversation in Screen 3."*

---

### Minute 3:00 – 4:00 | Screen 3: Emotion Dynamics (Affective Trajectories & Turning Points)
**Presenter Action**:
1. Click the **"3. Emotion Dynamics"** tab.
2. Point out the 4 KPI cards: Total Transitions (8), Emotion Shifts (7), Persistence (1), Shift Rate (88%).
3. Hover cursor over the **Chart.js Trajectory Area Chart** across turns 1 to 9.
4. Point to the **Speaker Interaction Patterns** card (same-speaker inertia vs cross-speaker partner influence).
5. Point to the **Salient Turning Points** card showing Turn #7 (*"Let's sit down with the team lead..."*).
6. Click the button: **"Inspect Turn #7 ➔"**.
7. The browser jumps back to Screen 2, smooth-scrolls to Bob's Turn #7 bubble, and highlights it with a blue pulsing ring.

**Talking Points**:
> *"Screen 3 translates turn-level classifications into conversational dynamics.*  
>  
> *At the top, we compute the fundamental dynamics invariant: Total Transitions (8) equals Emotion Shifts (7) plus Persistence (1), yielding an 88% shift rate.*  
>  
> *Our trajectory chart visualizes the probability mass shifting over time: red for anger during Alice's opening, transition through Bob's defensive neutral responses, and de-escalation toward the end.*  
>  
> *Notice our Turning Points algorithm identified Turn #7 as a critical inflection point, where Bob's de-escalation shifted the dialogue from Anger to Neutral. When I click 'Inspect Turn #7', the interface navigates straight back to the viewer and pulses the exact bubble where the turning point occurred."*

---

### Minute 4:00 – 5:00 | Screen 4: Model Comparison (Dissent & Multi-Model Consensus)
**Presenter Action**:
1. Click **"4. Model Comparison"**.
2. Point out the **Consensus Rate: 78%** banner.
3. Show the **$3 \times 3$ Pairwise Agreement Matrix**.
4. Check the box: **"Show Disagreements Only"** (filters view down from 9 turns to 2 turns).
5. Inspect Turn #1 side-by-side: DistilBERT (`Surprise 62%`), DialogueRNN (`Surprise 71%`), DialogueGCN (`Anger 88%`).

**Talking Points**:
> *"Here on Screen 4, we execute our unified multi-model comparison service. The identical conversation turns were passed simultaneously through DistilBERT, DialogueRNN, and DialogueGCN.*  
>  
> *Our pairwise matrix shows DistilBERT and DialogueRNN agreed on 100% of turns for this sample, while DialogueGCN exhibited 78% agreement. When I toggle 'Show Disagreements Only', we immediately isolate the two turns where models diverged.*  
>  
> *Look at Turn #1: Alice says, 'I can't believe you did that without consulting me first.' The utterance-only DistilBERT model sees 'can't believe' and classifies it as Surprise. But DialogueGCN, modeling relational graph edges between Alice and Bob's subsequent dispute, correctly identifies the expressed emotion as Anger.*  
>  
> *This demonstrates the tangible advantage of modeling conversational topology over isolated sentences."*

---

### Minute 5:00 – 6:00 | Screen 5: Research & Benchmarks (Evaluation & Limitations)
**Presenter Action**:
1. Click **"5. Research & Benchmarks"**.
2. Show the **Model Architecture Specifications** cards (DistilBERT 66.96M, DialogueRNN 670.8K, DialogueGCN 452.5K params).
3. Scroll to the **Verified Benchmark Evaluation (Canonical MELD Cohort A, N=2,328)** table.
4. Highlight that DialogueGCN achieves Macro F1 = 0.4385 and Weighted F1 = 0.6009 with $p=0.0036$ statistical significance over Context-1.
5. In the **Empirical Confusion Matrix Heatmap**, click the tabs: `[DialogueGCN]`, `[DialogueRNN]`, and `[DistilBERT]`.
6. Conclude by pointing to the **Methodological Boundaries & Research Limitations** panel.

**Talking Points**:
> *"Finally, Screen 5 exposes the complete research evidence base backing this application.*  
>  
> *All numbers here are dynamically loaded from our verified evaluation artifacts—zero hardcoded metrics. On the canonical MELD Cohort A subset of 2,328 turns, DialogueGCN achieves the highest Macro F1 of 0.4385 and Weighted F1 of 0.6009. More importantly, DialogueGCN uses 32% fewer parameters than DialogueRNN (452K vs 671K) and runs in under 7 milliseconds per turn.*  
>  
> *We have verified with non-parametric permutation testing that DialogueGCN yields statistically significant gains over naive context concatenation (p=0.0036). However, in the spirit of scientific integrity, we also note that its 0.71% point improvement over the single-turn baseline is not statistically significant (p=0.5879).*  
>  
> *Lastly, we emphasize the boundaries of this work: MELD is scripted TV comedy dialogue, vocal prosody is absent in text, and categorical emotion labels are simplified discrete approximations. This platform is an NLP research tool, not a clinical diagnostic system.*  
>  
> *Thank you, Professor. I welcome any questions."*

---

## 4. Prepared Answers for Anticipated Reviewer Questions

### Q1: "Why did naive context concatenation in DistilBERT perform worse than single utterances?"
> **Answer**: *"Concatenating previous text $(t-1 \oplus t)$ increases sequence length and dilutes the immediate emotional cues with previous context, causing the transformer to over-predict the majority class (`neutral`). Its Macro F1 dropped from 0.4314 to 0.3912 ($p=0.0036$). This proves that conversational context cannot be naively appended as raw tokens; it requires structured inductive biases like GRU states or graph edges."*

### Q2: "Can DialogueGCN be used in a real-time chatbot?"
> **Answer**: *"Not in its current bidirectional configuration. DialogueGCN uses a context window of past 10 turns and future 10 turns, which makes it ideal for retrospective conversation analytics. For real-time streaming conversational bots where future turns do not yet exist, our causal DialogueRNN architecture is the appropriate choice."*

### Q3: "How does the application prevent user data leakage?"
> **Answer**: *"We conducted a full privacy audit using headless Chrome CDP. User conversation text is stored exclusively in transient browser heap memory (`AppState`) and is never written to `localStorage`, cookies, or disk logs. Clicking 'Clear Session' immediately wipes all in-memory dialogue data."*
