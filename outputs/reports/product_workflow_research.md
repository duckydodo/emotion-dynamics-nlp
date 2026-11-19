# Product Workflow Research: MosaicChats and Hume AI
**Comparative Analysis, Interaction Paradigms, and Architectural Roadmap for Conversational Emotion Recognition**

**Document Version:** 1.0.0  
**Date:** November 14, 2026  
**Status:** Completed Research Investigation  
**Target Platform:** Emotion Dynamics (Conversational Emotion Analysis & Research)  
**Evaluated Models:** DistilBERT (Utterance-only), DialogueRNN (Sequential Recurrent), DialogueGCN (Graph Convolutional)

---

## 1. Executive Summary

This research report investigates the public-facing product workflows, user interaction paradigms, and analytical methodologies of two leading emotion and conversational intelligence platforms:
1. **MosaicChats** (`https://mosaicchats.com/`): A consumer-oriented chat analysis platform focused on relationship dynamics, sentiment timelines, and pop-psychological profiles from messaging exports (WhatsApp, Instagram, iMessage, Telegram, Messenger).
2. **Hume AI** (`https://www.hume.ai/`): A developer- and enterprise-focused affective computing platform rooted in Cowen & Keltner’s multidimensional semantic space theory, offering expressive voice interfaces (EVI), vocal prosody evaluation, and audio decision models.

### Key Findings
* **Interaction Paradigms:** MosaicChats demonstrates an intuitive file ingestion flow for heterogeneous text formats and temporal sentiment curves, but relies heavily on scientifically unsupported pop-psychology metrics (MBTI, compatibility percentages, love languages, romantic advice coaching, and commercial product/movie recommendations). Hume AI demonstrates rigorous dimensional modeling, continuous temporal probability streams, and acoustic prosodic alignment, but is tailored for voice-native developer APIs rather than post-hoc text conversation analysis.
* **Fit for Emotion Dynamics:** Our platform (`Emotion Dynamics`) operates on text-based conversational emotion recognition grounded in empirical NLP benchmarks (MELD: 7 emotion classes—*anger, disgust, fear, joy, sadness, surprise, neutral*) evaluated across three verified model families: **DistilBERT**, **DialogueRNN**, and **DialogueGCN**.
* **Actionable Insights:**
  1. *Adaptable UX Patterns:* Multi-format conversational upload guidance, bidirectional interactive timeline-to-message navigation, speaker symmetry metrics, and granular temporal emotion distributions.
  2. *Deliberately Excluded Features:* Pop-psychology personality profiling, compatibility indices, romantic advice chatbots ("Myrah"), and media/interest/movie extraction—all of which violate scientific validity and our research mandate.
  3. *Future Multimodal Trajectory:* An architectural bridge to multimodal conversational analysis (fusing speech prosody via open-source acoustic models like openSMILE or Wav2Vec 2.0 with our existing DialogueRNN and DialogueGCN backbones) and domain extension into customer service analytics (caller escalation velocity and agent de-escalation effectiveness).

---

## 2. MosaicChats Workflow Analysis

### 2.1 Onboarding and File Ingestion Workflow
MosaicChats structures its user acquisition around a frictionless three-step funnel:
1. **Source Selection & Export Guidance:** The landing page and upload portal (`/check-chat`) explicitly highlight support for five primary consumer messaging ecosystems: **WhatsApp**, **iMessage**, **Instagram DMs**, **Telegram**, and **Facebook Messenger**, alongside `.txt`, `.zip`, `.json`, and `.db` extensions. In-app contextual links ("*Don't have the file yet? Use 'How do I export?' below*") route users to platform-specific step-by-step export guides (e.g. exporting WhatsApp without media, requesting Instagram Data Download JSON).
2. **Drag-and-Drop Dropzone:** A clear, modern upload container with immediate format validation.
3. **Freemium Teaser vs. Pro Paywall:** The ingestion immediately generates a "Free Preview" displaying shallow conversational statistics (message counts, word clouds, inside jokes, and high-level strengths) before requiring payment or account creation to unlock the full report.
4. **Data Handling & Privacy Notice:** Prominently advertises client-side anonymization, transit encryption, and explicit user controls to delete chat history from the dashboard.

### 2.2 Conversation Presentation
The full report visualizes conversation dynamics across several panels:
* **Sentiment Timeline:** A week-by-week dual-stream curve tracking the emotional tone of both participants over time, highlighting emotional peaks, cooling-off intervals, and recovery duration following relationship friction.
* **Engagement & Balance Telemetry:** Bar and donut charts quantifying:
  * Message volume share (who texts more).
  * Response-time distribution (median response latency per speaker).
  * Conversation initiation rates (who reaches out first).
  * Activity heatmaps (active communication hours across days of the week).
* **Conversational Artifacts:** Word clouds of frequently used vocabulary, identified "inside jokes", and "viral moments" (memorable exchanges highlighted as styled message snippets).

### 2.3 Message-Level Inspection and Navigation
* **Timeline Drill-Down:** Clicking on inflection points or sentiment spikes on the timeline reveals the underlying conversational turns associated with that timeframe.
* **Turning Point Indicators:** Sudden drops or spikes in sentiment are flagged as notable events, prompting users to inspect what occurred during that conversational phase.

### 2.4 Communication of Uncertainty and Scientific Limitations
* **Marketing Overstatement:** MosaicChats positions its platform with claims such as *"Decode your relationship"* and *"AI reads your real conversations and reveals the patterns, sentiment, and compatibility you can't see yourself"*.
* **Lack of Error Bounds:** The interface presents single deterministic scores (e.g., "87% Compatibility", "INFP Personality", "Anxious Attachment Style") without confidence intervals, model calibration caveats, or data distribution warnings.
* **Absence of Domain Shift Acknowledgment:** The platform applies generalized LLM prompts to consumer chat without explaining prompt variability, hallucination risks, or the absence of clinical psychological validation.

---

## 3. Hume AI Workflow Analysis

### 3.1 Core Products and affect Computing Workflows
Hume AI provides an enterprise-grade affective computing suite:
1. **Empathic Voice Interface (EVI):** A real-time, low-latency speech-to-speech conversational foundation model that detects vocal tone and emotional expression to respond adaptively with appropriate prosody.
2. **Expression Measurement API:** Multimodal models that analyze audio, video, and text to output continuous emotional scores across high-dimensional semantic spaces.
3. **Audio Decision Model API:** An audio-native reasoning layer that produces structured, probabilistic judgments directly from speech without relying on intermediate ASR transcriptions.
4. **VoiceEQ Benchmark Platform:** A diagnostic framework for evaluating voice AI systems across real-world criteria including emotional alignment, prosody matching, conversational turn-taking, and latency.

### 3.2 Multimodal Prosody and Text Alignment
* **Acoustic Dimension Modeling:** Rather than reducing emotion to simplistic positive/negative valence or 6 basic Ekman categories, Hume's speech prosody models evaluate vocal delivery across up to 48 dimensional expression dimensions (e.g., *Amusement, Distress, Excitement, Contemplation, Doubt, Frustration, Joy, Relief, Sadness, Triumph*), grounded in Alan Cowen and Dacher Keltner's semantic space research.
* **Temporal Chunking & Speaker Diarization:** Predictions are segmented into micro-intervals (`time.begin` to `time.end` down to millisecond precision) and grouped by diarized speaker identifiers (`grouped_predictions[i].id = "speaker_0"`).
* **Continuous Probabilistic Outputs:** Each dimension is returned as a continuous floating-point probability or intensity score (typically `0.0` to `1.0`), reflecting statistical correlation with human perceptual judgments rather than arbitrary binary classifications.

```json
{
  "time": { "begin": 1.24, "end": 4.58 },
  "text": "I really didn't expect this outcome, but it makes sense now.",
  "emotions": [
    { "name": "Realization", "score": 0.8124 },
    { "name": "Surprise (positive)", "score": 0.6432 },
    { "name": "Relief", "score": 0.4510 },
    { "name": "Confusion", "score": 0.1205 },
    { "name": "Distress", "score": 0.0318 }
  ]
}
```

### 3.3 Customer Service Applications and Conversational Flow
Hume AI targets enterprise customer contact centers and automated support workflows:
* **Frustration & Escalation Detection:** Continuous tracking of vocal tension, pitch escalation ($F_0$ rise), and speech rate acceleration flags agitated customers before explicit verbal complaints emerge.
* **Agent De-escalation Telemetry:** Measuring whether agent responses (tone warmth, pacing, pause management) successfully reduce customer distress scores on subsequent turns.
* **Turn-Taking & Interruption Management:** Detecting barge-ins, awkward silences, and cross-talk to ensure smooth conversational transitions without robotic overlap.
* **Escalation Reduction:** Routing calls dynamically based on acoustic affect scores, prioritizing human supervisor handoffs only when emotional escalation breaches specified thresholds.

### 3.4 Presentation of Audio + Text Insights
* **Synchronized Playback Scrubbing:** As audio plays, an interactive timeline highlights the active utterance while synchronized horizontal bar gauges dynamically reflect the shifting probability distribution of expressed emotions.
* **Transcript Color Coding:** Transcripts are annotated with subtle tone badges and color-coded acoustic markers, allowing reviewers to visually scan dialogues for affective turning points.

### 3.5 Developer APIs and Integration Paradigms
* **Bidirectional WebSocket API:** Used for real-time streaming audio (PCM chunks) with millisecond-level response latency for live agents and EVI integrations.
* **REST Batch Processing:** Used for post-call analytics and historical call recording processing, returning structured JSON manifests.
* **Webhook Event Architecture:** Asynchronous notifications upon batch completion or critical event triggers (e.g. extreme caller distress alert).
* **Human Feedback Custom Models:** Tooling allowing enterprise teams to register custom domain labels and fine-tune classifiers against human ratings.

---

## 4. Evidence-Backed Feature Inventory

| Product | Feature Name | Category | Public Evidence / Observation | Intended User Problem | Data Modality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MosaicChats** | Multi-Platform Export Ingestion | Ingestion & Parsing | Upload UI supports `.txt`, `.zip`, `.json`, `.db` with logos for WhatsApp, iMessage, IG, Telegram, Messenger | Users struggle to export/clean proprietary chat data | Text / Zip / DB |
| **MosaicChats** | In-App Export Walkthroughs | Documentation / UX | Contextual modal links for *"How do I export?"* | Non-technical users need platform-specific instructions | Static Guide |
| **MosaicChats** | Sentiment Over Time Curve | Timeline Visualization | Weekly dual-speaker curve tracking emotional valence and recovery phases | Visualizing relationship highs and lows over time | Text-derived valence |
| **MosaicChats** | Response Dynamics Telemetry | Conversational Metrics | Charts displaying response latency distribution and initiation counts | Identifying communication imbalance or ghosting | Timestamps & Counts |
| **MosaicChats** | Viral Moments / Inside Jokes | Content Retrieval | Extracted highlight cards showcasing humorous or memorable turns | Surfacing memorable conversational memories | Text snippets |
| **MosaicChats** | Personalized Media Recommendations | Recommendation Engine | Surfacing songs, movies, books, and gifts based on conversation topics | Users looking for entertainment or gift ideas | Text topics |
| **MosaicChats** | Compatibility & MBTI Scoring | Pop-Psych Profiling | Percentage score (e.g. 88%) and Myers-Briggs 4-letter type assignment | Satisfying consumer curiosity about relationship viability | Uncalibrated inference |
| **MosaicChats** | "Myrah" AI Relationship Coach | Generative Chatbot | Interactive conversational agent answering questions about the chat | Users seeking relationship guidance and interpretation | RAG over chat history |
| **Hume AI** | Continuous Prosodic Measurement | Affective Modeling | REST/WebSocket API returning 48 dimensional expression scores | Measuring subtle vocal cues that text transcripts omit | Raw Audio / Speech |
| **Hume AI** | Temporal Utterance Alignment | Data Structuring | JSON schema grouping predictions by speaker and millisecond timestamps | Synchronizing acoustic affect with text turns | Audio + Transcript |
| **Hume AI** | VoiceEQ Diagnostic Platform | Model Benchmarking | Standardized evaluation measuring naturalness, emotion match, turn-taking | Benchmarking speech-to-speech AI systems | Multimodal benchmarks |
| **Hume AI** | Audio Decision Model API | Audio Reasoning | Audio-native query interface answering structured questions without ASR | Eliminating error compounding from ASR transcriptions | Raw Audio |
| **Hume AI** | Real-time De-escalation Tracking | Customer Service | Detection of acoustic agitation, vocal pitch variance, and de-escalation | Preventing customer churn and managing call center escalations | Audio streams |
| **Hume AI** | Human Feedback Fine-Tuning | Model Customization | Platform tooling to register custom scoring criteria from human judges | Aligning model outputs with specific domain rubrics | Human annotations |

---

## 5. Comparison of the Two Products

| Dimension | MosaicChats | Hume AI | Emotion Dynamics (Our Project) |
| :--- | :--- | :--- | :--- |
| **Target Audience** | Everyday consumers, dating app users, couples | Enterprise developers, ML engineers, AI research labs | NLP researchers, conversation analysts, academic evaluators |
| **Theoretical Foundation** | Pop-psychology (MBTI, Love Languages, Attachment Theory) | Empirical affect science (Cowen & Keltner multidimensional semantic space) | Computational NLP & Dialogue Systems (MELD, Ekman 7-class taxonomy) |
| **Input Modalities** | Exported text archives (WhatsApp TXT, Instagram JSON, etc.) | Raw audio streams, video recordings, spoken dialogue | Canonical structured text conversations (Message, Speaker, Timestamp) |
| **Model Architectures** | Proprietary LLM prompting / RAG pipeline | Foundation multimodal models (EVI, Octave, Decision Model) | **DistilBERT**, **DialogueRNN** (GRU-based), **DialogueGCN** (GCN-based) |
| **Data Handling & Privacy** | Consumer cloud upload, transient storage, deletion option | Enterprise HIPAA/SOC2 compliance, private cloud pipelines | 100% Local / On-premise execution via Docker; no cloud exfiltration |
| **Evaluation & Metrics** | No published scientific evaluation or error rates | Real World VoiceEQ benchmarks, human listener correlation | Matched-cohort evaluation: Accuracy, Weighted F1, Macro F1, confusion matrices |
| **Presentation Style** | Casual, colorful, gamified cards, relationship coach persona | High-density developer console, telemetry streams, probability distributions | Restrained, professional research dashboard with dark aesthetic (`#171B18`) |
| **Communication of Limits**| Low / None; presents speculative estimates as facts | Moderate; probabilistic scores and benchmark documentation | High; explicit MELD domain shift notices, "predicted expressed emotion" framing |

---

## 6. Features Worth Adapting for Emotion Dynamics

All adaptations are strictly scoped to our **text-only conversational platform** and our **three verified model families** (*DistilBERT*, *DialogueRNN*, *DialogueGCN*).

### Feature A: Multi-Format Conversation Ingestion Wizard with Contextual Validation
1. **What it does:** Provides an intuitive, robust drag-and-drop ingestion interface supporting raw text, WhatsApp TXT, Instagram JSON, and CSV, complete with immediate preview validation and format diagnosis.
2. **Concrete evidence:** Observed on MosaicChats (`/check-chat`), where clear platform badges and format guidance reduce friction for non-technical users.
3. **User problem it solves:** Users frequently possess exports from different chat applications with divergent timestamp conventions, multiline message wrapping, and irregular speaker delimiters.
4. **Fit with our 3 models and text-only scope:** High. Our backend already possesses verified canonical parsers for Text, Instagram, WhatsApp, and CSV. Adapting this into a guided import step with preview verification directly feeds our canonical `Conversation` schema required by DistilBERT, DialogueRNN, and DialogueGCN.
5. **Independent implementation plan:**
   - Enhance the Home/Import screen (`static/index.html`) with interactive format cards.
   - Display dynamic file preview tables (Speaker, Timestamp, Message Text) prior to confirming analysis.
   - Return actionable parser error suggestions (e.g., "Timestamp format unrecognized; expected `[DD/MM/YY, HH:MM:SS]`").
6. **Complexity assessment:** **Low**. The backend ingestion services (`src/ingestion/`) are already built and verified; only the frontend wizard and error display require refinement.
7. **Scientific or ethical concerns:** Low. Ingestion formatting is purely mechanical. Data remains strictly on the user's local instance.

### Feature B: Bidirectional Interactive Turning Point Navigation
1. **What it does:** Seamlessly connects analytical timeline spikes (e.g. sharp shifts from neutral to anger/sadness) with the exact message bubbles in the conversation viewer. Clicking a point on the timeline scrolls to and highlights the message; clicking a message highlights its coordinate on the timeline.
2. **Concrete evidence:** Observed in MosaicChats' timeline drill-down and Hume AI's time-synchronized transcript scrubber.
3. **User problem it solves:** Users cannot easily determine what specific conversational context caused an emotional shift without tedious manual scrolling between charts and transcripts.
4. **Fit with our 3 models and text-only scope:** Perfect fit. DialogueRNN and DialogueGCN compute contextual speaker states specifically across sequential turns. Inspecting turning points allows researchers to evaluate whether sequential context (DialogueRNN) or graph-relational context (DialogueGCN) correctly captured an emotional transition where utterance-only DistilBERT failed.
5. **Independent implementation plan:**
   - Standardize DOM element IDs in the conversation viewer: `#msg-<message_id>`.
   - In Chart.js / Canvas timeline click handlers, emit a custom event with `message_index`.
   - Scroll the Conversation Viewer smoothly using `scrollIntoView({ behavior: 'smooth', block: 'center' })` and apply a temporary CSS highlight ring.
   - Conversely, attach an intersection observer or click listener to messages to highlight the corresponding point on the timeline.
6. **Complexity assessment:** **Low to Medium**. Requires synchronized event handling between Chart.js instances and the DOM list.
7. **Scientific or ethical concerns:** None. Enhances interpretability and transparency of model predictions.

### Feature C: Conversational Structural Dynamics Metrics (Response & Initiation Balance)
1. **What it does:** Calculates descriptive conversational structural telemetry alongside emotion predictions: speaker turn shares, mean utterance lengths, dialogue burstiness, and speaker transition frequency.
2. **Concrete evidence:** Observed in MosaicChats' engagement statistics and Hume AI's turn-taking analysis.
3. **User problem it solves:** Conversation dynamics are defined not only by emotion labels, but also by structural asymmetries (e.g. one speaker dominating turns or exhibiting rapid mono-syllabic replies).
4. **Fit with our 3 models and text-only scope:** High. DialogueGCN explicitly builds its adjacency matrix using speaker identity and windowed turn transitions. Exposing structural turn metrics provides empirical insight into why DialogueGCN constructs specific edge weights across inter-speaker versus intra-speaker dependencies.
5. **Independent implementation plan:**
   - Implement a lightweight analytics utility in `src/analytics/structure.py` that computes:
     - Turn distribution percentage per speaker.
     - Average token / character length per speaker.
     - Inter-speaker transition matrix (probability of speaker B speaking after speaker A).
   - Display these metrics as restrained summary statistics in the Emotion Dynamics screen.
6. **Complexity assessment:** **Low**. Requires simple mathematical aggregation over the canonical message list.
7. **Scientific or ethical concerns:** Low, provided metrics are labeled strictly as descriptive statistics without speculative behavioral judgments.

### Feature D: Multidimensional Temporal Emotion Heatmap & Probability Stream
1. **What it does:** Visualizes the full 7-class probability distribution across all conversational turns as a stacked temporal area chart or heatmap, rather than reducing each turn to a single discrete argmax label.
2. **Concrete evidence:** Observed in Hume AI's continuous probability output streams and multivariable expression graphs.
3. **User problem it solves:** Discretized argmax emotion classifications mask model ambiguity (e.g. an utterance predicted as *Anger* at 0.36 probability with *Disgust* at 0.34 and *Neutral* at 0.30). Users need to see probability spreads to assess prediction confidence.
4. **Fit with our 3 models and text-only scope:** Direct fit. All three of our models (DistilBERT, DialogueRNN, DialogueGCN) output 7-dimensional softmax probability vectors for every utterance.
5. **Independent implementation plan:**
   - In the Emotion Dynamics screen, render a multi-class stream chart displaying the 7 MELD classes (*neutral, anger, disgust, fear, joy, sadness, surprise*) mapped to muted, distinct hues.
   - Allow users to toggle between "Dominant Class View" and "Probability Heatmap View".
6. **Complexity assessment:** **Medium**. Requires rendering a 7-series synchronized stream chart or matrix heatmap in Chart.js.
7. **Scientific or ethical concerns:** High scientific benefit. Directly mitigates the illusion of false classification certainty by transparently displaying probability distribution entropy.

### Feature E: Model Disagreement & Comparative Uncertainty Highlighting
1. **What it does:** Automatically flags conversational turns where the three model families disagree (e.g. DistilBERT predicts *Neutral*, DialogueRNN predicts *Anger*, DialogueGCN predicts *Disgust*), highlighting where conversational context alters predictions.
2. **Concrete evidence:** Inspired by Hume AI's VoiceEQ comparative model diagnostics and evaluation framework.
3. **User problem it solves:** Researchers and analysts want to rapidly pinpoint turns where sequential context (DialogueRNN) or conversational graph structure (DialogueGCN) makes a definitive difference over isolated text (DistilBERT).
4. **Fit with our 3 models and text-only scope:** Core strength of our unified three-model architecture. Our comparison service (`/api/models/compare`) already runs identical messages across all available models.
5. **Independent implementation plan:**
   - In the Model Comparison screen, compute an agreement index $A_i = 1$ if $M_1 = M_2 = M_3$, else $0$.
   - Display an "Analysis of Divergence" filter that surfaces only consensus turns ($A_i = 1$) or conflicting turns ($A_i = 0$).
   - Present a turn-level confusion matrix showing inter-model disagreement.
6. **Complexity assessment:** **Low to Medium**. The data already exists in the comparison response payload; requires frontend filtering and visual flag badges.
7. **Scientific or ethical concerns:** None. Fosters critical evaluation of AI models rather than passive acceptance.

---

## 7. Features Deliberately Excluded

To preserve research integrity, avoid pseudoscientific claims, and maintain our strict focus on computational NLP and conversational emotion recognition, the following features are **strictly out of scope and permanently excluded**:

```
+-------------------------------------------------------------------------------+
|                    DELIBERATELY EXCLUDED FEATURE MATRIX                       |
+------------------------------------+------------------------------------------+
| Excluded Feature                   | Reason for Rejection                     |
+------------------------------------+------------------------------------------+
| 1. Relationship Compatibility %    | Pseudoscientific; lacks psychometric      |
|                                    | validity; uncalibrated commercial gimmick.|
+------------------------------------+------------------------------------------+
| 2. MBTI & Personality Profiling    | Myers-Briggs lacks empirical support;    |
|                                    | chat text is insufficient for psychometrics|
+------------------------------------+------------------------------------------+
| 3. Automated Relationship Coaching | Unregulated psychological intervention;   |
|    (e.g., "Myrah" chatbot)         | hallucination risks; safety liability.   |
+------------------------------------+------------------------------------------+
| 4. Movie, Song, Book & Gift        | Trivial entity extraction; commercial     |
|    Recommendation Extraction       | distraction; dilutes affective NLP focus.|
+------------------------------------+------------------------------------------+
| 5. Psychological Certainty Claims  | Ignores MELD TV-dialogue domain shift;   |
|                                    | overconfidence without clinical context. |
+------------------------------------+------------------------------------------+
```

### Detailed Justifications:

1. **Relationship Compatibility Scoring (MosaicChats):**
   * *Mechanism:* MosaicChats computes a single percentage (e.g. "84% Match") ostensibly measuring marital or romantic longevity.
   * *Justification for Exclusion:* There is zero empirical validity supporting LLM-generated compatibility percentages from unstandardized text exports. Such metrics represent arbitrary heuristics that mislead vulnerable users regarding interpersonal relationships. Our platform analyzes *conversational emotion dynamics*, not romantic destiny.

2. **Personality Profiling: MBTI, Attachment Styles, Love Languages (MosaicChats):**
   * *Mechanism:* Categorizing users into Myers-Briggs types (e.g. ENFP vs INTJ) or attachment classifications (anxious, avoidant, secure) from chat logs.
   * *Justification for Exclusion:* MBTI has been widely discredited within academic psychology due to poor test-retest reliability and lack of bimodal trait distributions. Inferring clinically sensitive psychological constructs (such as attachment trauma) from conversational text violates ethical standards in affective computing.

3. **Automated Romantic Advice & Relationship Coaching Chatbots (MosaicChats' "Myrah"):**
   * *Mechanism:* An unconstrained generative AI agent offering advice on whether a partner is losing interest or how to resolve an argument.
   * *Justification for Exclusion:* Generative coaching agents present severe safety risks, including hallucinated relationship dynamics, sycophantic reinforcement of user biases, and potential harm in emotionally volatile or abusive contexts. Our product is an **analytical research instrument**, not an autonomous relationship counselor.

4. **Personalized Media and Interest Extraction (Movies, Songs, Books, Gifts) (MosaicChats):**
   * *Mechanism:* Scanning messages for mentions of movies, bands, or consumer products to generate gift guides or entertainment recommendations.
   * *Justification for Exclusion:* While technically straightforward via Named Entity Recognition (NER), media and product recommendations are an irrelevant commercial distraction. Adding product/movie recommendation engines clutters the UI, introduces commercial bias, and completely departs from our mission of advancing multi-turn conversational emotion recognition and sequential context modeling.

5. **Psychological Certainty and Definitive Emotional Labeling:**
   * *Mechanism:* Presenting predictions as absolute ground truth of a speaker's internal emotional state.
   * *Justification for Exclusion:* In accordance with modern affective computing ethics, all outputs must be explicitly designated as **"predicted expressed emotion"** based on visible text cues. Furthermore, because our models were trained on MELD (conversational television dialogue from *Friends*), the interface must maintain visible disclaimers regarding domain shift when processing natural interpersonal chat.

---

## 8. Future Audio and Customer-Service Research Direction

While our immediate implementation is strictly text-based, Hume AI's success demonstrates the value of acoustic prosody in conversational understanding. This section outlines a research roadmap for extending Emotion Dynamics to audio and customer service domains.

### 8.1 Text-to-Speech & Speech-to-Text Integration Pipeline
To transition from text-only transcripts to multimodal dialogue without compromising architectural modularity:
1. **Audio Ingestion & Preprocessing:** Accept standard multi-channel audio recordings (WAV, FLAC, MP3). Run voice activity detection (VAD) using Silero-VAD to segment active speech regions.
2. **Speech-to-Text with Word-Level Timestamps:** Integrate OpenAI Whisper (or open-source Conformer ASR) to transcribe audio into text while preserving micro-timestamps ($t_{start}, t_{end}$) per word and turn.
3. **Speaker Diarization:** Apply PyAnnote.audio to cluster acoustic segments into speaker identities (`speaker_0`, `speaker_1`), resolving turn ownership without user intervention.

### 8.2 Acoustic Prosody Feature Extraction
Rather than relying on closed-source APIs, an independent research system can extract acoustic affect representations using two complementary approaches:
* **Knowledge-Based Acoustic Feature Sets:** Utilize `openSMILE` to extract the standardized **eGeMAPS** (extended Geneva Minimalistic Acoustic Parameter Set, 88 features) covering:
  * Frequency parameters: Pitch ($F_0$), jitter, formant frequencies ($F_1, F_2, F_3$).
  * Energy/Amplitude parameters: Shimmer, loudness, harmonics-to-noise ratio (HNR).
  * Spectral parameters: Alpha ratio, Hammarberg index, spectral slope.
* **Self-Supervised Pretrained Representations:** Extract dense temporal representations using `Wav2Vec 2.0`, `HuBERT`, or `WavLM` pretrained on conversational speech corpora.

### 8.3 Multimodal Architectural Fusion with DialogueRNN and DialogueGCN
Our existing sequential and graph backbones are mathematically well-suited for multimodal fusion:

```
                          MULTIMODAL FUSION ARCHITECTURE

    Utterance t (Audio)                    Utterance t (Text)
           │                                       │
           ▼                                       ▼
    ┌──────────────┐                        ┌──────────────┐
    │  WavLM /     │                        │  DistilBERT  │
    │  openSMILE   │                        │  Tokenizer   │
    └──────┬───────┘                        └──────┬───────┘
           │ Acoustic Embedding                    │ Textual Embedding
           │ a_t in R^{d_a}                        │ e_t in R^{d_t}
           └───────────────────┬───────────────────┘
                               │ Concatenation / Cross-Attention
                               ▼
                      Fused Utterance Vector
                         u_t = [e_t ; a_t]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   ┌─────────────────┐                  ┌───────────────────┐
   │   DialogueRNN   │                  │    DialogueGCN    │
   │  Speaker GRU    │                  │  Relational Graph │
   │  & Emotion GRU  │                  │  & Speaker Edges  │
   └────────┬────────┘                  └─────────┬─────────┘
            ▼                                     ▼
   Sequential Prosodic                   Graph Relational Affect
   Context Prediction                    Interaction Prediction
```

1. **DialogueRNN Multimodal Adaptation:**
   * At turn $t$, the input representation becomes $u_t = \mathbf{W}_f [\mathbf{e}_{text, t} \,\|\, \mathbf{a}_{audio, t}] + \mathbf{b}_f$.
   * The speaker GRU updates speaker state $q_t^{(s)}$ using both what was said and how it was vocalized.
   * Vocal distress or agitation directly modulates the emotion GRU transition, capturing acoustic emotional momentum.
2. **DialogueGCN Multimodal Adaptation:**
   * Node features are initialized with multimodal fused vectors $u_i$.
   * Edge weights $\alpha_{ij}$ in the graph convolution layer can incorporate acoustic divergence:
     $$\alpha_{ij} = \text{Softmax}\left(\frac{(u_i \mathbf{W}_Q)(u_j \mathbf{W}_K)^T}{\sqrt{d}} - \lambda \cdot \Delta \tau_{ij}\right)$$
     where $\Delta \tau_{ij}$ represents the pause duration or conversational latency between turns $i$ and $j$.

### 8.4 Customer Service Affect Telemetry
In commercial and customer service settings, conversation dynamics offer immediate diagnostic value:
* **Caller Escalation Velocity ($v_{esc}$):** The rate of increase in negative emotional probability (e.g. *Anger* or *Distress*) over consecutive caller turns:
  $$v_{esc} = \frac{\Delta P(\text{Anger})}{\Delta t}$$
* **Agent De-escalation Effectiveness ($\eta_{de-esc}$):** The observed reduction in caller negative valence following specific agent interventions:
  $$\eta_{de-esc} = P(\text{Anger})_{caller, t-1} - P(\text{Anger})_{caller, t+1}$$
* **Sentiment Divergence Index:** The euclidean distance between agent emotional composure and caller emotional volatility, identifying situations where an overly cheerful agent frustrates an agitated caller ("toxic positivity mismatch").
* **Turn-Taking Latency & Interruption Frequency:** Monitoring cross-talk intervals and response latency to flag system lag or agent impatience.

---

## 9. Recommended Product Improvements

Ranked by **Scientific Relevance**, **UX Value**, and **Implementation Complexity**:

```
+----+------------------------------------+----------+-------------+------------+----------+
| Rk | Proposed Improvement               | Screen   | Relevance   | Complexity | Priority |
+----+------------------------------------+----------+-------------+------------+----------+
| 1  | Bidirectional Turning Point Nav    | Dynamics | High        | Low        | P0       |
|    | (Timeline click jumps to message)  | & Viewer | (Interpret) |            |          |
+----+------------------------------------+----------+-------------+------------+----------+
| 2  | Model Disagreement Inspector       | Model    | High        | Low        | P0       |
|    | (Highlight turns where models clash| Compare  | (Research)  |            |          |
+----+------------------------------------+----------+-------------+------------+----------+
| 3  | Multi-Format Ingestion Wizard      | Home /   | High        | Low-Medium | P1       |
|    | (Format detection & table preview) | Import   | (Usability) |            |          |
+----+------------------------------------+----------+-------------+------------+----------+
| 4  | Structural Conversation Metrics    | Dynamics | Medium      | Low        | P1       |
|    | (Turn ratios, token counts, balance| & Viewer | (Analytical)|            |          |
+----+------------------------------------+----------+-------------+------------+----------+
| 5  | 7-Class Probability Stream Chart   | Dynamics | High        | Medium     | P2       |
|    | (Full softmax distribution view)   |          | (Uncertainty|            |          |
+----+------------------------------------+----------+-------------+------------+----------+
| 6  | Local Data Deletion & Session Clear| Home /   | Medium      | Low        | P2       |
|    | (Instant privacy purge button)     | Global   | (Privacy)   |            |          |
+----+------------------------------------+----------+-------------+------------+----------+
| 7  | Audio Extension Architecture Spec  | Research | High        | High       | Future   |
|    | (Multimodal fusion design study)   | / Docs   | (Roadmap)   |            | Spec     |
+----+------------------------------------+----------+-------------+------------+----------+
```

### Prioritization Rationale:
* **Top Priority (P0 - Immediate Low-Hanging Fruits):** Bidirectional turning point navigation and model disagreement highlighting utilize data already generated by our backend. They provide massive interpretive power for academic reviewers and professors without modifying core ML inference pipelines.
* **Secondary Priority (P1 - Polish & Ergonomics):** Ingestion table previews and structural conversation telemetry enhance the consumer-friendliness of the application while preventing ingestion failures.
* **Tertiary Priority (P2 - Advanced Analytics):** The 7-class probability stream provides full transparency into model uncertainty, reinforcing that our application avoids deceptive psychological certainty.

---

## 10. Sources, Citations, and Documented Evidence

### 10.1 MosaicChats Documentation & Public Endpoints
* **Platform Homepage:** `https://mosaicchats.com/` (Accessed November 14, 2026).
  * *Evidence:* Taglines, advertised dimensions (01 Sentiment Timeline, 02 Compatibility Score, 03 MBTI Analysis, 04 Response Patterns, 05 Activity Heatmap, 06 Viral Moments, "Myrah" relationship guide).
* **Ingestion Portal:** `https://mosaicchats.com/check-chat` (Accessed November 14, 2026).
  * *Evidence:* Multi-format support (`.txt`, `.zip`, `.json`, `.db`), platform list (WhatsApp, iMessage, Instagram, Telegram, Messenger), drag-and-drop file upload target.
* **Feature Breakdown & Recommendations:** `https://mosaicchats.com/chat-analysis` and related sitemap pages.
  * *Evidence:* Automated media recommendations (songs, films/movies, books, gifts) extracted from conversation topics; freemium teaser structure.
* **Privacy & Mission:** `https://mosaicchats.com/privacy-mission` & `/privacy-policy`.
  * *Evidence:* Anonymization claims, cloud encryption, user deletion controls.

### 10.2 Hume AI Documentation & Developer Resources
* **Official Website:** `https://www.hume.ai/` (Accessed November 14, 2026).
  * *Evidence:* Empathic Voice Interface (EVI), Audio Decision Model API, VoiceEQ benchmarking suite, Octave TTS model.
* **Developer Platform & Documentation:** `https://dev.hume.ai/` & `https://app.hume.ai/`.
  * *Evidence:* Multimodal Expression Measurement API, 48-dimensional emotion space taxonomy, millisecond temporal alignment schemas (`time.begin`, `time.end`, `speaker_0`).
* **Research Foundation:**
  * Cowen, A. S., & Keltner, D. (2017). *Self-report captures 27 distinct categories of emotion bridged by continuous gradients*. Proceedings of the National Academy of Sciences (PNAS).
  * Cowen, A. S., et al. (2020). *Sixteen facial expressions occur in similar contexts worldwide*. Nature.
* **Customer Service & Enterprise Applications:**
  * Real World VoiceEQ documentation (`https://hume.ai/voiceeq`): Standardized evaluation criteria for conversational turn-taking, prosodic alignment, and de-escalation efficiency.

### 10.3 Core Codebase References (Emotion Dynamics)
* **Model Inference Registry:** `src/models/registry.py` (Defines `distilbert`, `dialoguernn`, and `dialoguegcn`).
* **Comparison Service:** `src/models/comparison.py` (Multi-model consensus and divergence evaluator).
* **Ingestion Parsers:** `src/ingestion/` (`text_parser.py`, `whatsapp_parser.py`, `instagram_parser.py`, `csv_parser.py`).
* **Canonical Schema:** `src/schemas/conversation.py` (`Message`, `Conversation`).
* **Frontend Controller:** `static/app.js` and `static/index.html`.
* **Verified Benchmark Dataset:** MELD (Multimodal EmotionLines Dataset, Chen et al., 2018).

---
*Report compiled independently for the Emotion Dynamics Research Platform. No proprietary code or private assets were extracted or replicated during this study.*
