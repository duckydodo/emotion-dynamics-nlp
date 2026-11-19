# Known Limitations and Methodological Boundaries

**Repository**: `emotion-dynamics-nlp`  
**Deliverable**: `outputs/reports/known_limitations.md`  
**Target Audience**: Academic Reviewers, Research Advisors, and Ethical Review Boards  
**Status**: Comprehensive Disclosure of Scientific and Operational Boundaries  

---

## 1. Corpus Domain Limitations (MELD & Scripted Television Dialogue)

### 1.1 The *Friends* Sitcom Bias
The primary empirical benchmark used throughout this project is the **Multimodal EmotionLines Dataset (MELD)** (Poria et al., 2019). While widely accepted in the literature as a standard benchmark for Emotion Recognition in Conversation (ERC), MELD has significant domain-specific characteristics:
- **Theatrical Acting & Comedic Tropes**: Utterances are taken from scripted television comedy (*Friends*). Dialogue lines are written by television screenwriters and delivered by professional actors to maximize comedic timing, punchlines, and theatricality.
- **Inflated Sarcasm and Irony**: Sitcom dialogue features a disproportionately high incidence of sarcastic irony, hyperbole, and deadpan delivery. Sarcastic statements (e.g., Chandler Bing’s signature phrasing) often have surface text that is completely antithetical to the underlying emotional subtext.
- **Non-Generalizability to Real-World Discourse**: Models trained exclusively on MELD cannot be assumed to generalize zero-shot to natural workplace conversations, legal transcripts, customer support chats, or clinical interviews, where emotional expression is more restrained, subtle, or domain-specialized.

---

## 2. Domain Shift to Real-World Digital Messaging

### 2.1 Lexical and Syntactic Discrepancies
When users import real-world digital chat exports (e.g., WhatsApp, Instagram, SMS) into the application, several severe domain shift phenomena occur:
- **Punctuation and Syntax Decay**: Real-world chat features lowercase-only text, omitted punctuation, run-on clauses, and intentional typographic stretching (e.g., *"nooooo pleaaase"*). In contrast, MELD transcripts are professionally edited with clean orthography and standard punctuation.
- **Emojis and Non-Verbal Artifacts**: Real digital communication relies heavily on emojis, stickers, animated GIFs, voice notes, and reaction taps. In our text-only pipeline, non-verbal markers are either stripped, tokenized as unknown tokens, or noted in warnings (e.g., `<Media omitted>`).
- **Turn Granularity & Multi-Message Bursts**: Human messaging involves rapid bursts of short fragments sent in quick succession by the same speaker (e.g., four consecutive two-word messages) rather than the single, cohesive dialogue turns characteristic of theatrical television scripts.

---

## 3. Subjectivity and Noise in Emotion Annotations

### 3.1 The Limits of the 7-Class Ekman Taxonomy
The classification schema is constrained to seven discrete categories:
$$\mathcal{Y} = \{\text{anger}, \text{disgust}, \text{fear}, \text{joy}, \text{neutral}, \text{sadness}, \text{surprise}\}$$
This taxonomy introduces fundamental theoretical limitations:
- **Discrete vs. Dimensional Affect**: Modern psychological theories (e.g., Russell’s Circumplex Model of Affect, Barrett’s Theory of Constructed Emotion) demonstrate that human affect is continuous, dynamic, and multi-dimensional (Valence, Arousal, Dominance), rather than cleanly partitioned into mutually exclusive discrete bins.
- **Blended and Ambivalent Emotions**: Real human interactions frequently elicit mixed affective states—such as bitter-sweetness, nostalgic joy-sadness, or nervous excitement. Forcing an annotator or a softmax classifier to choose exactly one mutually exclusive discrete label induces substantial classification noise.
- **Inter-Annotator Disagreement**: Subjective perception of emotion varies across cultural backgrounds, genders, and personal temperaments. MELD exhibits an inter-annotator Fleiss' $\kappa \approx 0.43$, indicating only moderate agreement among human annotators. Consequently, there is an empirical ceiling to model accuracy on this benchmark.

---

## 4. Confidence Calibration and Softmax Fallacies

### 4.1 Uncalibrated Softmax Probabilities
The application outputs turn-by-turn confidence percentages derived directly from the neural network's final softmax layer:
$$p(y = c \mid u_t) = \frac{\exp(z_c)}{\sum_{j} \exp(z_j)}$$
It is a well-established mathematical fact (Guo et al., 2017) that modern deep neural networks, particularly when fine-tuned with cross-entropy loss, are **not calibrated**:
- **Overconfident Predictions**: The model frequently outputs confidence scores $> 90\%$ on ambiguous or incorrect turns. A 90% softmax confidence score **does not** imply a 90% Bayesian posterior probability that the human speaker actually experienced that emotion.
- **Logit Margins, Not True Uncertainty**: Softmax scores reflect relative logit margins across the pre-defined 7 classes, not calibrated probability measures of true human psychological intent.
- **Mitigation in UI**: To prevent users from misinterpreting these numbers, the interface prominently displays explicit disclaimer tooltips stating: *"Notice: Softmax confidence is uncalibrated and represents relative logit margin, not mathematical probability of true emotion."*

---

## 5. Severe Class Imbalance and Minority Class Ceilings

### 5.1 The Neutral Majority Bias
MELD exhibits an acute class imbalance:

```
[Neutral: 47.9%] ████████████████████████████████████████
[Joy:     15.4%] ████████████
[Anger:   13.2%] ██████████
[Surprise:10.9%] ████████
[Sadness:  8.0%] ██████
[Disgust:  2.7%] ██
[Fear:     1.9%] █
```

This distribution severely skews model behavior:
- **Majority Class Attraction**: In ambiguous turns, models exhibit a strong statistical prior pulling predictions toward `neutral`. In the DialogueGCN confusion matrix on Cohort A, 885 of 1,116 true neutral turns were correctly identified, but 80 true sadness turns and 78 true anger turns were falsely predicted as neutral.
- **Minority Sample Scarcity**: In our canonical test cohort ($N = 2,328$), there are only **44 instances of Fear** and **63 instances of Disgust**. With such small sample sizes, a shift of just 3 or 4 turn predictions alters per-class F1 by $\pm 5\%$. Models struggle to learn robust feature invariants for minority classes without class-weighting (which in turn degrades overall macro performance by inducing false alarms).

---

## 6. Methodological Limits of Interpreting Emotion Transitions

### 6.1 Transition Math vs. Deep Psychological Causation
The platform computes conversational dynamics metrics, including Total Transitions ($T$), Emotion Shifts ($S$), Persistence ($P$), Shift Rate ($R_{\text{shift}}$), Same-Speaker Inertia ($R_{\text{inertia}}$), and Cross-Speaker Influence ($R_{\text{cross-shift}}$).

While these mathematical formulations provide structured conversational metrics, their interpretation requires caution:
- **Markovian Simplification**: The calculation of shifts ($y_t \ne y_{t-1}$) assumes a first-order Markovian step. In reality, emotional inertia in human beings operates across longer temporal horizons, latent conversational goals, and personal history that cannot be captured by adjacent discrete label diffs.
- **Correlation vs. Causal Interpersonal Influence**: When Partner B shifts to Anger after Partner A's utterance, the model records a cross-speaker shift. However, statistical co-occurrence does not establish cognitive causality. Partner B's anger may have been triggered by an external environmental distraction, a pre-existing bias, or an unvoiced realization unrelated to Partner A's immediate words.
- **Turning Point Heuristics**: The turning point algorithm identifies turns where confidence deltas or discrete labels change significantly. These are mathematical inflection points in the model's posterior surface; they must not be conflated with clinical psychological breakthroughs or legal turning points.

---

## 7. Modality and Architectural Topology Boundaries

### 7.1 Text-Only Transcripts vs. Multimodal Signals
Human emotional communication is intrinsically multimodal (Mehrabian, 1971):
- **Acoustic Pitch and Prosody**: Sarcastic praise vs. genuine praise is distinguished almost entirely by fundamental frequency ($F_0$), vocal intensity, and speech tempo.
- **Visual Micro-Expressions and Gaze**: Subtle facial muscle movements (Action Units) convey contempt, suppressed sadness, or feigned smiles.
- **Text-Only Limitation**: Our models operate solely on textual transcripts. An utterance like *"Oh, wonderful!"* spoken with dripping cynicism will inevitably be classified by text models as `joy` or `surprise`, because the cynical signal existed entirely in the audio waveform.

### 7.2 DialogueGCN Offline Topology vs. DialogueRNN Causal Steaming
- **DialogueGCN (Bidirectional Graph)**: DialogueGCN constructs edges across past 10 turns and future 10 turns ($\max(1, i-10) \le j \le \min(N, i+10)$). This allows the model to leverage future responses to disambiguate earlier statements. Consequently, DialogueGCN is strictly an **offline, retrospective analysis tool**. It cannot be deployed in a live, real-time chatbot or streaming phone agent where future turns do not yet exist.
- **DialogueRNN (Causal Stream)**: DialogueRNN is strictly causal ($0 \dots t$), making it suitable for streaming applications, but it sacrifices the contextual benefit of hindsight that DialogueGCN exploits.

---

## 8. Non-Clinical Use and Ethical Safeguards

### 8.1 Explicit Prohibition of Diagnostic or Surveillance Applications
This software is strictly an academic research tool designed for computational linguistics and natural language processing analysis. Under no circumstances should this system be used for:
1. **Clinical or Psychiatric Diagnosis**: The system does not measure depression, anxiety, personality disorders, or mental health status.
2. **Employment Screening or Surveillance**: The system must not be used to score, rank, monitor, or make employment decisions regarding customer service agents or interview candidates.
3. **Automated Deception or Lie Detection**: The system does not detect veracity or deceit.
4. **Autonomous High-Stakes Decision Making**: Categorical predictions should never be used as sole inputs into legal, educational, or medical interventions.
