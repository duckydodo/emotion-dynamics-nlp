# Multimodal Emotion Dynamics: Future Vacation Research Extension Roadmap

**Author:** NLP Research Team  
**Date:** November 2026  
**Document Type:** Vacation Research Specification & Architectural Roadmap  
**Target Modalities:** Text + Audio Prosody + Conversational Topology  
**Primary Benchmarks:** MELD Multimodal Splits, IEMOCAP (Four-Class Benchmark)  
**Status:** Approved Research Extension Specification

---

## 1. Executive Motivation & The Textual Ceiling

In Phase 3 of our empirical investigation, the cross-model disagreement analysis on Canonical Cohort A ($N=2,328$) uncovered a fundamental constraint:
$$\text{Universal Textual Error Rate} = 31.66\% \quad (737 \text{ turns})$$

Where all three text-based architectures (**DistilBERT**, **DialogueRNN**, and **DialogueGCN**) simultaneously fail to identify ground truth affective labels. Our qualitative and gradient-based attribution audits revealed two primary mechanisms driving this ceiling:

1. **Sarcastic Lexical Inversion**: Surface transformer representations are susceptible to positive lexical attraction. In statements such as *"Oh that's just fantastic, another flat tire in the middle of a thunderstorm,"* the positive valence of `"fantastic"` ($+0.242$ Integrated Gradients salience) forces a high-confidence misclassification as **Joy** ($89.9\%$). The true negative affect (*Anger* / *Disgust*) is encoded exclusively in acoustic pitch variance, vocal elongation, and sarcastic prosodic contours.
2. **Short Ambiguous Backchannels**: Over 18% of MELD utterances consist of three words or fewer (*"Yeah"*, *"Okay"*, *"Oh"*, *"What?"*). In text, these turns are nearly indistinguishable from *Neutral*; in speech, pitch inflection ($F_0$ contour) and intensity reliably distinguish joyful agreement from neutral acknowledgment or stunned disbelief.

This roadmap details the engineering, modeling, and evaluation protocol for extending the Emotion Dynamics platform into a unified **multimodal conversational architecture** during the upcoming vacation research sprint.

---

## 2. Acoustic Feature Extraction Pipeline

```
                               ACOUSTIC PROSODY PIPELINE
 
   Raw Dialogue Audio Clip (.wav / 16kHz)
               │
       ┌───────┴───────────────────────────────────────┐
       ▼                                               ▼
  [Acoustic Expert Features]              [Self-Supervised Embeddings]
  openSMILE (eGeMAPS Set)                 WavLM Large / wav2vec 2.0
  88 low-level descriptors                Temporal Transformer Layers
  (Pitch F0, Jitter, Shimmer, HNR)        (Contextual Audio Representations)
       │                                               │
       ▼                                               ▼
  Normalized Prosodic Vector (88-dim)     Mean-Pooled Speech Vector (1024-dim)
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
            Acoustic Utterance Representation a_t (d_a)
```

### 2.1 Handcrafted Acoustic Descriptors: openSMILE & eGeMAPS
We will utilize the **extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPS)** (Eyben et al., 2016), an 88-dimensional standardized acoustic feature representation extracted via `openSMILE`:
* **Frequency Parameters**: Fundamental frequency ($F_0$), pitch contour slope, formant frequencies ($F_1, F_2, F_3$), and bandwidths.
* **Energy Parameters**: Sound energy, loudness, spectral harmonic-to-noise ratio (HNR).
* **Spectral Parameters**: Alpha ratio, Hammarberg index, spectral flux, and MFCCs 1–4.
* **Temporal Dynamics**: Jitter (pitch perturbation), shimmer (amplitude perturbation), and unvoiced-to-voiced duration ratios.

### 2.2 Deep Self-Supervised Speech Encoders: WavLM
While eGeMAPS captures interpretable vocal physiology, modern self-supervised speech transformers capture subtle paralinguistic nuance:
* **Model Backbone**: `WavLM-Large` (Chen et al., 2022; 316M parameters, trained on 94,000 hours of speech).
* **Extraction Protocol**: Raw 16 kHz audio is passed through WavLM. The final hidden states are mean-pooled across time frames to yield a dense 1024-dimensional acoustic embedding $\mathbf{a}_t$.

### 2.3 Timestamp Synchronization via Whisper
To align subword lexical tokens with acoustic temporal frames:
* **ASR Engine**: `whisper-large-v3` running locally in Docker.
* **Word-Level Timestamps**: Uses cross-attention dynamic time warping (DTW) to obtain start and end millisecond timestamps for each word $w_{t, k}$. This enables fine-grained cross-modal attention between specific words (e.g., *"fantastic"*) and their exact localized pitch spikes.

---

## 3. Multimodal Conversational Architecture

```
                       MULTIMODAL CONVERSATIONAL FUSION
 
   Text Utterance u_t                    Audio Waveform a_t
   (DistilBERT [CLS], d=768)             (WavLM Pooled, d=1024)
          │                                     │
          └──────────────┬──────────────────────┘
                         ▼
              Cross-Modal Attention Block
              Q_text = u_t W_Q , K_audio = a_t W_K
              (Text-Guided Acoustic Gating)
                         │
                         ▼
             Fused Multimodal Turn Representation m_t
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  DialogueRNN Fusion              DialogueGCN Fusion
  Global GRU Input: [m_t || q_t]  Relational Edge Weights:
  Party GRU Input:  [m_t || g_t]  alpha_ij = Sim(m_i, m_j)
         │                               │
         └───────────────┬───────────────┘
                         ▼
             Final Multimodal Prediction
             (7-Class Emotion Probability)
```

### 3.1 Cross-Modal Attention and Gating
Rather than simple concatenation $[\mathbf{e}_t \,\|\, \mathbf{a}_t]$, which allows dominant text features to overpower audio, we employ a **bidirectional cross-modal transformer layer**:

$$\mathbf{H}_{T \to A} = \text{Softmax}\left(\frac{(\mathbf{E} \mathbf{W}_Q^{(T)})(\mathbf{A} \mathbf{W}_K^{(A)})^T}{\sqrt{d_k}}\right) (\mathbf{A} \mathbf{W}_V^{(A)})$$

$$\mathbf{H}_{A \to T} = \text{Softmax}\left(\frac{(\mathbf{A} \mathbf{W}_Q^{(A)})(\mathbf{E} \mathbf{W}_K^{(T)})^T}{\sqrt{d_k}}\right) (\mathbf{E} \mathbf{W}_V^{(T)})$$

A learnable highway gating mechanism balances textual and acoustic channels dynamically:
$$\mathbf{z}_t = \sigma(\mathbf{W}_g [\mathbf{H}_{T \to A} \,\|\, \mathbf{H}_{A \to T}])$$
$$\mathbf{m}_t = \mathbf{z}_t \odot \mathbf{H}_{T \to A} + (1 - \mathbf{z}_t) \odot \mathbf{H}_{A \to T}$$

### 3.2 Integration into DialogueRNN and DialogueGCN
1. **Multimodal DialogueRNN**:
   - The fused vector $\mathbf{m}_t$ replaces the unimodal text embedding $\mathbf{x}_t$ in the Global GRU ($g_t$) and Party GRU ($q_t$).
   - When an interlocutor speaks with angry prosody, their acoustic intensity directly updates their party state $q_t^{(s)}$, preserving cross-speaker tension even if subsequent utterances use polite text.
2. **Multimodal DialogueGCN**:
   - Relational edge weights $\alpha_{ij}$ are recomputed using multimodal similarity:
     $$\alpha_{ij} = \text{Softmax}\left(\frac{(\mathbf{m}_i \mathbf{W}_Q)(\mathbf{m}_j \mathbf{W}_K)^T}{\sqrt{d}}\right)$$
   - This prevents dialogue nodes with polite words but aggressive audio from having their edge attention suppressed.

---

## 4. Application & Telemetry Extension: The De-escalation Index

In conversational analytics (e.g., customer support, dispute mediation), detecting emotions after the fact is insufficient. The vacation research plan includes implementing real-time conversational affect telemetry:

```
+-------------------------------------------------------------------------------+
|                       AFFECT TELEMETRY METRICS SUITE                          |
+--------------------------+----------------------------------------------------+
| Escalation Velocity      | Rate of positive-to-negative valence transition    |
| (v_esc)                  | per unit time: d(Valence) / dt                     |
+--------------------------+----------------------------------------------------+
| De-escalation Index      | Probability that an agent turn successfully drives |
| (D_idx)                  | a customer from negative affect back to neutral.   |
+--------------------------+----------------------------------------------------+
| Interlocutor Entrainment | Spectral convergence between speakers' fundamental |
| (Prosodic Mirroring)     | frequencies F0 across consecutive turns.           |
+--------------------------+----------------------------------------------------+
```

### Mathematical Definition of the De-escalation Index ($D_\text{idx}$):
Given customer turns $C_{t-1}$ and $C_{t+1}$ flanking agent intervention $A_t$:

$$D_\text{idx}(A) = P\left(\text{Valence}(C_{t+1}) \ge 0 \;\middle|\; \text{Valence}(C_{t-1}) < 0, \text{Intervention}(A_t)\right)$$

This metric directly evaluates agent conversational efficacy grounded in empirical conversational transitions rather than subjective satisfaction surveys.

---

## 5. Vacation Research Implementation Schedule

```
  Week 1: Audio Preprocessing & ASR Synchronization
  ├── Download official MELD audio clips (.wav) and verify checksums
  ├── Execute openSMILE eGeMAPS feature extraction across all splits
  └── Run Whisper large-v3 for forced word-level millisecond alignment
 
  Week 2: Self-Supervised Audio Representations
  ├── Benchmark WavLM-Large feature extraction in Docker
  └── Save cached multimodal tensors (meld_train_multimodal.npz)
 
  Week 3: Cross-Modal Architecture & Training
  ├── Implement bidirectional cross-modal attention fusion layer
  ├── Train Multimodal DialogueRNN (unweighted and class-weighted)
  └── Train Multimodal DialogueGCN (relational multimodal graph)
 
  Week 4: Benchmarking, Sarcasm Auditing & Publication Write-up
  ├── Evaluate on Canonical Cohort A (N=2,328) for matched comparison
  ├── Verify error reduction on the 737 universal error turns
  └── Complete academic paper manuscript for conference submission
```

---

## 6. Ethical Guardrails & Clinical Non-Certainty

As with our text platform, all multimodal extensions will strictly adhere to ethical constraints:
1. **Perceived Expressed Emotion Only**: The application will prominently state that acoustic-lexical models classify *perceived expressed communication cues*, not internal physiological emotions or authentic mental health states.
2. **Zero Biometric Profiling**: No voiceprint biometric identification, gender profiling, or persistent user tracking.
3. **No Pop-Psychology Extensions**: Strict prohibition against romantic compatibility percentages, emotional lie detection, or automated relationship counseling.
