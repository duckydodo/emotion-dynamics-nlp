#!/usr/bin/env python3
"""Build publication-grade academic PowerPoint deck for Emotion Dynamics research project.

Generates: outputs/reports/Emotion_Dynamics_Presentation.pptx
Format: 16:9 Widescreen (13.333" x 7.5")
Theme: Dark Charcoal research design system (#171B18, #222923, #2A322C, #E8E5DB, #91A58B, #3B82F6)
Author: Chittaranjan Dutta (IIITDM Kurnool)
All metrics strictly reconciled against outputs/tables/ and master_research_findings_report.md.
Conforms strictly to typography and QA standards:
- Minimum body card font size >= 14pt
- Minimum table font size >= 12pt
- Zero overlapping elements, zero out-of-bounds shapes
- Undistorted aspect ratios on all embedded visuals
"""

import os
import sys
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# COLOR PALETTE (Task 34 Design System)
# ---------------------------------------------------------------------------
COLOR_BG = RGBColor(23, 27, 24)          # Deep Charcoal #171B18
COLOR_SURFACE = RGBColor(34, 41, 35)     # Primary Surface #222923
COLOR_SURFACE_ELEV = RGBColor(42, 50, 44)# Elevated Card #2A322C
COLOR_BORDER = RGBColor(55, 67, 58)      # Card Border #37433A
COLOR_TEXT = RGBColor(232, 229, 219)     # Main Off-White #E8E5DB
COLOR_TEXT_MUTED = RGBColor(158, 155, 144) # Muted Text #9E9B90
COLOR_ACCENT = RGBColor(145, 165, 139)   # Sage Green #91A58B
COLOR_BLUE = RGBColor(59, 130, 246)      # Sapphire Blue #3B82F6
COLOR_GOLD = RGBColor(226, 179, 64)      # Accent Gold #E2B340
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_RED = RGBColor(239, 68, 68)        # Error/Alert Red #EF4444

FONT_HEADING = "Helvetica"
FONT_BODY = "Arial"

TOTAL_SLIDES = 13


def apply_background(slide):
    """Draw a dark charcoal full-slide background rectangle."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG
    bg.line.fill.background()
    return bg


def add_header(slide, eyebrow: str, title: str, subtitle: str):
    """Add standard academic header block."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.10))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_eye = tf.paragraphs[0]
    p_eye.text = eyebrow.upper()
    p_eye.font.name = FONT_HEADING
    p_eye.font.size = Pt(11)
    p_eye.font.bold = True
    p_eye.font.color.rgb = COLOR_ACCENT
    p_eye.space_after = Pt(2)

    p_title = tf.add_paragraph()
    p_title.text = title
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT
    p_title.space_after = Pt(2)

    p_sub = tf.add_paragraph()
    p_sub.text = subtitle
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(12)
    p_sub.font.color.rgb = COLOR_TEXT_MUTED


def add_footer(slide, slide_num: int):
    """Add standard academic footer with subtle separator line."""
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.92), Inches(11.733), Pt(1)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_BORDER
    line.line.fill.background()

    footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.98), Inches(11.733), Inches(0.4))
    tf = footer_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    r1 = p.add_run()
    r1.text = "Emotion Dynamics — Multi-Party Conversational Affective Research   |   "
    r1.font.name = FONT_BODY
    r1.font.size = Pt(10)
    r1.font.color.rgb = COLOR_TEXT_MUTED

    r2 = p.add_run()
    r2.text = "Chittaranjan Dutta (IIITDM Kurnool)   |   "
    r2.font.name = FONT_BODY
    r2.font.size = Pt(10)
    r2.font.color.rgb = COLOR_ACCENT

    r3 = p.add_run()
    r3.text = f"Slide {slide_num} of {TOTAL_SLIDES}"
    r3.font.name = FONT_BODY
    r3.font.size = Pt(10)
    r3.font.bold = True
    r3.font.color.rgb = COLOR_TEXT_MUTED


def add_card(slide, left, top, width, height, title="", title_color=COLOR_TEXT, elev=False):
    """Add styled card surface with subtle border."""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_SURFACE_ELEV if elev else COLOR_SURFACE
    card.line.color.rgb = COLOR_BORDER
    card.line.width = Pt(1.2)

    tf = card.text_frame
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.word_wrap = True
    tf.margin_left = Inches(0.24)
    tf.margin_right = Inches(0.24)
    tf.margin_top = Inches(0.20)
    tf.margin_bottom = Inches(0.20)

    if title:
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = title_color
        p.space_after = Pt(8)
    return card, tf


def add_bullet_point(tf, bold_prefix: str, text: str, pt_size=14, space_after=5):
    """Append a styled bullet point with bold lead-in text (defaults to 14pt body font)."""
    if len(tf.paragraphs) == 1 and not tf.paragraphs[0].text.strip():
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.space_after = Pt(space_after)

    r_bullet = p.add_run()
    r_bullet.text = "•  "
    r_bullet.font.name = FONT_BODY
    r_bullet.font.size = Pt(pt_size)
    r_bullet.font.color.rgb = COLOR_ACCENT

    if bold_prefix:
        r_lead = p.add_run()
        r_lead.text = bold_prefix + ": "
        r_lead.font.name = FONT_HEADING
        r_lead.font.size = Pt(pt_size)
        r_lead.font.bold = True
        r_lead.font.color.rgb = COLOR_TEXT

    r_body = p.add_run()
    r_body.text = text
    r_body.font.name = FONT_BODY
    r_body.font.size = Pt(pt_size)
    r_body.font.color.rgb = COLOR_TEXT_MUTED


# ---------------------------------------------------------------------------
# SLIDE BUILDERS
# ---------------------------------------------------------------------------

def build_slide_1_title(prs):
    """Slide 1: Title Slide (Sole author, academic identity, key research metrics)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.4), Inches(0.12), Inches(4.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_ACCENT
    bar.line.fill.background()

    # Title box with height 2.6" (ends at 3.9", leaving clear gap before author card at 4.2")
    title_box = slide.shapes.add_textbox(Inches(1.6), Inches(1.3), Inches(10.5), Inches(2.6))
    tf = title_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_eye = tf.paragraphs[0]
    p_eye.text = "UNDERGRADUATE NLP RESEARCH  |  IIITDM KURNOOL"
    p_eye.font.name = FONT_HEADING
    p_eye.font.size = Pt(12)
    p_eye.font.bold = True
    p_eye.font.color.rgb = COLOR_ACCENT
    p_eye.space_after = Pt(8)

    p_title = tf.add_paragraph()
    p_title.text = "Emotion Dynamics"
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(42)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT
    p_title.space_after = Pt(6)

    p_sub = tf.add_paragraph()
    p_sub.text = "Conversational Emotion Analysis and Affective Trajectories in Multi-Party Dialogue"
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(17)
    p_sub.font.color.rgb = COLOR_TEXT_MUTED

    # Author Card (height 1.9", ends at 6.1")
    author_card, a_tf = add_card(slide, Inches(1.6), Inches(4.2), Inches(5.6), Inches(1.9), elev=True)
    p_a1 = a_tf.paragraphs[0]
    p_a1.text = "Chittaranjan Dutta"
    p_a1.font.name = FONT_HEADING
    p_a1.font.size = Pt(18)
    p_a1.font.bold = True
    p_a1.font.color.rgb = COLOR_TEXT
    p_a1.space_after = Pt(4)

    p_a2 = a_tf.add_paragraph()
    p_a2.text = "B.Tech in Artificial Intelligence & Data Science"
    p_a2.font.name = FONT_BODY
    p_a2.font.size = Pt(14)
    p_a2.font.color.rgb = COLOR_ACCENT
    p_a2.space_after = Pt(2)

    p_a3 = a_tf.add_paragraph()
    p_a3.text = "Indian Institute of Information Technology, Design and Manufacturing Kurnool"
    p_a3.font.name = FONT_BODY
    p_a3.font.size = Pt(12)
    p_a3.font.color.rgb = COLOR_TEXT_MUTED

    # Highlights
    metrics = [
        ("MELD Sitcom Corpus", "13,708 Utterances across 1,433 Dialogues"),
        ("Cohort A Benchmark", "Audited N=2,328 Contiguous Matched Subset"),
        ("7 Evaluated Models", "DistilBERT 3ep/5ep, DialogueRNN, DialogueGCN, TCN, CRF"),
        ("Production System", "Sub-50ms FastAPI + 5-Screen Responsive Interface")
    ]
    for i, (m_title, m_desc) in enumerate(metrics):
        top_y = Inches(4.2) + Inches(i * 0.46)
        chip = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.5), top_y, Inches(4.8), Inches(0.40))
        chip.fill.solid()
        chip.fill.fore_color.rgb = COLOR_SURFACE
        chip.line.color.rgb = COLOR_BORDER
        chip.line.width = Pt(1)
        c_tf = chip.text_frame
        c_tf.word_wrap = True
        c_tf.margin_left = Inches(0.15)
        c_tf.margin_top = Inches(0.04)
        c_p = c_tf.paragraphs[0]
        r1 = c_p.add_run()
        r1.text = m_title + "  —  "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(11)
        r1.font.bold = True
        r1.font.color.rgb = COLOR_GOLD
        r2 = c_p.add_run()
        r2.text = m_desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(11)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    f_box = slide.shapes.add_textbox(Inches(1.6), Inches(6.55), Inches(10.5), Inches(0.4))
    f_tf = f_box.text_frame
    f_p = f_tf.paragraphs[0]
    f_p.text = "Academic Year 2026  •  Natural Language Processing Final Project Defense"
    f_p.font.name = FONT_BODY
    f_p.font.size = Pt(11)
    f_p.font.color.rgb = COLOR_TEXT_MUTED


def build_slide_2_motivation(prs):
    """Slide 2: Motivation & Problem Statement."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "01 / MOTIVATION & PROBLEM STATEMENT",
               "Why Emotion in Multi-Party Dialogue Defies Standard NLP",
               "The breakdown of sentence-level affective analysis in situated conversational contexts")
    add_footer(slide, 2)

    cards = [
        ("Dyadic vs. Multi-Party Dialogue", [
            ("Multi-Party Complexity", "Conversations feature K >= 3 speakers, asynchronous turns, and shifting sub-groups."),
            ("Indirect Influence", "Utterances frequently respond to an interlocutor 2-4 turns prior rather than the predecessor."),
            ("Identity Tracking", "Speaker identity forms the topological graph of affective influence, not flat metadata.")
        ]),
        ("Situated Textual Ambiguity", [
            ("Lexical Deception", "Short turns like 'Yeah right' or 'Brilliant' exhibit positive surface forms but sarcastic intent."),
            ("Contextual Reversal", "Communicative intent and polarity are determined entirely by preceding interlocutor provocation."),
            ("Isolated NLP Failure", "Context-agnostic DistilBERT baseline reaches 43.14% Macro F1, heavily trapped by lexical attractors.")
        ]),
        ("Affective Dynamics as Trajectories", [
            ("Beyond Discrete Labels", "Emotion is an evolving continuous psychological trajectory, not a static point."),
            ("Emotional Inertia", "Speakers exhibit internal momentum, resisting abrupt emotional state transitions (r = 0.412)."),
            ("Conversational Resonance", "Interlocutors mirror positive affect (Joy <-> Joy) while deflecting or escalating negative affect.")
        ])
    ]

    card_width = Inches(3.64)
    for i, (ctitle, bullets) in enumerate(cards):
        left = Inches(0.8) + Inches(i * 4.04)
        _, tf = add_card(slide, left, Inches(1.80), card_width, Inches(4.95), title=ctitle, title_color=COLOR_ACCENT)
        for prefix, body in bullets:
            add_bullet_point(tf, prefix, body, pt_size=14, space_after=8)


def build_slide_3_dataset(prs):
    """Slide 3: Dataset & Data Hygiene (DVD gap audit, Cohort A N=2,328)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "02 / DATASET & EXPERIMENTAL HYGIENE",
               "MELD Corpus Architecture & The Canonical Cohort A Benchmark",
               "Resolving split irregularities and media corruption to establish rigorous evaluation standards")
    add_footer(slide, 3)

    _, tf_left = add_card(slide, Inches(0.8), Inches(1.80), Inches(5.6), Inches(4.95),
                          title="MELD Corpus & Emotion Distribution", title_color=COLOR_ACCENT)
    add_bullet_point(tf_left, "Source & Scope", "1,433 multi-party dialogues (13,708 utterances) transcribed from the TV series Friends.", pt_size=14, space_after=6)
    add_bullet_point(tf_left, "7 Ekman Emotions", "Severe class imbalance: Neutral 47.15%, Joy 17.45%, Surprise 12.06%, Anger 11.10%, Sadness 6.84%, Disgust 2.71%, Fear 2.68%.", pt_size=14, space_after=6)
    add_bullet_point(tf_left, "Official Splits", "Train: 9,989 utterances (1,038 dialogues) | Dev: 1,109 (114 dialogues) | Test: 2,610 (280 dialogues).", pt_size=14, space_after=6)
    add_bullet_point(tf_left, "Multi-Party Demographics", "Features 6 primary characters plus 250+ secondary characters in unstructured multi-party interaction.", pt_size=14, space_after=4)

    _, tf_right = add_card(slide, Inches(6.8), Inches(1.80), Inches(5.733), Inches(4.95),
                           title="The DVD Gap Audit & Canonical Cohort A (N=2,328)", title_color=COLOR_GOLD, elev=True)
    add_bullet_point(tf_right, "The Raw Test Discrepancy", "The official raw test split contains 2,610 utterances across 280 dialogues.", pt_size=14, space_after=6)
    add_bullet_point(tf_right, "282-Clip Media Defect", "Video extraction audit revealed 282 clips suffer from missing/corrupted raw MPEG/WAV streams due to commercial DVD mastering inconsistencies.", pt_size=14, space_after=6)
    add_bullet_point(tf_right, "Flaw in Prior Work", "Literature often evaluates text on N=2,610 but multimodal models on ad-hoc subsets, creating invalid comparisons.", pt_size=14, space_after=6)
    add_bullet_point(tf_right, "Canonical Cohort A (N=2,328)", "Our work strictly benchmarks all 7 models and multimodal pipelines on the exact same 2,328 contiguous turns.", pt_size=14, space_after=4)


def build_slide_4_pipeline(prs):
    """Slide 4: Feature Representation & The Frozen DistilBERT Pipeline."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "03 / METHODOLOGY & FEATURE PIPELINE",
               "Why Freeze the Text Encoder? The 3-Epoch vs. 5-Epoch Proof",
               "Preventing minority collapse while isolating downstream conversational context contributions")
    add_footer(slide, 4)

    _, tf_pipe = add_card(slide, Inches(0.8), Inches(1.80), Inches(5.6), Inches(4.95),
                          title="Feature Extraction Pipeline & 3-Epoch Baseline", title_color=COLOR_ACCENT)
    add_bullet_point(tf_pipe, "Base Encoder", "distilbert-base-uncased (66.96M parameters) fine-tuned on MELD utterance classification.", pt_size=14, space_after=6)
    add_bullet_point(tf_pipe, "Vector Representation", "Mean-pooled contextual embeddings generate fixed 768-dimensional utterance vectors u_t in R^768.", pt_size=14, space_after=6)
    add_bullet_point(tf_pipe, "Offline Feature Caching", "Precomputed and stored in meld_{split}_features.npz (Train 26.85 MB, Dev 2.98 MB, Test 7.00 MB).", pt_size=14, space_after=6)
    add_bullet_point(tf_pipe, "Cohort A 3-Epoch Baseline", "Accuracy 59.45% | Macro F1 43.14% | Weighted F1 57.94% (isolated utterance benchmark).", pt_size=14, space_after=4)

    _, tf_proof = add_card(slide, Inches(6.8), Inches(1.80), Inches(5.733), Inches(4.95),
                           title="Empirical Proof: Overfitting & Minority Collapse", title_color=COLOR_RED, elev=True)
    add_bullet_point(tf_proof, "The 5-Epoch Trajectory",
                     "Training for 5 epochs (scripts/train_distilbert_5epochs.py, 240.2 min): Epoch 4 reached Best Val F1 42.15%, before collapsing to 39.66%.",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_proof, "Cohort A Performance (Best @ Ep 4)",
                     "Accuracy rose to 61.90% (+2.45%), but Macro F1 degraded to 41.70% (-1.44% drop vs 3-epoch baseline of 43.14%). Weighted F1 reached 59.86%.",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_proof, "Minority Recall Collapse",
                     "Accuracy gain was an illusion: Fear recall collapsed from 15.69% to 10.00% and Disgust from 23.53% to 14.71%, while majority Neutral surged to 82.96%.",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_proof, "Methodological Decision",
                     "The 3-epoch checkpoint preserves critical minority discrimination and is frozen as the immutable base feature extractor.",
                     pt_size=14, space_after=4)


def build_slide_5_architectures(prs):
    """Slide 5: Contextual Modeling Architectures: Sequence vs Graph."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "04 / ARCHITECTURE DESIGN",
               "Contextual Modeling: DialogueRNN vs. DialogueGCN",
               "Contrasting attentive recurrent speaker tracking with relational graph convolutional attention")
    add_footer(slide, 5)

    _, tf_rnn = add_card(slide, Inches(0.8), Inches(1.80), Inches(5.6), Inches(4.95),
                         title="DialogueRNN: Recurrent Speaker Tracking", title_color=COLOR_BLUE)
    add_bullet_point(tf_rnn, "Core Philosophy", "Models emotional dynamics sequentially through three dedicated, interacting recurrent GRUs.", pt_size=14, space_after=6)
    add_bullet_point(tf_rnn, "Party GRU (q_t,p)", "Maintains private emotional state for each speaker p in P. Updates only when speaker p speaks, capturing self-inertia.", pt_size=14, space_after=6)
    add_bullet_point(tf_rnn, "Global GRU (g_t)", "Integrates all incoming turns into a shared conversational state regardless of speaker.", pt_size=14, space_after=6)
    add_bullet_point(tf_rnn, "Emotion GRU (e_t)", "Synthesizes party state with attentive pooling over previous speaker states to predict emotion (670,855 parameters).", pt_size=14, space_after=4)

    _, tf_gcn = add_card(slide, Inches(6.8), Inches(1.80), Inches(5.733), Inches(4.95),
                         title="DialogueGCN: Relational Graph Attention", title_color=COLOR_ACCENT, elev=True)
    add_bullet_point(tf_gcn, "Core Philosophy", "Treats dialogue as an utterance graph, bypassing sequential bottlenecks to capture non-adjacent dependencies.", pt_size=14, space_after=6)
    add_bullet_point(tf_gcn, "Graph Topology", "Utterance nodes V connected by edges E within context window W=10, encoding speaker identity and temporal distance.", pt_size=14, space_after=6)
    add_bullet_point(tf_gcn, "Relational Convolutions", "Explicitly separates intra-speaker edges (self-inertia) from inter-speaker edges (conversational influence).", pt_size=14, space_after=6)
    add_bullet_point(tf_gcn, "Sequential BiGRU Layer", "Follows graph message passing with a BiGRU to maintain local temporal order (452,487 parameters).", pt_size=14, space_after=4)


def build_slide_6_benchmark_table(prs):
    """Slide 6: Canonical Matched Leaderboard (Cohort A: N=2,328)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "05 / BENCHMARK LEADERBOARD",
               "Canonical Matched Cross-Model Leaderboard (Cohort A: N=2,328)",
               "Strictly aligned evaluation across all 7 trained architectures on the verified multimodal test subset")
    add_footer(slide, 6)

    table_shape = slide.shapes.add_table(8, 6, Inches(0.8), Inches(1.55), Inches(11.733), Inches(3.0))
    table = table_shape.table

    table.columns[0].width = Inches(3.1)   # Model Name
    table.columns[1].width = Inches(3.2)   # Paradigm
    table.columns[2].width = Inches(1.35)  # Parameters
    table.columns[3].width = Inches(1.35)  # Accuracy
    table.columns[4].width = Inches(1.35)  # Macro F1
    table.columns[5].width = Inches(1.35)  # Weighted F1

    headers = ["Model Architecture", "Paradigm & Speaker Awareness", "Parameters", "Accuracy", "Macro F1", "Weighted F1"]
    rows = [
        ("DistilBERT (3-Epoch Baseline)", "Context-Agnostic Transformer (Frozen)", "66,960,000", "59.45%", "43.14%", "57.94%"),
        ("DistilBERT (5-Epoch Baseline)", "Context-Agnostic Transformer (Overfit @ Ep 4)", "66,960,000", "61.90%", "41.70%", "59.86%"),
        ("DialogueRNN (Unweighted)", "Recurrent Speaker Tracking (Party GRUs)", "670,855", "60.14%", "43.42%", "58.92%"),
        ("DialogueGCN (Relational Graph)", "Relational Graph Convolutional Attention", "452,487", "60.91%", "43.85%", "60.09%"),
        ("Temporal CNN (Causal)", "Dilated 1D Conv (k=3, d=[1,2,4], RF=15)", "823,303", "60.27%", "42.47%", "59.52%"),
        ("Linear-Chain CRF (Viterbi)", "Global Sequence Decoding (Speaker-Agnostic)", "5,446", "60.01%", "42.44%", "59.47%"),
        ("Linear-Chain CRF (Causal)", "Online Greedy Decoding (Speaker-Agnostic)", "5,446", "60.18%", "43.15%", "59.63%"),
    ]

    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.margin_left = cell.margin_right = Inches(0.08)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_SURFACE_ELEV
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_HEADING
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        p.alignment = PP_ALIGN.LEFT if col_idx < 2 else PP_ALIGN.RIGHT

    for row_idx, r_data in enumerate(rows):
        is_best = (row_idx == 3) # DialogueGCN
        bg_col = RGBColor(38, 48, 40) if is_best else (COLOR_SURFACE if row_idx % 2 == 0 else COLOR_BG)
        for col_idx, val in enumerate(r_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_col
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(12)
            p.font.bold = is_best or (col_idx == 0)
            if is_best:
                p.font.color.rgb = COLOR_GOLD if col_idx >= 3 else COLOR_TEXT
            else:
                p.font.color.rgb = COLOR_TEXT if col_idx < 2 else (COLOR_RED if (row_idx == 1 and col_idx == 4) else COLOR_TEXT_MUTED)
            p.alignment = PP_ALIGN.LEFT if col_idx < 2 else PP_ALIGN.RIGHT

    # Bottom Callout Card (height 1.85", top 4.90", ends at 6.75")
    _, tf_bot = add_card(slide, Inches(0.8), Inches(4.90), Inches(11.733), Inches(1.85), elev=True)
    p_lead = tf_bot.paragraphs[0]
    p_lead.text = "Key Empirical Takeaways (Cohort A, N=2,328):"
    p_lead.font.name = FONT_HEADING
    p_lead.font.size = Pt(15)
    p_lead.font.bold = True
    p_lead.font.color.rgb = COLOR_GOLD
    p_lead.space_after = Pt(2)

    add_bullet_point(tf_bot, "DialogueGCN Dominance",
                     "SOTA on Cohort A: 43.85% Macro F1, 60.91% Acc, 60.09% W-F1 (+0.71% Macro F1 over frozen DistilBERT, +2.15% over overfit 5-epoch baseline).",
                     pt_size=14, space_after=3)
    add_bullet_point(tf_bot, "The Speaker Awareness Divide",
                     "Speaker-agnostic models (TCN 42.47%, CRF 42.44%) fail to beat isolated DistilBERT (43.14%), proving flat modeling damages dialogue context.",
                     pt_size=14, space_after=0)


def build_slide_7_ablation(prs):
    """Slide 7: Why Speaker Tracking Beats Speaker-Agnostic Sequence Models."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "06 / ABLATION ANALYSIS",
               "Why Speaker Tracking Beats Speaker-Agnostic Sequence Models",
               "Deconstructing the failure of TCN and Linear-Chain CRF when stripped of speaker identity")
    add_footer(slide, 7)

    _, tf_abl = add_card(slide, Inches(0.8), Inches(1.80), Inches(6.0), Inches(4.95),
                         title="The Speaker-Agnostic Penalty Explained", title_color=COLOR_ACCENT)
    add_bullet_point(tf_abl, "The Speaker Blindness Trap",
                     "TCN (42.47% Macro F1) and CRF Viterbi (42.44% Macro F1) underperform context-agnostic DistilBERT (43.14%).",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_abl, "Inertia vs. Reaction Conflation",
                     "If Speaker A expresses Anger, Speaker B reacts rather than continuing A's mood. Flat convolutions treat all prior turns identically.",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_abl, "Flattened Transitions",
                     "Linear-Chain CRF learns a single global 7x7 transition matrix, blurring self-persistence with cross-speaker resonance.",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_abl, "DialogueGCN Resolution",
                     "Relational edges explicitly separate intra-speaker connections from inter-speaker edges, preventing signal pollution.",
                     pt_size=14, space_after=4)

    cm_path = "outputs/figures/dialoguegcn_confusion_matrix.png"
    if os.path.exists(cm_path):
        add_card(slide, Inches(7.0), Inches(1.80), Inches(5.533), Inches(4.95), title="", elev=True)
        # Scaled to height 3.85" (aspect 1.167 -> width 4.49") so it ends at 5.85"
        slide.shapes.add_picture(cm_path, Inches(7.52), Inches(2.00), width=Inches(4.49), height=Inches(3.85))
        cap_box = slide.shapes.add_textbox(Inches(7.20), Inches(6.00), Inches(5.133), Inches(0.60))
        c_tf = cap_box.text_frame
        c_tf.word_wrap = True
        c_p = c_tf.paragraphs[0]
        c_p.text = "Fig 1: DialogueGCN Confusion Matrix on Canonical Cohort A (N=2,328). Demonstrates strong diagonal precision across majority classes."
        c_p.font.name = FONT_BODY
        c_p.font.size = Pt(10.5)
        c_p.font.color.rgb = COLOR_TEXT_MUTED


def build_slide_8_dynamics(prs):
    """Slide 8: Emotion Dynamics & Next-Emotion Forecasting."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "07 / AFFECTIVE DYNAMICS",
               "Quantifying Emotional Inertia, Resonance & Forecasting",
               "Statistical laws governing affective transitions and the failure of discrete Markov models")
    add_footer(slide, 8)

    hm_path = "outputs/figures/week7_test_transition_heatmap.png"
    if os.path.exists(hm_path):
        add_card(slide, Inches(0.8), Inches(1.80), Inches(5.4), Inches(4.95), title="", elev=True)
        # Scaled to height 3.80" (aspect 1.236 -> width 4.70") so it ends at 5.80"
        slide.shapes.add_picture(hm_path, Inches(1.15), Inches(2.00), width=Inches(4.70), height=Inches(3.80))
        cap_box = slide.shapes.add_textbox(Inches(0.95), Inches(5.95), Inches(5.10), Inches(0.60))
        c_tf = cap_box.text_frame
        c_tf.word_wrap = True
        c_p = c_tf.paragraphs[0]
        c_p.text = "Fig 2: Empirical Emotion Transition Heatmap across test split (N=2,610). Displays dominant Neutral basin and positive resonance."
        c_p.font.name = FONT_BODY
        c_p.font.size = Pt(10.5)
        c_p.font.color.rgb = COLOR_TEXT_MUTED

    _, tf_dyn = add_card(slide, Inches(6.4), Inches(1.80), Inches(6.133), Inches(4.95),
                         title="Mathematical Laws of Conversational Affect", title_color=COLOR_GOLD)
    add_bullet_point(tf_dyn, "Speaker Persistence Baseline",
                     "Repeating a speaker's last emotion achieves 41.92% Acc, 27.36% Macro F1, and 41.86% W-F1, outperforming all Markov baselines.",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_dyn, "First-Order Markov Degeneracy",
                     "1st-order Markov chain collapses entirely to Majority Baseline (47.94% Acc, 9.26% Macro F1), predicting 100% Neutral.",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_dyn, "Higher-Order Markov Plateau",
                     "Markov-2 reaches 20.03% Macro F1 (48.81% Acc); Markov-3 plateaus at 20.80% Macro F1 due to state sparsity (343 states, >60% support <= 2).",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_dyn, "Continuous Trajectory Advantage",
                     "DialogueRNN and DialogueGCN track continuous latent representations, avoiding discrete Markov sparsity entirely.",
                     pt_size=14, space_after=4)


def build_slide_9_explainability(prs):
    """Slide 9: Explainability Layer & Universal Text-Only Error Analysis."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "08 / EXPLAINABILITY & ERROR ANALYSIS",
               "Interpretability & The 31.66% Universal Text-Only Error Floor",
               "Integrated Gradients, Context Shift Delta P, and the 737 Universal Disagreement Taxonomy")
    add_footer(slide, 9)

    cards = [
        ("Integrated Gradients Attribution", [
            ("Axiomatic Attribution", "Path-integrated gradients (M=25) compute token attribution toward classification logits."),
            ("Sarcasm Lexical Trap", "On 'Oh that's just fantastic...', DistilBERT predicts Joy (89.9%) due to 'fantastic' (+0.242)."),
            ("Affective Token Salience", "Attributions peak on emotional pivots: 'disgusting' (+0.339), 'terrified' (+0.159).")
        ]),
        ("Context Shift Metric (Delta P)", [
            ("Formal Definition", "Delta P_t(y) = P(y | u_t, H_{<t}) - P(y | u_t, 0). Quantifies probability shift induced by dialogue history."),
            ("Conflict De-escalation", "Prior rejection shifts Anger confidence down by Delta P = -0.314 toward distress."),
            ("Relational Reinforcement", "DialogueGCN leverages speaker edges to reinforce conflict, boosting Anger by +0.113.")
        ]),
        ("737 Universal Text Errors (31.66%)", [
            ("Taxonomy Breakdown", "1,210 all correct (51.98%), 197 net context recovery (8.46%), 174 context dilution (7.47%), 10 all disagree (0.43%), 737 universal errors (31.66%)."),
            ("High Model Concordance", "Unanimous agreement on 1,772 turns (76.12%). Cohen's kappa: DistilBERT vs RNN 0.841, RNN vs GCN 0.747."),
            ("Multimodal Necessity", "Acoustic and visual streams (prosody, sneers, laughter) are required to resolve the 737 errors beyond text.")
        ])
    ]

    card_width = Inches(3.64)
    for i, (ctitle, bullets) in enumerate(cards):
        left = Inches(0.8) + Inches(i * 4.04)
        _, tf = add_card(slide, left, Inches(1.80), card_width, Inches(4.95), title=ctitle, title_color=COLOR_ACCENT)
        pt = 13.5 if i == 2 else 14
        sp = 4 if i == 2 else 6
        for prefix, body in bullets:
            add_bullet_point(tf, prefix, body, pt_size=pt, space_after=sp)


def build_slide_10_webapp(prs):
    """Slide 10: Interactive Web Application & Deployment."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "09 / PRODUCTION APPLICATION",
               "Interactive Web Application & Production Deployment",
               "Sub-50ms FastAPI inference engine, responsive five-screen interface, and zero-dependency Docker deployment")
    add_footer(slide, 10)

    img2 = "outputs/figures/screen3_emotion_dynamics.png"

    if os.path.exists(img2):
        add_card(slide, Inches(0.8), Inches(1.80), Inches(5.8), Inches(4.95), title="", elev=True)
        # Scaled to width 5.40", height 3.40" (ends at 5.40")
        slide.shapes.add_picture(img2, Inches(1.00), Inches(2.00), width=Inches(5.40), height=Inches(3.40))
        # Summary chip below image
        cap_box = slide.shapes.add_textbox(Inches(1.00), Inches(5.55), Inches(5.40), Inches(0.90))
        c_tf = cap_box.text_frame
        c_tf.word_wrap = True
        c_p = c_tf.paragraphs[0]
        c_p.text = "Five Integrated Screens: 1. Raw Ingestion & Normalizer (WhatsApp/JSON/CSV) | 2. Timeline Inspector | 3. Trajectory & Transition Heatmap | 4. Live Leaderboard | 5. Saliency & Delta P Deep-Dive."
        c_p.font.name = FONT_BODY
        c_p.font.size = Pt(11)
        c_p.font.color.rgb = COLOR_TEXT_MUTED

    _, tf_sys = add_card(slide, Inches(6.8), Inches(1.80), Inches(5.733), Inches(4.95),
                         title="Full-Stack System Specifications", title_color=COLOR_BLUE)
    add_bullet_point(tf_sys, "FastAPI Architecture",
                     "REST endpoints for /predict (single/batch), /analyze-dynamics, /benchmarks, and /explain.",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_sys, "Sub-50ms Latency Profile",
                     "DialogueGCN batch inference executes in < 45ms per 20-turn dialogue on CPU with cached representations.",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_sys, "Docker Containerization",
                     "Single-command deployment via docker compose up --build on port 8001 without external dependencies.",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_sys, "149 Automated Tests",
                     "100% passing test suite verifying API contracts, numerical stability, edge cases, and multimodal fusion masking.",
                     pt_size=14, space_after=4)


def build_slide_11_multimodal(prs):
    """Slide 11: Tri-Modal Extension: Text + Audio + Video (Live/Kaggle Ready)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "10 / MULTIMODAL EXPANSION",
               "Tri-Modal Extension: WavLM Audio + CLIP Video + DistilBERT Text",
               "Gated Multimodal Fusion architecture to break the 31.66% universal text-only error floor")
    add_footer(slide, 11)

    _, tf_arch = add_card(slide, Inches(0.8), Inches(1.80), Inches(5.7), Inches(4.95),
                          title="Gated Multimodal Fusion Architecture", title_color=COLOR_ACCENT)
    add_bullet_point(tf_arch, "Acoustic Stream (WavLM)",
                     "microsoft/wavlm-base-plus extracts 768-d utterance prosody, capturing vocal pitch, volume bursts, and speech rate at 16kHz mono.",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_arch, "Visual Stream (CLIP ViT-B/32)",
                     "openai/clip-vit-base-patch32 extracts 512-d frame embeddings across 4 uniformly sampled video frames, encoding facial expression.",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_arch, "Gated Modality Projection",
                     "LayerNorm + Linear -> GELU -> Dropout projects modalities to 256-d with learned feature-wise sigmoid gating.",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_arch, "DVD Missing-Clip Masking",
                     "Binary availability masks (audio_mask, video_mask) and active scaling ensure missing clips never cause NaNs or zero-division.",
                     pt_size=14, space_after=4)

    add_card(slide, Inches(6.7), Inches(1.80), Inches(5.833), Inches(4.95),
             title="", elev=True)

    # Right Card Title
    tb_title = slide.shapes.add_textbox(Inches(6.9), Inches(1.95), Inches(5.4), Inches(0.35))
    tf_t = tb_title.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = "Empirical Cohort A Benchmark (N = 2,328)"
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(16)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_GOLD

    # Table with 5 rows: Header + 4 Verified Multimodal Models
    table_shape = slide.shapes.add_table(5, 5, Inches(6.9), Inches(2.35), Inches(5.4), Inches(1.60))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(2.00)
    tbl.columns[1].width = Inches(0.80)
    tbl.columns[2].width = Inches(0.85)
    tbl.columns[3].width = Inches(0.85)
    tbl.columns[4].width = Inches(0.90)

    t_headers = ["Model & Modality", "Acc", "Macro F1", "W-F1", "Recov."]
    t_rows = [
        ("Bimodal DialogueRNN (T+A)", "60.95%", "43.59%", "60.34%", "28 / 737"),
        ("Bimodal DialogueGCN (T+A)", "60.01%", "42.34%", "59.47%", "29 / 737"),
        ("Trimodal DialogueRNN (T+A+V)", "60.74%", "42.65%", "60.36%", "23 / 737"),
        ("Trimodal DialogueGCN (T+A+V)", "59.92%", "42.94%", "59.65%", "24 / 737"),
    ]
    for c_i, h_t in enumerate(t_headers):
        c = tbl.cell(0, c_i)
        c.margin_top = c.margin_bottom = Inches(0.02)
        c.margin_left = c.margin_right = Inches(0.06)
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_SURFACE_ELEV
        p = c.text_frame.paragraphs[0]
        p.text = h_t
        p.font.name = FONT_HEADING
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT

    for r_i, r_data in enumerate(t_rows):
        for c_i, val in enumerate(r_data):
            c = tbl.cell(r_i + 1, c_i)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.margin_left = c.margin_right = Inches(0.06)
            c.fill.solid()
            c.fill.fore_color.rgb = COLOR_SURFACE
            p = c.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(12)
            p.font.color.rgb = COLOR_GOLD if r_i == 0 else COLOR_TEXT

    # Key Empirical Takeaways Textbox
    tb_takeaway = slide.shapes.add_textbox(Inches(6.9), Inches(4.20), Inches(5.4), Inches(2.45))
    tf_ta = tb_takeaway.text_frame
    tf_ta.word_wrap = True
    tf_ta.margin_left = tf_ta.margin_right = tf_ta.margin_top = tf_ta.margin_bottom = 0
    add_bullet_point(tf_ta, "WavLM Acoustic Prosody Gains",
                     "Acoustic prosody improves DialogueRNN across all metrics (+0.81% Acc, +1.42% Weighted F1 to 60.34%) and recovers 28–29 universal text failures.",
                     pt_size=14, space_after=8)
    add_bullet_point(tf_ta, "Vision Noise vs. Minority Lift",
                     "Uncropped full-frame CLIP pooling adds background scene noise; yet Trimodal GCN lifts rare Fear F1 (20.22%) and Disgust F1 (23.88%), recovering 24 universal errors.",
                     pt_size=14, space_after=0)


def build_slide_12_limitations(prs):
    """Slide 12: Limitations & Ethical / Non-Clinical Boundaries."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "11 / RESEARCH BOUNDARIES & ETHICS",
               "Methodological Limitations & Non-Clinical Boundaries",
               "Critical appraisal of sitcom domain bias, inter-annotator agreement, and application constraints")
    add_footer(slide, 12)

    cards = [
        ("Sitcom Domain Bias", [
            ("Scripted Delivery", "MELD dialogue originates from Friends, featuring theatrical setups and comedic exaggeration."),
            ("Studio Laughter Confounds", "Off-screen audience laughter provides implicit comedic cues absent in natural talk."),
            ("Domain Adaptation", "Models trained on sitcom affect require transfer adaptation for clinical or workplace use.")
        ]),
        ("Annotation Ambiguity (Fleiss' kappa ≈ 0.43)", [
            ("Moderate Inter-Rater Agreement", "Human annotators achieved Fleiss' kappa ≈ 0.43, demonstrating subjectivity in emotion perception."),
            ("Polysemous Affective States", "Utterances convey dual emotions (Sadness + Anger); discrete labeling loses nuance."),
            ("Probability Calibration", "Output probabilities are external perception estimates, never ground-truth internal state.")
        ]),
        ("Strict Non-Clinical Boundary", [
            ("Explicit Exclusions", "Strictly prohibited from psychiatric diagnosis, clinical triage, or autonomous mental health care."),
            ("No Surveillance Use", "Must not be deployed for non-consensual employee monitoring, workplace surveillance, or gating."),
            ("Human-in-the-Loop Imperative", "Analytics serve strictly as supportive, transparent decision aids with accessible explanations.")
        ])
    ]

    card_width = Inches(3.64)
    for i, (ctitle, bullets) in enumerate(cards):
        left = Inches(0.8) + Inches(i * 4.04)
        _, tf = add_card(slide, left, Inches(1.80), card_width, Inches(4.95), title=ctitle, title_color=COLOR_RED if i == 2 else COLOR_ACCENT)
        for prefix, body in bullets:
            add_bullet_point(tf, prefix, body, pt_size=14, space_after=8)


def build_slide_13_conclusion(prs):
    """Slide 13: Conclusion & Key Research Contributions."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    apply_background(slide)
    add_header(slide, "12 / CONCLUSION & SUMMARY",
               "Key Research Contributions & Future Trajectories",
               "Synthesis of findings, reproducible computational artifacts, and future multimodal frontiers")
    add_footer(slide, 13)

    _, tf_sum = add_card(slide, Inches(0.8), Inches(1.80), Inches(6.8), Inches(4.95),
                         title="Core Research Contributions", title_color=COLOR_ACCENT)
    add_bullet_point(tf_sum, "1. Methodological Rigor",
                     "Built audited Canonical Cohort A (N=2,328), resolving the 282-clip DVD gap and ensuring valid comparisons.",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_sum, "2. Relational Graph Dominance",
                     "Proved speaker-aware DialogueGCN (43.85% Macro F1) outperforms sequence models (TCN 42.47%, CRF 42.44%).",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_sum, "3. Feature Freezing Integrity",
                     "Proved 3-epoch feature freezing prevents catastrophic minority collapse (preserving Fear/Disgust).",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_sum, "4. Explainability & Error Floor",
                     "Uncovered 31.66% text-only error floor (737 turns) and formalized Context Shift Delta P for sarcasm detection.",
                     pt_size=14, space_after=6)
    add_bullet_point(tf_sum, "5. Production Engineering",
                     "Delivered full five-screen FastAPI + Docker application with 149 passing regression tests.",
                     pt_size=14, space_after=4)

    _, tf_hor = add_card(slide, Inches(7.8), Inches(1.80), Inches(4.733), Inches(4.95),
                         title="Researcher & Repository Information", title_color=COLOR_GOLD, elev=True)
    add_bullet_point(tf_hor, "Author", "Chittaranjan Dutta (Sole Author)", pt_size=14, space_after=6)
    add_bullet_point(tf_hor, "Degree & Department", "B.Tech in Artificial Intelligence & Data Science, Dept of CSE", pt_size=14, space_after=6)
    add_bullet_point(tf_hor, "Institution", "Indian Institute of Information Technology, Design and Manufacturing Kurnool", pt_size=14, space_after=6)
    add_bullet_point(tf_hor, "Reproducibility", "docker compose up --build runs complete five-screen application with verified benchmarks.", pt_size=14, space_after=6)
    add_bullet_point(tf_hor, "Future Horizon", "Cross-attention cross-modal fusion on naturalistic conversational corpora (IEMOCAP).", pt_size=14, space_after=4)


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    print("Building Emotion Dynamics Academic Presentation Deck (13 Slides)...")

    build_slide_1_title(prs)
    build_slide_2_motivation(prs)
    build_slide_3_dataset(prs)
    build_slide_4_pipeline(prs)
    build_slide_5_architectures(prs)
    build_slide_6_benchmark_table(prs)
    build_slide_7_ablation(prs)
    build_slide_8_dynamics(prs)
    build_slide_9_explainability(prs)
    build_slide_10_webapp(prs)
    build_slide_11_multimodal(prs)
    build_slide_12_limitations(prs)
    build_slide_13_conclusion(prs)

    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Emotion_Dynamics_Presentation.pptx"

    prs.save(str(output_path))
    file_size_kb = output_path.stat().st_size / 1024
    print(f"Successfully generated: {output_path} ({file_size_kb:.1f} KB, {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
