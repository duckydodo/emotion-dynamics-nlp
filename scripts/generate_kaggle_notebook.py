#!/usr/bin/env python3
"""Build and update notebooks/30_kaggle_multimodal_meld.ipynb with audited schemas and hardened discovery."""

import json
import ast
from pathlib import Path


def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (CUDA)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # --------------------------------------------------------------------------
    # Cell 0: Title & Header
    # --------------------------------------------------------------------------
    cell_0_md = (
        "# Emotion Dynamics: Multimodal Conversational Emotion Recognition (MELD)\n"
        "### Tri-Modal Fusion: WavLM Audio + CLIP-ViT Video + Frozen DistilBERT Text\n"
        "**Author:** Chittaranjan Dutta (IIITDM Kurnool)  \n"
        "**Target Platform:** Kaggle Notebook (GPU T4 x 1 or T4 x 2)  \n"
        "**Canonical Benchmark:** Cohort A ($N = 2,328$ audited multimodal turns)  \n"
        "\n"
        "---\n"
        "### Workflow Overview\n"
        "1. **Environment Setup & Discovery:** Scans `/kaggle/input/` for MELD CSVs, `.mp4` video clips, and pre-extracted features.\n"
        "2. **Acoustic Extraction (WavLM Base+):** 768-d utterance prosody vectors via PyAV mono resampling at 16,000 Hz.\n"
        "3. **Visual Extraction (CLIP ViT-B/32):** 512-d frame embeddings across 4 uniform frames per utterance.\n"
        "4. **Feature Alignment & Cohort A Audit:** Aligns Text, Audio, Video representations by `(Dialogue_ID, Utterance_ID)`.\n"
        "5. **Gated Multimodal Fusion & Contextual Training:** Trains Bimodal and Trimodal variants of `DialogueRNN` and `DialogueGCN`.\n"
        "6. **Universal Error Recovery & Packaging:** Audits recovery across the 737 universal text-only error turns, plots confusion matrices, and packages `/kaggle/working/multimodal_results_bundle.zip`.\n"
    )
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [cell_0_md]})

    # --------------------------------------------------------------------------
    # Cell 1: Markdown Setup
    # --------------------------------------------------------------------------
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Cell 1: Environment Setup & Automated Resource Discovery\n"
                   "Installs required libraries, configures GPU device, scans `/kaggle/input/` for raw MELD data and companion bundle."]
    })

    # --------------------------------------------------------------------------
    # Cell 2: Code Cell 1
    # --------------------------------------------------------------------------
    cell_2_code = '''import os
import sys
import re
import glob
import json
import time
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available:  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    device = torch.device("cuda")
else:
    print("[WARNING] CUDA not available. Running on CPU (slower).")
    device = torch.device("cpu")

# Install minimal lightweight dependencies if missing
try:
    import av
    print(f"[OK] PyAV is installed (version: {av.__version__}).")
except ImportError:
    print("[INSTALL] Installing av (PyAV for fast video/audio decoding)...")
    os.system("pip install -q av")
    import av

try:
    import transformers
    print(f"[OK] Hugging Face Transformers version: {transformers.__version__}")
except ImportError:
    print("[INSTALL] Installing transformers...")
    os.system("pip install -q transformers")
    import transformers

# Create output and features directory
WORKING_DIR = Path("/kaggle/working")
FEATURES_DIR = WORKING_DIR / "features"
RESULTS_DIR = WORKING_DIR / "results"
CLIPS_EXTRACT_DIR = WORKING_DIR / "extracted_clips"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Automated Discovery of Input Datasets
# --------------------------------------------------------------------------
INPUT_ROOT = Path("/kaggle/input")
print(f"\\nScanning {INPUT_ROOT} for MELD datasets and companion bundle...")

# 1. Discover MELD CSVs
meld_csvs = {}
for split in ["train", "dev", "test"]:
    pattern = f"{split}_sent_emo.csv"
    matches = list(INPUT_ROOT.rglob(pattern))
    if matches:
        meld_csvs[split] = matches[0]
        print(f"  [+] Found {split} CSV: {matches[0]}")
    else:
        print(f"  [!] Missing {split} CSV: {pattern} (will fallback to companion NPZ)")

# 2. Discover Pre-extracted Features and Taxonomy
companion_files = {}
for target in [
    "meld_train_features.npz", "meld_dev_features.npz", "meld_test_features.npz",
    "disagreement_taxonomy.json", "distilbert_test_predictions.csv",
    "dialoguernn_unweighted_test_predictions.csv", "dialoguegcn_test_predictions.csv"
]:
    matches = list(INPUT_ROOT.rglob(target))
    if matches:
        companion_files[target] = matches[0]
        print(f"  [+] Found Companion File: {target} -> {matches[0]}")
    else:
        print(f"  [?] Not found: {target}")

# 3. Discover Raw .mp4 Video Clips
print("\\nScanning for .mp4 video clips across /kaggle/input/...")
all_mp4 = list(INPUT_ROOT.rglob("*.mp4")) + list(INPUT_ROOT.rglob("*.MP4"))
mp4_files = [p for p in all_mp4 if not p.name.startswith("._")]
print(f"Direct .mp4 clips detected: {len(mp4_files)}")

# If no .mp4 files found directly, check for tar.gz/zip archives (common in MELD.Raw)
if len(mp4_files) == 0:
    archive_files = (
        list(INPUT_ROOT.rglob("*.tar.gz"))
        + list(INPUT_ROOT.rglob("*.tgz"))
        + list(INPUT_ROOT.rglob("*.tar"))
        + list(INPUT_ROOT.rglob("*.zip"))
    )
    if archive_files:
        print(f"Found {len(archive_files)} archive file(s). Extracting video clips to {CLIPS_EXTRACT_DIR}...")
        CLIPS_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        for arch in archive_files:
            arch_low = arch.name.lower()
            if any(term in arch_low for term in ["train", "dev", "val", "test", "split", "meld"]):
                print(f"  Extracting archive: {arch.name}...")
                try:
                    if arch.suffix in [".zip"]:
                        with zipfile.ZipFile(arch, "r") as zf:
                            zf.extractall(CLIPS_EXTRACT_DIR)
                    else:
                        with tarfile.open(arch, "r:*") as tf:
                            tf.extractall(CLIPS_EXTRACT_DIR)
                except Exception as e:
                    print(f"  [!] Error extracting {arch.name}: {e}")
        all_ext_mp4 = list(CLIPS_EXTRACT_DIR.rglob("*.mp4")) + list(CLIPS_EXTRACT_DIR.rglob("*.MP4"))
        mp4_files = [p for p in all_ext_mp4 if not p.name.startswith("._")]
        print(f"Clips detected after archive extraction: {len(mp4_files)}")

# Robust Video Filename and Split Mapping
# Regex matches dia{d}_utt{u}.mp4$ anchored at end of filename (handles prefixes like final_videos_testdia0_utt0.mp4)
dia_utt_pattern = re.compile(r"dia(\\d+)_utt(\\d+)\\.mp4$", re.IGNORECASE)

def infer_split_from_path(p: Path) -> str:
    """Infer train/dev/test split reliably from directory structure or filename prefix."""
    for part in reversed(p.parts[:-1]):
        part_low = part.lower()
        if "train" in part_low:
            return "train"
        elif "dev" in part_low or "val" in part_low:
            return "dev"
        elif "test" in part_low:
            return "test"
    fname_low = p.name.lower()
    if "train" in fname_low:
        return "train"
    elif "dev" in fname_low or "val" in fname_low:
        return "dev"
    elif "test" in fname_low:
        return "test"
    return "unknown"

video_lookup = {}  # (split, dia_id, utt_id) -> Path, and fallback (dia_id, utt_id) -> Path
for p in mp4_files:
    if p.name.startswith("._"):
        continue
    m = dia_utt_pattern.search(p.name)
    if m:
        dia_id = int(m.group(1))
        utt_id = int(m.group(2))
        split = infer_split_from_path(p)
        video_lookup[(split, dia_id, utt_id)] = p
        video_lookup[(dia_id, utt_id)] = p

print(f"Successfully indexed {len(video_lookup)} video lookup entries.")
print("[OK] Cell 1 complete.")
'''
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_2_code]})

    # --------------------------------------------------------------------------
    # Cell 3: Markdown Acoustic
    # --------------------------------------------------------------------------
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Cell 2: Batched Acoustic Feature Extraction (microsoft/wavlm-base-plus)\n"
                   "Decodes audio streams via PyAV, resamples to 16,000 Hz 1D float32 mono, caps at 15.0s, and extracts 768-d representations."]
    })

    # --------------------------------------------------------------------------
    # Cell 4: Code Cell 2 (Acoustic)
    # --------------------------------------------------------------------------
    cell_4_code = '''from transformers import AutoFeatureExtractor, AutoModel
import av

WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
print(f"Loading WavLM model: {WAVLM_MODEL_ID}...")
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
audio_model = AutoModel.from_pretrained(WAVLM_MODEL_ID).to(device)
audio_model.eval()

def extract_audio_from_mp4(
    mp4_path: Union[str, Path],
    target_sr: int = 16000,
    max_seconds: float = 15.0
) -> Optional[np.ndarray]:
    """
    Decodes audio stream from mp4 using PyAV and resamples to 16,000 Hz 1D float32 mono.
    Rejects missing, corrupt, empty, or ultra-short (<400 samples / 25ms) waveforms.
    """
    if not mp4_path or not Path(mp4_path).exists():
        return None
    container = None
    try:
        container = av.open(str(mp4_path))
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            return None

        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        frames = []
        total_samples = 0
        max_samples = int(target_sr * max_seconds)

        for frame in container.decode(audio_streams[0]):
            resampled = resampler.resample(frame)
            for rf in resampled:
                arr = rf.to_ndarray().flatten().astype(np.float32) / 32768.0
                frames.append(arr)
                total_samples += len(arr)
                if total_samples >= max_samples:
                    break
            if total_samples >= max_samples:
                break

        if not frames:
            return None

        audio = np.concatenate(frames).astype(np.float32)[:max_samples]
        if audio is None or audio.ndim != 1 or audio.size < 400:
            return None
        return audio
    except Exception:
        return None
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

def extract_wavlm_features_for_split(
    split: str,
    video_map: Dict[Any, Path],
    meld_csv_path: Optional[Path] = None,
    companion_npz_path: Optional[Path] = None,
    save_dir: Path = FEATURES_DIR,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 768-d pooled WavLM representations for all utterances in a split.
    Corrupt, silent, or missing clips cleanly receive zero-vector with mask=0.0.
    """
    save_path = save_dir / f"audio_{split}.npz"
    if save_path.exists():
        print(f"[REUSE] Loading cached acoustic features for {split} from {save_path}...")
        data = np.load(save_path, allow_pickle=True)
        return data["features"], data["masks"]

    # Determine dialogue_id and utterance_id pairs from CSV or companion NPZ
    dia_ids, utt_ids = [], []
    if meld_csv_path and Path(meld_csv_path).exists():
        df = pd.read_csv(meld_csv_path)
        col_d = "Dialogue_ID" if "Dialogue_ID" in df.columns else "dialogue_id"
        col_u = "Utterance_ID" if "Utterance_ID" in df.columns else "utterance_id"
        dia_ids = df[col_d].astype(int).tolist()
        utt_ids = df[col_u].astype(int).tolist()
    elif companion_npz_path and Path(companion_npz_path).exists():
        npz = np.load(companion_npz_path, allow_pickle=True)
        dia_ids = (npz["dialogue_ids"] if "dialogue_ids" in npz else npz["dialogue_id"]).astype(int).tolist()
        utt_ids = (npz["utterance_ids"] if "utterance_ids" in npz else npz["utterance_id"]).astype(int).tolist()
    else:
        raise ValueError(f"Neither CSV nor companion NPZ available for {split} split!")

    N = len(dia_ids)
    features = np.zeros((N, 768), dtype=np.float32)
    masks = np.zeros((N,), dtype=np.float32)

    print(f"\\nExtracting WavLM acoustic features for {split} split ({N} utterances)...")
    batch_signals = []
    batch_indices = []

    for i in range(N):
        d_id = dia_ids[i]
        u_id = utt_ids[i]
        clip_path = video_map.get((split, d_id, u_id)) or video_map.get((d_id, u_id))

        audio = None
        if clip_path:
            audio = extract_audio_from_mp4(clip_path)

        if audio is not None and audio.size >= 400:
            batch_signals.append(audio)
            batch_indices.append(i)

        if len(batch_signals) >= batch_size or (i == N - 1 and batch_signals):
            try:
                inputs = audio_feature_extractor(batch_signals, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                        outputs = audio_model(input_values=input_values, attention_mask=attention_mask)
                        last_hidden = outputs.last_hidden_state
                        if attention_mask is not None:
                            feat_mask = audio_model._get_feature_vector_attention_mask(last_hidden.shape[1], attention_mask)
                            mask_expanded = feat_mask.unsqueeze(-1).to(last_hidden.dtype).expand_as(last_hidden)
                            pooled = (last_hidden * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1.0)
                        else:
                            pooled = last_hidden.mean(dim=1)

                pooled_np = pooled.cpu().float().numpy()
                for idx, emb in zip(batch_indices, pooled_np):
                    features[idx] = emb
                    masks[idx] = 1.0
            except Exception as batch_err:
                print(f"  [!] Batch WavLM extraction error ({batch_err}); falling back to per-item extraction...")
                for sig, b_idx in zip(batch_signals, batch_indices):
                    try:
                        inp_s = audio_feature_extractor([sig], sampling_rate=16000, return_tensors="pt", padding=True)
                        iv_s = inp_s.input_values.to(device)
                        am_s = inp_s.get("attention_mask", None)
                        if am_s is not None:
                            am_s = am_s.to(device)
                        with torch.no_grad():
                            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                                out_s = audio_model(input_values=iv_s, attention_mask=am_s)
                                lh_s = out_s.last_hidden_state
                                if am_s is not None:
                                    fm_s = audio_model._get_feature_vector_attention_mask(lh_s.shape[1], am_s)
                                    me_s = fm_s.unsqueeze(-1).to(lh_s.dtype).expand_as(lh_s)
                                    p_s = (lh_s * me_s).sum(dim=1) / torch.clamp(me_s.sum(dim=1), min=1.0)
                                else:
                                    p_s = lh_s.mean(dim=1)
                        features[b_idx] = p_s.squeeze(0).cpu().float().numpy()
                        masks[b_idx] = 1.0
                    except Exception as item_err:
                        print(f"    [!] Item {b_idx} extraction failed ({item_err}); masked out.")
                        features[b_idx] = np.zeros(768, dtype=np.float32)
                        masks[b_idx] = 0.0

            batch_signals.clear()
            batch_indices.clear()

        if (i + 1) % 500 == 0 or (i + 1) == N:
            valid_pct = (masks[:i+1].sum() / (i + 1)) * 100.0
            print(f"  Processed {i+1}/{N} clips | Active Audio: {valid_pct:.1f}%")

    np.savez_compressed(save_path, features=features, masks=masks)
    print(f"[SAVED] Acoustic features saved to {save_path} ({masks.sum():.0f}/{N} valid).")
    return features, masks

audio_features = {}
audio_masks = {}
for split in ["train", "dev", "test"]:
    csv_p = meld_csvs.get(split)
    npz_p = companion_files.get(f"meld_{split}_features.npz")
    if csv_p or npz_p:
        feat, mask = extract_wavlm_features_for_split(
            split=split,
            video_map=video_lookup,
            meld_csv_path=csv_p,
            companion_npz_path=npz_p
        )
        audio_features[split] = feat
        audio_masks[split] = mask

print("[OK] Cell 2 complete.")
'''
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_4_code]})

    # --------------------------------------------------------------------------
    # Cell 5: Markdown Visual
    # --------------------------------------------------------------------------
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Cell 3: Batched Visual Feature Extraction (openai/clip-vit-base-patch32)\n"
                   "Samples 4 uniform video frames per clip, encodes via CLIP-ViT, and mean-pools into 512-d representations."]
    })

    # --------------------------------------------------------------------------
    # Cell 6: Code Cell 3 (Visual)
    # --------------------------------------------------------------------------
    cell_6_code = '''import cv2
from transformers import CLIPModel, AutoImageProcessor

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
print(f"Loading CLIP Vision model: {CLIP_MODEL_ID}...")
clip_image_processor = AutoImageProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
clip_model.eval()

def extract_frames_from_mp4(
    mp4_path: Union[str, Path],
    num_frames: int = 4
) -> Optional[List[np.ndarray]]:
    """
    Extracts num_frames uniform RGB frames from mp4 video.
    Returns None if missing, corrupt, or 0 frames present.
    """
    if not mp4_path or not Path(mp4_path).exists():
        return None
    cap = None
    try:
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)

        if not frames:
            return None
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return frames
    except Exception:
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

def extract_clip_features_for_split(
    split: str,
    video_map: Dict[Any, Path],
    meld_csv_path: Optional[Path] = None,
    companion_npz_path: Optional[Path] = None,
    save_dir: Path = FEATURES_DIR,
    batch_size: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 512-d pooled CLIP ViT representations for all utterances in a split.
    Corrupt or missing video clips cleanly receive zero-vector with mask=0.0.
    """
    save_path = save_dir / f"video_{split}.npz"
    if save_path.exists():
        print(f"[REUSE] Loading cached visual features for {split} from {save_path}...")
        data = np.load(save_path, allow_pickle=True)
        return data["features"], data["masks"]

    dia_ids, utt_ids = [], []
    if meld_csv_path and Path(meld_csv_path).exists():
        df = pd.read_csv(meld_csv_path)
        col_d = "Dialogue_ID" if "Dialogue_ID" in df.columns else "dialogue_id"
        col_u = "Utterance_ID" if "Utterance_ID" in df.columns else "utterance_id"
        dia_ids = df[col_d].astype(int).tolist()
        utt_ids = df[col_u].astype(int).tolist()
    elif companion_npz_path and Path(companion_npz_path).exists():
        npz = np.load(companion_npz_path, allow_pickle=True)
        dia_ids = (npz["dialogue_ids"] if "dialogue_ids" in npz else npz["dialogue_id"]).astype(int).tolist()
        utt_ids = (npz["utterance_ids"] if "utterance_ids" in npz else npz["utterance_id"]).astype(int).tolist()
    else:
        raise ValueError(f"Neither CSV nor companion NPZ available for {split} split!")

    N = len(dia_ids)
    features = np.zeros((N, 512), dtype=np.float32)
    masks = np.zeros((N,), dtype=np.float32)

    print(f"\\nExtracting CLIP visual features for {split} split ({N} utterances)...")
    batch_clips = []
    batch_indices = []

    for i in range(N):
        d_id = dia_ids[i]
        u_id = utt_ids[i]
        clip_path = video_map.get((split, d_id, u_id)) or video_map.get((d_id, u_id))

        frames = None
        if clip_path:
            frames = extract_frames_from_mp4(clip_path, num_frames=4)

        if frames is not None and len(frames) == 4:
            batch_clips.append(frames)
            batch_indices.append(i)

        if len(batch_clips) >= batch_size or (i == N - 1 and batch_clips):
            flat_images = [img for clip in batch_clips for img in clip]
            try:
                inputs = clip_image_processor(images=flat_images, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(device)

                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                        feats = clip_model.get_image_features(pixel_values=pixel_values)
                        if not isinstance(feats, torch.Tensor):
                            feats = feats.pooler_output
                        frame_embs_512 = feats[:, :512]
                        clip_embs = frame_embs_512.view(len(batch_clips), 4, 512).mean(dim=1)

                embs_np = clip_embs.cpu().float().numpy()
                for idx, emb in zip(batch_indices, embs_np):
                    features[idx] = emb
                    masks[idx] = 1.0
            except Exception as batch_err:
                print(f"  [!] Batch CLIP extraction error ({batch_err}); falling back to per-item extraction...")
                for clip, b_idx in zip(batch_clips, batch_indices):
                    try:
                        inp_s = clip_image_processor(images=clip, return_tensors="pt")
                        pv_s = inp_s.pixel_values.to(device)
                        with torch.no_grad():
                            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                                f_s = clip_model.get_image_features(pixel_values=pv_s)
                                if not isinstance(f_s, torch.Tensor):
                                    f_s = f_s.pooler_output
                                ce_s = f_s[:, :512].mean(dim=0)
                        features[b_idx] = ce_s.cpu().float().numpy()
                        masks[b_idx] = 1.0
                    except Exception as item_err:
                        print(f"    [!] Item {b_idx} CLIP extraction failed ({item_err}); masked out.")
                        features[b_idx] = np.zeros(512, dtype=np.float32)
                        masks[b_idx] = 0.0

            batch_clips.clear()
            batch_indices.clear()

        if (i + 1) % 500 == 0 or (i + 1) == N:
            valid_pct = (masks[:i+1].sum() / (i + 1)) * 100.0
            print(f"  Processed {i+1}/{N} clips | Active Video: {valid_pct:.1f}%")

    np.savez_compressed(save_path, features=features, masks=masks)
    print(f"[SAVED] Visual features saved to {save_path} ({masks.sum():.0f}/{N} valid).")
    return features, masks

video_features = {}
video_masks = {}
for split in ["train", "dev", "test"]:
    csv_p = meld_csvs.get(split)
    npz_p = companion_files.get(f"meld_{split}_features.npz")
    if csv_p or npz_p:
        feat, mask = extract_clip_features_for_split(
            split=split,
            video_map=video_lookup,
            meld_csv_path=csv_p,
            companion_npz_path=npz_p
        )
        video_features[split] = feat
        video_masks[split] = mask

print("[OK] Cell 3 complete.")
'''
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_6_code]})

    # --------------------------------------------------------------------------
    # Cell 7: Markdown Alignment
    # --------------------------------------------------------------------------
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Cell 4: Multimodal Feature Alignment & Canonical Cohort A Verification\n"
                   "Loads text NPZs with `allow_pickle=True`, validates exact schemas, constructs dialogue sequences, and audits Canonical Cohort A ($N = 2,328$)."]
    })

    # --------------------------------------------------------------------------
    # Cell 8: Code Cell 4 (Alignment)
    # --------------------------------------------------------------------------
    cell_8_code = '''EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}

def load_text_features(companion_files: Dict[str, Path]) -> Dict[str, Any]:
    out = {}
    for split in ["train", "dev", "test"]:
        fname = f"meld_{split}_features.npz"
        if fname in companion_files:
            print(f"Loading {fname} from {companion_files[fname]}...")
            data = np.load(companion_files[fname], allow_pickle=True)
            out[split] = data
        else:
            raise FileNotFoundError(f"Missing {fname} in companion files!")
    return out

text_data = load_text_features(companion_files)

class DialogueDataset:
    def __init__(
        self,
        split: str,
        text_npz: Any,
        audio_feat: np.ndarray,
        audio_m: np.ndarray,
        video_feat: np.ndarray,
        video_m: np.ndarray
    ):
        self.split = split

        # Exact audited schema keys from companion NPZ
        text_features = text_npz["features"]
        text_labels = text_npz["label_ids"] if "label_ids" in text_npz else text_npz["labels"]
        text_speakers = text_npz["speakers"] if "speakers" in text_npz else text_npz["speaker"]
        text_dia_ids = text_npz["dialogue_ids"] if "dialogue_ids" in text_npz else text_npz["dialogue_id"]
        text_utt_ids = text_npz["utterance_ids"] if "utterance_ids" in text_npz else text_npz["utterance_id"]

        self.dialogues = []
        self.cohort_a_pairs = []
        unique_dias = np.unique(text_dia_ids)

        for d_id in unique_dias:
            indices = np.where(text_dia_ids == d_id)[0]
            order = np.argsort(text_utt_ids[indices])
            sorted_idx = indices[order]

            d_text = torch.from_numpy(text_features[sorted_idx]).float()
            d_audio = torch.from_numpy(audio_feat[sorted_idx]).float()
            d_video = torch.from_numpy(video_feat[sorted_idx]).float()
            d_audio_m = torch.from_numpy(audio_m[sorted_idx]).float()
            d_video_m = torch.from_numpy(video_m[sorted_idx]).float()
            d_labels = torch.from_numpy(text_labels[sorted_idx]).long()

            d_spk = text_speakers[sorted_idx]
            spk_to_id = {}
            p_ids = []
            for s in d_spk:
                if s not in spk_to_id:
                    spk_to_id[s] = len(spk_to_id)
                p_ids.append(spk_to_id[s])
            d_parties = torch.tensor(p_ids, dtype=torch.long)
            d_utts = text_utt_ids[sorted_idx]

            # Canonical Cohort A condition: target_index >= 1 and contiguous turn (u_t == u_{t-1} + 1)
            cohort_a_mask = torch.zeros(len(sorted_idx), dtype=torch.bool)
            for t in range(1, len(sorted_idx)):
                if d_utts[t] == d_utts[t-1] + 1:
                    cohort_a_mask[t] = True
                    if split == "test":
                        self.cohort_a_pairs.append((int(d_id), int(d_utts[t])))

            self.dialogues.append({
                "dialogue_id": int(d_id),
                "length": len(sorted_idx),
                "text": d_text,
                "audio": d_audio,
                "video": d_video,
                "audio_mask": d_audio_m,
                "video_mask": d_video_m,
                "labels": d_labels,
                "party_ids": d_parties,
                "speakers": list(d_spk),
                "utterance_ids": d_utts,
                "cohort_a_mask": cohort_a_mask,
            })

datasets = {}
for split in ["train", "dev", "test"]:
    datasets[split] = DialogueDataset(
        split=split,
        text_npz=text_data[split],
        audio_feat=audio_features[split],
        audio_m=audio_masks[split],
        video_feat=video_features[split],
        video_m=video_masks[split]
    )
    print(f"Dataset {split}: {len(datasets[split].dialogues)} dialogues.")

test_dialogues = datasets["test"].dialogues
total_test_turns = sum(d["length"] for d in test_dialogues)
cohort_a_test_turns = sum(d["cohort_a_mask"].sum().item() for d in test_dialogues)
cohort_a_test_pairs = datasets["test"].cohort_a_pairs

print(f"\\nVerification of Canonical Cohort A on Test Split:")
print(f"  Full Test Turns:        {total_test_turns} (Expected: 2,610)")
print(f"  Cohort A Matched Turns: {cohort_a_test_turns} (Expected: 2,328)")
assert total_test_turns == 2610, f"Expected 2610 test turns, got {total_test_turns}"
assert cohort_a_test_turns == 2328, f"Expected 2328 Cohort A test turns, got {cohort_a_test_turns}"
assert len(cohort_a_test_pairs) == 2328, f"Expected 2328 pairs, got {len(cohort_a_test_pairs)}"
print("[PASS] Canonical Cohort A test size exactly matches 2,328 turns.")
print("[OK] Cell 4 complete.")
'''
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_8_code]})

    # --------------------------------------------------------------------------
    # Cell 9: Markdown Models
    # --------------------------------------------------------------------------
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Cell 5: Model Architectures, Training Loop, and Canonical Cohort A Evaluation\n"
                   "Inlines `GatedMultimodalFusion`, `DialogueRNN`, `DialogueGCN`, and trains 4 multimodal models with AdamW and early stopping."]
    })

    # --------------------------------------------------------------------------
    # Cell 10: Code Cell 5 (Models & Training)
    # --------------------------------------------------------------------------
    cell_10_code = '''from sklearn.metrics import classification_report, f1_score, accuracy_score

class GatedMultimodalFusion(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 512,
        out_dim: int = 256,
        modalities: Tuple[str, ...] = ("text", "audio", "video"),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.modalities = tuple(modalities)
        dim_map = {"text": text_dim, "audio": audio_dim, "video": video_dim}

        self.input_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.gates = nn.ModuleDict()

        for m in self.modalities:
            in_d = dim_map[m]
            self.input_norms[m] = nn.LayerNorm(in_d)
            self.projections[m] = nn.Sequential(
                nn.Linear(in_d, out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.gates[m] = nn.Linear(out_dim, out_dim)

        self.fusion_norm = nn.LayerNorm(out_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_2d = (text.dim() == 2)
        if is_2d:
            text = text.unsqueeze(0)
            if audio is not None: audio = audio.unsqueeze(0)
            if video is not None: video = video.unsqueeze(0)
            if audio_mask is not None: audio_mask = audio_mask.unsqueeze(0)
            if video_mask is not None: video_mask = video_mask.unsqueeze(0)

        B, T, _ = text.shape
        device = text.device
        dtype = text.dtype

        mod_in = {"text": text}
        if "audio" in self.modalities:
            mod_in["audio"] = audio if audio is not None else torch.zeros(B, T, 768, device=device, dtype=dtype)
        if "video" in self.modalities:
            mod_in["video"] = video if video is not None else torch.zeros(B, T, 512, device=device, dtype=dtype)

        mod_mask = {
            "text": torch.ones(B, T, 1, device=device, dtype=dtype),
            "audio": (audio_mask.unsqueeze(-1) if audio_mask is not None else torch.ones(B, T, 1, device=device, dtype=dtype)),
            "video": (video_mask.unsqueeze(-1) if video_mask is not None else torch.ones(B, T, 1, device=device, dtype=dtype)),
        }

        gated_feats = []
        active_counts = torch.zeros(B, T, 1, device=device, dtype=dtype)

        for m in self.modalities:
            inp = torch.nan_to_num(mod_in[m], nan=0.0)
            m_mask = mod_mask[m]
            normed = self.input_norms[m](inp)
            h_m = self.projections[m](normed)
            g_m = torch.sigmoid(self.gates[m](h_m))
            effective_gate = g_m * m_mask
            h_gated = effective_gate * h_m
            gated_feats.append(h_gated)
            active_counts = active_counts + m_mask

        fused = torch.stack(gated_feats, dim=0).sum(dim=0)
        scale = torch.clamp(active_counts, min=1.0)
        scaled = fused / torch.sqrt(scale)
        u_t = self.fusion_norm(scaled + self.fusion_mlp(scaled))

        return u_t.squeeze(0) if is_2d else u_t

class DialogueRNN(nn.Module):
    def __init__(self, feature_dim: int = 256, global_dim: int = 128, party_dim: int = 128, emotion_dim: int = 64, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)
        self.global_gru = nn.GRUCell(256 + party_dim, global_dim)
        self.context_attn = nn.Linear(global_dim, global_dim, bias=False)
        self.party_gru = nn.GRUCell(256 + global_dim, party_dim)
        self.emotion_gru = nn.GRUCell(party_dim, emotion_dim)
        self.fc = nn.Sequential(
            nn.Linear(emotion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.global_dim = global_dim
        self.party_dim = party_dim
        self.emotion_dim = emotion_dim

    def forward(self, features: torch.Tensor, party_ids: torch.Tensor) -> torch.Tensor:
        T, D = features.shape
        device = features.device
        proj = self.dropout(self.layer_norm(self.input_proj(features)))

        num_parties = int(party_ids.max().item()) + 1 if party_ids.numel() > 0 else 1
        q = [torch.zeros(self.party_dim, device=device) for _ in range(num_parties)]
        g = torch.zeros(self.global_dim, device=device)
        e = [torch.zeros(self.emotion_dim, device=device) for _ in range(num_parties)]

        logits_list = []
        g_history = []

        for t in range(T):
            u_t = proj[t]
            p_t = int(party_ids[t].item())
            q_prev = q[p_t]

            inp_g = torch.cat([u_t, q_prev], dim=-1).unsqueeze(0)
            g = self.global_gru(inp_g, g.unsqueeze(0)).squeeze(0)
            g_history.append(g)

            H_g = torch.stack(g_history, dim=0)
            scores = torch.matmul(H_g, self.context_attn(g)) / np.sqrt(self.global_dim)
            alpha = torch.softmax(scores, dim=0)
            c_t = (alpha.unsqueeze(-1) * H_g).sum(dim=0)

            inp_p = torch.cat([u_t, c_t], dim=-1).unsqueeze(0)
            q[p_t] = self.party_gru(inp_p, q[p_t].unsqueeze(0)).squeeze(0)

            e[p_t] = self.emotion_gru(q[p_t].unsqueeze(0), e[p_t].unsqueeze(0)).squeeze(0)
            logits_t = self.fc(e[p_t])
            logits_list.append(logits_t)

        return torch.stack(logits_list, dim=0)

class RGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_relations: int = 2):
        super().__init__()
        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_rel = nn.Parameter(torch.empty(num_relations, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.attn_proj = nn.Linear(in_features, in_features, bias=False)
        nn.init.xavier_uniform_(self.w_rel)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, rel_mask: torch.Tensor) -> torch.Tensor:
        proj = self.attn_proj(h)
        scores = torch.matmul(h, proj.transpose(-1, -2))
        scores = scores.masked_fill(~adj.bool(), -1e9)
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha.masked_fill(~adj.bool(), 0.0)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        out = self.w_self(h)
        for r in range(2):
            w_r = self.w_rel[r]
            r_mask = rel_mask[r]
            weighted_adj = alpha * r_mask
            out = out + torch.matmul(weighted_adj, torch.matmul(h, w_r))

        return torch.relu(out + self.bias)

class DialogueGCN(nn.Module):
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128, num_classes: int = 7, dropout: float = 0.1):
        super().__init__()
        self.bigru = nn.GRU(feature_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.rgcn1 = RGCNLayer(hidden_dim, hidden_dim)
        self.rgcn2 = RGCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def build_graph(self, party_ids: torch.Tensor, wp: int = 10, wf: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        T = party_ids.shape[0]
        device = party_ids.device
        adj = torch.zeros(T, T, device=device)
        rel_mask = torch.zeros(2, T, T, device=device)
        if T <= 1:
            return adj, rel_mask
        for i in range(T):
            start = max(0, i - wp)
            end = min(T, i + wf + 1)
            for j in range(start, end):
                if i != j:
                    adj[i, j] = 1.0
                    if party_ids[i] == party_ids[j]:
                        rel_mask[0, i, j] = 1.0
                    else:
                        rel_mask[1, i, j] = 1.0
        return adj, rel_mask

    def forward(self, features: torch.Tensor, party_ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.bigru(features.unsqueeze(0))
        h = h.squeeze(0)
        adj, rel_mask = self.build_graph(party_ids)
        h = self.rgcn1(h, adj, rel_mask)
        h = self.rgcn2(h, adj, rel_mask)
        h = self.dropout(h)
        return self.classifier(h)

class MultimodalModel(nn.Module):
    def __init__(self, backbone: str, modalities: Tuple[str, ...]):
        super().__init__()
        self.backbone_name = backbone
        self.modalities = modalities
        self.fusion = GatedMultimodalFusion(modalities=modalities, out_dim=256)
        if backbone == "rnn":
            self.net = DialogueRNN(feature_dim=256)
        elif backbone == "gcn":
            self.net = DialogueGCN(feature_dim=256)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, dia: Dict[str, Any]) -> torch.Tensor:
        u_t = self.fusion(
            text=dia["text"].to(device),
            audio=dia["audio"].to(device) if "audio" in self.modalities else None,
            video=dia["video"].to(device) if "video" in self.modalities else None,
            audio_mask=dia["audio_mask"].to(device) if "audio" in self.modalities else None,
            video_mask=dia["video_mask"].to(device) if "video" in self.modalities else None,
        )
        return self.net(u_t, dia["party_ids"].to(device))

def train_and_eval_model(name: str, backbone: str, modalities: Tuple[str, ...], epochs: int = 12, lr: float = 3e-4) -> Dict[str, Any]:
    print(f"\\n{'='*70}\\nTRAINING {name} [{backbone.upper()} | Modalities: {modalities}]\\n{'='*70}")
    torch.manual_seed(42)
    model = MultimodalModel(backbone=backbone, modalities=modalities).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    best_dev_macro = 0.0
    best_state = None

    train_dias = datasets["train"].dialogues
    dev_dias = datasets["dev"].dialogues

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        perm = np.random.permutation(len(train_dias))

        for idx in perm:
            dia = train_dias[idx]
            optimizer.zero_grad()
            logits = model(dia)
            targets = dia["labels"].to(device)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        dev_preds, dev_targets = [], []
        with torch.no_grad():
            for dia in dev_dias:
                logits = model(dia)
                preds = logits.argmax(dim=-1).cpu().numpy()
                dev_preds.extend(preds)
                dev_targets.extend(dia["labels"].numpy())

        dev_macro = f1_score(dev_targets, dev_preds, average="macro", zero_division=0)
        dev_acc = accuracy_score(dev_targets, dev_preds)
        scheduler.step(dev_macro)

        if dev_macro > best_dev_macro:
            best_dev_macro = dev_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = "(* Best Dev Macro F1)"
        else:
            mark = ""

        if epoch % 3 == 0 or epoch == epochs or mark:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/len(train_dias):.4f} | Dev Acc: {dev_acc*100:.2f}% | Dev Macro F1: {dev_macro*100:.2f}% {mark}")

    ckpt_path = RESULTS_DIR / f"{name}_best.pt"
    torch.save(best_state, ckpt_path)
    loaded_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded_state)
    model.eval()

    test_dias = datasets["test"].dialogues
    cohort_preds = []
    cohort_targets = []

    with torch.no_grad():
        for dia in test_dias:
            logits = model(dia)
            preds = logits.argmax(dim=-1).cpu().numpy()
            mask_a = dia["cohort_a_mask"].numpy()
            targets_a = dia["labels"].numpy()
            cohort_preds.extend(preds[mask_a])
            cohort_targets.extend(targets_a[mask_a])

    cohort_preds = np.array(cohort_preds)
    cohort_targets = np.array(cohort_targets)
    assert len(cohort_preds) == 2328, f"Expected 2328 Cohort A predictions, got {len(cohort_preds)}"

    acc = accuracy_score(cohort_targets, cohort_preds)
    macro_f1 = f1_score(cohort_targets, cohort_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(cohort_targets, cohort_preds, average="weighted", zero_division=0)
    per_class = f1_score(cohort_targets, cohort_preds, average=None, zero_division=0)

    print(f"\\n--- Canonical Cohort A Benchmark Results (N=2,328) for {name} ---")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  Macro F1:    {macro_f1*100:.2f}%")
    print(f"  Weighted F1: {weighted_f1*100:.2f}%")
    for emo, score in zip(EMOTIONS, per_class):
        print(f"    {emo:<10}: F1 = {score*100:.2f}%")

    return {
        "model_name": name,
        "backbone": backbone,
        "modalities": "+".join(modalities),
        "cohort_a_accuracy": float(acc),
        "cohort_a_macro_f1": float(macro_f1),
        "cohort_a_weighted_f1": float(weighted_f1),
        "per_class_f1": {emo: float(s) for emo, s in zip(EMOTIONS, per_class)},
        "predictions": cohort_preds,
        "targets": cohort_targets,
    }

models_to_train = [
    ("Bimodal_DialogueRNN", "rnn", ("text", "audio")),
    ("Bimodal_DialogueGCN", "gcn", ("text", "audio")),
    ("Trimodal_DialogueRNN", "rnn", ("text", "audio", "video")),
    ("Trimodal_DialogueGCN", "gcn", ("text", "audio", "video")),
]

results = {}
for name, bb, mods in models_to_train:
    results[name] = train_and_eval_model(name, bb, mods, epochs=12, lr=3e-4)

print("[OK] Cell 5 complete.")
'''
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_10_code]})

    # --------------------------------------------------------------------------
    # Cell 11: Markdown Recovery & Export
    # --------------------------------------------------------------------------
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Cell 6: Universal Error Recovery Audit & Export Packaging\n"
                   "Reconstructs the 737 universal error mask on Cohort A, audits recovered turns, generates comparison tables/plots, and packages `multimodal_results_bundle.zip`."]
    })

    # --------------------------------------------------------------------------
    # Cell 12: Code Cell 6 (Export & Packaging)
    # --------------------------------------------------------------------------
    cell_12_code = '''import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("\\n" + "="*70 + "\\nSTATISTICAL COMPARISON & UNIVERSAL ERROR RECOVERY AUDIT\\n" + "="*70)

# 1. Derive exact 737 Universal Text Error Mask on Cohort A (N=2,328)
universal_error_mask = np.zeros(2328, dtype=bool)

df_db_path = companion_files.get("distilbert_test_predictions.csv")
df_rnn_path = companion_files.get("dialoguernn_unweighted_test_predictions.csv")
df_gcn_path = companion_files.get("dialoguegcn_test_predictions.csv")

if df_db_path and df_rnn_path and df_gcn_path:
    df_db = pd.read_csv(df_db_path)
    df_rnn = pd.read_csv(df_rnn_path)
    df_gcn = pd.read_csv(df_gcn_path)

    db_map = dict(zip(zip(df_db["dialogue_id"].astype(int), df_db["utterance_id"].astype(int)), df_db["predicted_emotion"].str.lower()))
    rnn_map = dict(zip(zip(df_rnn["dialogue_id"].astype(int), df_rnn["utterance_id"].astype(int)), df_rnn["pred_emotion"].str.lower()))
    gcn_map = dict(zip(zip(df_gcn["dialogue_id"].astype(int), df_gcn["utterance_id"].astype(int)), df_gcn["pred_emotion"].str.lower()))
    gold_map = dict(zip(zip(df_db["dialogue_id"].astype(int), df_db["utterance_id"].astype(int)), df_db["true_emotion"].str.lower()))

    for idx, (d_id, u_id) in enumerate(cohort_a_test_pairs):
        pair = (int(d_id), int(u_id))
        g = gold_map.get(pair)
        pdb = db_map.get(pair)
        prnn = rnn_map.get(pair)
        pgcn = gcn_map.get(pair)
        all_wrong = (pdb != g) and (prnn != g) and (pgcn != g)
        all_disagree = all_wrong and (pdb != prnn) and (pdb != pgcn) and (prnn != pgcn)
        if all_wrong and not all_disagree:
            universal_error_mask[idx] = True

num_universal_errors = int(universal_error_mask.sum())
print(f"Universal text errors identified on Cohort A: {num_universal_errors} / 2328 (Expected: 737)")

# 2. Build Benchmark Comparison Table with exact expected headers
summary_rows = []
for name, res in results.items():
    preds = res["predictions"]
    targets = res["targets"]
    if num_universal_errors > 0:
        recovered = int((preds[universal_error_mask] == targets[universal_error_mask]).sum())
        rec_str = f"{recovered}"
    else:
        recovered = 0
        rec_str = "N/A"

    res["recovered_universal_errors"] = recovered

    summary_rows.append({
        "Model": name,
        "Modalities": res["modalities"],
        "CohortA_Accuracy": f"{res['cohort_a_accuracy']*100:.2f}%",
        "CohortA_Macro_F1": f"{res['cohort_a_macro_f1']:.4f}",
        "CohortA_Weighted_F1": f"{res['cohort_a_weighted_f1']:.4f}",
        "Universal_Errors_Recovered": rec_str,
        "Fear_F1": f"{res['per_class_f1']['fear']*100:.2f}%",
        "Disgust_F1": f"{res['per_class_f1']['disgust']*100:.2f}%",
    })

comparison_df = pd.DataFrame(summary_rows)
comparison_df.to_csv(RESULTS_DIR / "multimodal_comparison.csv", index=False)
print("\\n--- Canonical Benchmark Comparison Table ---")
print(comparison_df.to_string(index=False))

# 3. Plot Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(res["targets"], res["predictions"], labels=range(7))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=axes[i], cbar=False)
    axes[i].set_title(f"{name} (Macro F1: {res['cohort_a_macro_f1']*100:.2f}%)")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Ground Truth")

plt.tight_layout()
cm_plot_path = RESULTS_DIR / "multimodal_confusion_matrices.png"
plt.savefig(cm_plot_path, dpi=200)
plt.close()
print(f"\\n[SAVED] Confusion matrices plot saved to {cm_plot_path}")

# 4. Save Structured JSON Summary matching ingest_kaggle_multimodal_results.py
clean_summary = {
    "total_turns": 2328,
    "universal_text_errors": num_universal_errors,
    "models": {
        name: {
            "modalities": res["modalities"],
            "cohort_a_accuracy": float(res["cohort_a_accuracy"]),
            "cohort_a_macro_f1": float(res["cohort_a_macro_f1"]),
            "cohort_a_weighted_f1": float(res["cohort_a_weighted_f1"]),
            "recovered_universal_errors": int(res.get("recovered_universal_errors", 0)),
            "per_class_f1": res["per_class_f1"],
        }
        for name, res in results.items()
    }
}

summary_json_path = RESULTS_DIR / "multimodal_summary.json"
with open(summary_json_path, "w") as f:
    json.dump(clean_summary, f, indent=2)
print(f"[SAVED] Structured JSON summary saved to {summary_json_path}")

# 5. Package Working Results into Downloadable Zip Bundle
BUNDLE_ZIP = WORKING_DIR / "multimodal_results_bundle.zip"
with zipfile.ZipFile(BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in RESULTS_DIR.iterdir():
        zf.write(f, arcname=f.name)

print(f"\\n" + "="*70)
print(f"[SUCCESS] Multimodal Results Bundle created: {BUNDLE_ZIP} ({BUNDLE_ZIP.stat().st_size / (1024**2):.2f} MB)")
print("Ready for 1-click download from Kaggle.")
print("="*70)
'''
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_12_code]})

    # Write notebook file
    nb_path = Path("notebooks/30_kaggle_multimodal_meld.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Saved notebook: {nb_path} ({len(nb['cells'])} cells)")

    # Verify AST of every code cell
    print("Validating Python AST of all code cells...")
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code_str = "".join(cell["source"])
            ast.parse(code_str)
            print(f"  [OK] Cell {idx} AST parsed successfully.")

    print("All notebook cells successfully verified!")


if __name__ == "__main__":
    create_notebook()
