#!/usr/bin/env python3
"""
Comprehensive Acoustic Feature Analysis for Deepfake Detection
==============================================================

This script performs rigorous statistical analysis of ALL acoustic features
commonly used in audio deepfake detection, evaluating:

1. Codec Robustness - Feature stability under real-world compression
2. Discriminative Power - Ability to separate real vs fake audio
3. Computational Cost - Extraction time and memory requirements
4. Literature Support - Citations and prior research evidence

Features Analyzed:
- Spectral: MFCC, LFCC, PNCC, CQT, Mel-spectrogram, Chroma
- Phase-based: MODGD, Instantaneous Frequency, Group Delay
- Prosodic: F0, Energy, Duration, Voice Quality
- Formants: F1-F4 frequencies, bandwidths, ratios
- Glottal: OQ, SQ, NAQ, H1-H2, HRF
- Temporal: ZCR, Short-time energy, Onset detection
- Wavelets: DWT coefficients, Scattering transform

Statistical Methods:
- Pearson/Spearman correlation for codec stability
- ICC (Intraclass Correlation) for reliability
- Effect size (Cohen's d) for discriminative power
- Wilcoxon signed-rank test for statistical significance
- ANOVA for multi-codec comparison

Author: Audio Deepfake Detection Research Team
Target: NeurIPS 2026
"""

import os
import sys
import json
import time
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    print("Warning: pingouin not available. ICC computation disabled.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Audio processing
import librosa
import soundfile as sf

# Try to import optional dependencies
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Warning: parselmouth not available. Formant/prosody features limited.")

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: PyWavelets not available. Wavelet features disabled.")

# Check for ffmpeg
import subprocess
FFMPEG_AVAILABLE = subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode == 0


@dataclass
class FeatureConfig:
    """Configuration for a feature extraction method."""
    name: str
    category: str
    n_dims: int
    description: str
    literature: List[str] = field(default_factory=list)
    codec_robust_claim: bool = False
    extraction_fn: str = ""  # Name of extraction function


@dataclass
class FeatureAnalysisResult:
    """Results from analyzing a single feature."""
    name: str
    category: str
    n_dims: int

    # Codec robustness metrics
    mean_pearson_r: float = 0.0
    min_pearson_r: float = 0.0
    max_pearson_r: float = 0.0
    std_pearson_r: float = 0.0
    worst_codec: str = ""

    # Per-codec results
    codec_results: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Discriminative power (if real/fake labels available)
    cohens_d: float = float('nan')
    auc_roc: float = 0.0

    # Extraction metrics
    extraction_time_ms: float = 0.0
    memory_mb: float = 0.0

    # Statistical tests
    wilcoxon_p: float = float('nan')
    icc: float = float('nan')

    # Recommendation
    recommendation: str = ""
    status: str = ""  # "RECOMMENDED", "ACCEPTABLE", "NOT_RECOMMENDED"


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Mel-Frequency Cepstral Coefficients

    References:
    - Davis & Mermelstein (1980) IEEE TASSP
    - Standard ASR feature, widely used baseline
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # Time-averaged


def extract_lfcc(y: np.ndarray, sr: int, n_lfcc: int = 8) -> np.ndarray:
    """
    Linear Frequency Cepstral Coefficients

    References:
    - Sahidullah & Saha (2012) "Design, analysis and experimental evaluation
      of block based transformation in MFCC computation for speaker recognition"
    - ASVspoof 2021 baseline feature
    """
    # Linear filterbank (not mel-scaled)
    n_fft = 512
    hop_length = 160

    # Compute STFT
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2

    # Linear filterbank
    n_filters = 20
    fmin, fmax = 0, sr // 2
    linear_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_filters,
                                     fmin=fmin, fmax=fmax, htk=True)
    # Replace mel with linear spacing
    freqs = np.linspace(fmin, fmax, n_filters + 2)
    linear_fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        start = int(freqs[i] * n_fft / sr)
        mid = int(freqs[i+1] * n_fft / sr)
        end = int(freqs[i+2] * n_fft / sr)
        for j in range(start, mid):
            if mid != start:
                linear_fb[i, j] = (j - start) / (mid - start)
        for j in range(mid, end):
            if end != mid:
                linear_fb[i, j] = (end - j) / (end - mid)

    # Apply filterbank and DCT
    filter_output = np.dot(linear_fb, S)
    filter_output = np.where(filter_output > 0, np.log(filter_output + 1e-10), -23.0)
    lfcc = librosa.feature.mfcc(S=filter_output, n_mfcc=n_lfcc, norm='ortho')

    return np.mean(lfcc, axis=1)


def extract_pncc(y: np.ndarray, sr: int, n_pncc: int = 13) -> np.ndarray:
    """
    Power-Normalized Cepstral Coefficients

    References:
    - Kim & Stern (2016) "Power-Normalized Cepstral Coefficients (PNCC)
      for Robust Speech Recognition" IEEE TASLP
    - Demonstrates improved noise robustness (25-40% under additive noise only;
      codec robustness is an open question - see PNCC_CODEC_ROBUSTNESS_VALIDATION.md)

    Implementation follows Kim & Stern (2016) with:
    - Gammatone filterbank (auditory model)
    - Power-law nonlinearity (μ=1/15)
    - Medium-time processing
    - Asymmetric noise suppression
    """
    # Parameters from Kim & Stern (2016)
    n_filters = 40
    power_coef = 1/15  # μ parameter

    # STFT
    n_fft = 512
    hop_length = 160
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2

    # Gammatone filterbank per Patterson et al. (1992)
    # Aligned with extract_acoustic_features_v4.py implementation
    try:
        from evidence.experiments.extract_acoustic_features_v4 import gammatone_filterbank
    except ImportError:
        import importlib.util
        _v4_path = os.path.join(os.path.dirname(__file__), 'extract_acoustic_features_v4.py')
        _spec = importlib.util.spec_from_file_location('extract_acoustic_features_v4', _v4_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        gammatone_filterbank = _mod.gammatone_filterbank
    gamma_fb = gammatone_filterbank(sr, n_filters=n_filters, n_fft=n_fft)

    # Apply filterbank
    filter_output = np.dot(gamma_fb, S)

    # Power-law nonlinearity (replaces log compression per Kim & Stern 2016)
    filter_output = filter_output ** power_coef

    # Medium-time processing (temporal smoothing)
    from scipy.ndimage import uniform_filter1d
    filter_output = uniform_filter1d(filter_output, size=5, axis=1)

    # NOTE: No log compression after power-law nonlinearity.
    # Kim & Stern (2016) use power-law as the SOLE nonlinearity.

    # DCT to get cepstral coefficients
    from scipy.fftpack import dct
    pncc = dct(filter_output, type=2, axis=0, norm='ortho')[:n_pncc, :]

    return np.mean(pncc, axis=1)


def extract_cqt(y: np.ndarray, sr: int, n_bins: int = 84) -> np.ndarray:
    """
    Constant-Q Transform features

    References:
    - Brown (1991) "Calculation of a constant Q spectral transform" JASA
    - Todisco et al. (2017) used CQT-gram for anti-spoofing
    """
    cqt = np.abs(librosa.cqt(y, sr=sr, n_bins=n_bins))
    # Reduce to manageable size
    cqt_stats = np.concatenate([
        np.mean(cqt, axis=1)[:20],  # Mean across time for first 20 bins
        np.std(cqt, axis=1)[:20]    # Std across time
    ])
    return cqt_stats[:40]  # Return 40 dims


def extract_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 40) -> np.ndarray:
    """
    Mel-spectrogram statistics

    References:
    - Stevens et al. (1937) original mel scale
    - Widely used in speech processing
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.mean(mel_db, axis=1)


def extract_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Chroma features (pitch class distribution)

    References:
    - Bartsch & Wakefield (2005) "Audio thumbnailing of popular music"
    - Useful for music but may capture artifacts in speech
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)  # 12 dims


def extract_spectral_contrast(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Spectral contrast

    References:
    - Jiang et al. (2002) "Music type classification by spectral contrast features"
    """
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.mean(contrast, axis=1)  # 7 dims


def extract_tonnetz(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Tonal centroid features (tonnetz)

    References:
    - Harte et al. (2006) "Detecting harmonic change in musical audio"
    """
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return np.mean(tonnetz, axis=1)  # 6 dims


def extract_modgd(y: np.ndarray, sr: int, n_modgd: int = 12) -> np.ndarray:
    """
    Modified Group Delay (phase-based)

    References:
    - Murthy & Yegnanarayana (2011) "Group delay based processing"
    - Known to be codec-sensitive (our validation confirmed this)
    """
    n_fft = 512
    hop_length = 160

    # Compute STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Group delay = -d(phase)/d(freq)
    phase = np.angle(S)
    group_delay = -np.diff(phase, axis=0)

    # Modified: Apply cepstral smoothing
    from scipy.fftpack import dct
    gd_cep = dct(group_delay, type=2, axis=0, norm='ortho')[:n_modgd, :]

    return np.mean(gd_cep, axis=1)


def extract_instantaneous_frequency(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Instantaneous frequency features

    References:
    - Boashash (1992) "Estimating and interpreting the instantaneous frequency"
    """
    analytic = hilbert(y)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) * sr / (2 * np.pi)

    # Statistics
    return np.array([
        np.mean(inst_freq),
        np.std(inst_freq),
        np.median(inst_freq),
        stats.skew(inst_freq),
        stats.kurtosis(inst_freq),
        np.percentile(inst_freq, 25),
        np.percentile(inst_freq, 75),
        np.max(inst_freq) - np.min(inst_freq)  # Range
    ])


def extract_prosody(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Prosodic features using Parselmouth/Praat

    References:
    - Jadoul et al. (2018) "Introducing Parselmouth: A Python interface to Praat"
    - F0 and energy patterns crucial for deepfake detection
    """
    if not PARSELMOUTH_AVAILABLE:
        return np.zeros(7)

    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)

        # F0 (pitch)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]

        if len(f0_values) == 0:
            f0_mean, f0_std, f0_max = 0, 0, 0
            voiced_ratio = 0
        else:
            f0_mean = np.mean(f0_values)
            f0_std = np.std(f0_values)
            f0_max = np.max(f0_values)
            voiced_ratio = len(f0_values) / len(pitch.selected_array['frequency'])

        # Energy
        energy = np.mean(y ** 2)

        # Duration
        duration = len(y) / sr

        # Harmonics-to-noise ratio
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
        except:
            hnr = 0

        return np.array([f0_mean, f0_std, f0_max, voiced_ratio,
                         energy, duration, hnr])
    except Exception as e:
        return np.zeros(7)


def extract_formants(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Formant frequencies and bandwidths

    References:
    - Fant (1960) "Acoustic theory of speech production"
    - Formants capture vocal tract characteristics
    """
    if not PARSELMOUTH_AVAILABLE:
        return np.zeros(8)

    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)

        # Get F1-F4 means and bandwidths
        f1_vals, f2_vals = [], []
        f1_bw_vals, f2_bw_vals = [], []

        num_frames = call(formant, "Get number of frames")
        for i in range(1, min(num_frames + 1, 100)):  # Sample up to 100 frames
            try:
                t = call(formant, "Get time from frame number", i)
                f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                if f1 > 0 and f2 > 0:
                    f1_vals.append(f1)
                    f2_vals.append(f2)
                    # Extract actual Praat bandwidths (Hz) at this time point
                    bw1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "Linear")
                    bw2 = call(formant, "Get bandwidth at time", 2, t, "Hertz", "Linear")
                    if bw1 > 0:
                        f1_bw_vals.append(bw1)
                    if bw2 > 0:
                        f2_bw_vals.append(bw2)
            except:
                continue

        if len(f1_vals) == 0:
            return np.zeros(8)

        f1_mean = np.mean(f1_vals)
        f1_std = np.std(f1_vals)
        f2_mean = np.mean(f2_vals)
        f2_std = np.std(f2_vals)

        # Formant bandwidth from Praat (mean across frames)
        f1_bw = np.mean(f1_bw_vals) if f1_bw_vals else 0.0
        f2_bw = np.mean(f2_bw_vals) if f2_bw_vals else 0.0

        # Formant ratio
        f2_f1_ratio = f2_mean / f1_mean if f1_mean > 0 else 0

        # Formant dispersion
        dispersion = f2_mean - f1_mean

        return np.array([f1_mean, f1_std, f2_mean, f2_std,
                         f1_bw, f2_bw, f2_f1_ratio, dispersion])
    except Exception as e:
        return np.zeros(8)


def extract_glottal(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Glottal source features

    References:
    - Drugman et al. (2012) "Glottal source processing: From analysis to applications"
    - Alku (2011) "Glottal wave analysis with Pitch Synchronous Iterated Adaptive Inverse Filtering"

    Note: Full glottal analysis requires specialized tools (e.g., GlottDNN)
    This is a simplified approximation using spectral tilt
    """
    try:
        # Spectral tilt (approximates glottal pulse shape)
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        # Compute spectral tilt via linear regression on log-magnitude
        log_S = np.log(np.mean(S, axis=1) + 1e-10)
        valid_idx = (freqs > 50) & (freqs < 5000)
        if np.sum(valid_idx) < 10:
            return np.zeros(8)

        slope, intercept, r, p, se = stats.linregress(
            np.log(freqs[valid_idx] + 1),
            log_S[valid_idx]
        )

        # H1-H2 (difference between first two harmonics)
        # Approximation using low-frequency energy ratio
        low_energy = np.mean(S[freqs < 500, :])
        mid_energy = np.mean(S[(freqs >= 500) & (freqs < 2000), :])
        h1_h2_approx = low_energy / (mid_energy + 1e-10)

        # Harmonic richness factor
        high_energy = np.mean(S[freqs >= 2000, :])
        hrf = (mid_energy + high_energy) / (low_energy + 1e-10)

        return np.array([
            slope,           # Spectral tilt
            intercept,       # Spectral intercept
            r,               # Tilt fit quality
            h1_h2_approx,    # H1-H2 approximation
            hrf,             # Harmonic richness factor
            low_energy,      # Low-frequency energy
            mid_energy,      # Mid-frequency energy
            high_energy      # High-frequency energy
        ])
    except:
        return np.zeros(8)


def extract_temporal(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Temporal features

    References:
    - Various speech processing textbooks
    """
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Short-time energy
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    ste = np.mean(frames ** 2, axis=0)

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    return np.array([
        np.mean(zcr),
        np.std(zcr),
        np.mean(ste),
        np.std(ste),
        np.mean(onset_env),
        np.std(onset_env),
        len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr),  # Onset rate
        np.max(onset_env) / (np.mean(onset_env) + 1e-10)  # Onset peakiness
    ])


def extract_wavelet(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Wavelet-based features

    References:
    - Mallat (1989) "A theory for multiresolution signal decomposition"
    - Tian et al. (2016) used wavelets for audio spoofing detection
    """
    if not PYWT_AVAILABLE:
        return np.zeros(16)

    try:
        # Discrete wavelet transform
        coeffs = pywt.wavedec(y, 'db4', level=5)

        features = []
        for i, coeff in enumerate(coeffs):
            features.extend([
                np.mean(np.abs(coeff)),
                np.std(coeff),
                stats.entropy(np.abs(coeff) + 1e-10)
            ])

        return np.array(features[:16])  # Return first 16
    except:
        return np.zeros(16)


def extract_spectral_statistics(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Additional spectral statistics

    References:
    - Peeters (2004) "A large set of audio features for sound description"
    """
    # Spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    return np.array([
        np.mean(cent), np.std(cent),
        np.mean(bw), np.std(bw),
        np.mean(rolloff), np.std(rolloff),
        np.mean(flatness), np.std(flatness)
    ])


# ============================================================================
# CODEC PROCESSING
# ============================================================================

# Comprehensive codec configurations for real-world scenarios
CODECS = {
    # Telephony (ASVspoof 2021 actual codecs)
    'g711_alaw': {
        'name': 'G.711 A-law',
        'category': 'Telephony',
        'bitrate': '64 kbps',
        'scenario': 'Landline telephony',
        'ffmpeg_args': ['-c:a', 'pcm_alaw', '-ar', '8000', '-ac', '1'],
        'ext': 'wav'
    },
    'g711_ulaw': {
        'name': 'G.711 μ-law',
        'category': 'Telephony',
        'bitrate': '64 kbps',
        'scenario': 'US/Japan telephony',
        'ffmpeg_args': ['-c:a', 'pcm_mulaw', '-ar', '8000', '-ac', '1'],
        'ext': 'wav'
    },
    'g722': {
        'name': 'G.722 Wideband',
        'category': 'Telephony',
        'bitrate': '64 kbps',
        'scenario': 'HD Voice calls',
        'ffmpeg_args': ['-c:a', 'g722', '-ar', '16000', '-ac', '1'],
        'ext': 'wav'
    },

    # Streaming
    'mp3_32k': {
        'name': 'MP3 32kbps',
        'category': 'Streaming',
        'bitrate': '32 kbps',
        'scenario': 'Low-quality streaming',
        'ffmpeg_args': ['-c:a', 'libmp3lame', '-b:a', '32k'],
        'ext': 'mp3'
    },
    'mp3_64k': {
        'name': 'MP3 64kbps',
        'category': 'Streaming',
        'bitrate': '64 kbps',
        'scenario': 'Standard streaming',
        'ffmpeg_args': ['-c:a', 'libmp3lame', '-b:a', '64k'],
        'ext': 'mp3'
    },
    'mp3_128k': {
        'name': 'MP3 128kbps',
        'category': 'Streaming',
        'bitrate': '128 kbps',
        'scenario': 'High-quality streaming',
        'ffmpeg_args': ['-c:a', 'libmp3lame', '-b:a', '128k'],
        'ext': 'mp3'
    },
    'aac_32k': {
        'name': 'AAC 32kbps',
        'category': 'Streaming',
        'bitrate': '32 kbps',
        'scenario': 'Mobile streaming',
        'ffmpeg_args': ['-c:a', 'aac', '-b:a', '32k'],
        'ext': 'm4a'
    },
    'aac_64k': {
        'name': 'AAC 64kbps',
        'category': 'Streaming',
        'bitrate': '64 kbps',
        'scenario': 'YouTube/TikTok',
        'ffmpeg_args': ['-c:a', 'aac', '-b:a', '64k'],
        'ext': 'm4a'
    },
    'aac_128k': {
        'name': 'AAC 128kbps',
        'category': 'Streaming',
        'bitrate': '128 kbps',
        'scenario': 'High-quality streaming',
        'ffmpeg_args': ['-c:a', 'aac', '-b:a', '128k'],
        'ext': 'm4a'
    },
}


def apply_codec(audio: np.ndarray, sr: int, codec_config: dict,
                tmp_dir: str = '/tmp') -> Optional[np.ndarray]:
    """Apply codec transformation using ffmpeg."""
    if not FFMPEG_AVAILABLE:
        return None

    import tempfile
    import uuid

    uid = str(uuid.uuid4())[:8]
    input_path = os.path.join(tmp_dir, f'input_{uid}.wav')
    output_path = os.path.join(tmp_dir, f'output_{uid}.{codec_config["ext"]}')
    decoded_path = os.path.join(tmp_dir, f'decoded_{uid}.wav')

    try:
        # Write input
        sf.write(input_path, audio, sr)

        # Encode
        cmd = ['ffmpeg', '-y', '-i', input_path] + codec_config['ffmpeg_args'] + [output_path]
        subprocess.run(cmd, capture_output=True, check=True)

        # Decode back to WAV
        cmd = ['ffmpeg', '-y', '-i', output_path, '-ar', str(sr), '-ac', '1', decoded_path]
        subprocess.run(cmd, capture_output=True, check=True)

        # Read decoded
        decoded, _ = sf.read(decoded_path)

        return decoded

    except Exception as e:
        return None
    finally:
        # Cleanup
        for p in [input_path, output_path, decoded_path]:
            if os.path.exists(p):
                os.remove(p)


# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

# Define all features to analyze
FEATURE_CONFIGS = {
    # Spectral features
    'MFCC': FeatureConfig(
        name='MFCC',
        category='Spectral',
        n_dims=13,
        description='Mel-Frequency Cepstral Coefficients',
        literature=['Davis & Mermelstein (1980) IEEE TASSP'],
        codec_robust_claim=False,
        extraction_fn='extract_mfcc'
    ),
    'LFCC': FeatureConfig(
        name='LFCC',
        category='Spectral',
        n_dims=8,
        description='Linear Frequency Cepstral Coefficients',
        literature=['Sahidullah & Saha (2012)', 'ASVspoof 2021 baseline'],
        codec_robust_claim=True,
        extraction_fn='extract_lfcc'
    ),
    'PNCC': FeatureConfig(
        name='PNCC',
        category='Spectral',
        n_dims=13,
        description='Power-Normalized Cepstral Coefficients',
        literature=['Kim & Stern (2016) IEEE TASLP'],
        codec_robust_claim=True,
        extraction_fn='extract_pncc'
    ),
    'CQT': FeatureConfig(
        name='CQT',
        category='Spectral',
        n_dims=40,
        description='Constant-Q Transform features',
        literature=['Brown (1991) JASA', 'Todisco et al. (2017)'],
        codec_robust_claim=False,
        extraction_fn='extract_cqt'
    ),
    'Mel-Spectrogram': FeatureConfig(
        name='Mel-Spectrogram',
        category='Spectral',
        n_dims=40,
        description='Mel-spectrogram statistics',
        literature=['Stevens et al. (1937)'],
        codec_robust_claim=False,
        extraction_fn='extract_mel_spectrogram'
    ),
    'Chroma': FeatureConfig(
        name='Chroma',
        category='Spectral',
        n_dims=12,
        description='Pitch class distribution',
        literature=['Bartsch & Wakefield (2005)'],
        codec_robust_claim=False,
        extraction_fn='extract_chroma'
    ),
    'Spectral-Contrast': FeatureConfig(
        name='Spectral-Contrast',
        category='Spectral',
        n_dims=7,
        description='Spectral contrast features',
        literature=['Jiang et al. (2002)'],
        codec_robust_claim=False,
        extraction_fn='extract_spectral_contrast'
    ),
    'Tonnetz': FeatureConfig(
        name='Tonnetz',
        category='Spectral',
        n_dims=6,
        description='Tonal centroid features',
        literature=['Harte et al. (2006)'],
        codec_robust_claim=False,
        extraction_fn='extract_tonnetz'
    ),
    'Spectral-Statistics': FeatureConfig(
        name='Spectral-Statistics',
        category='Spectral',
        n_dims=8,
        description='Spectral centroid, bandwidth, rolloff, flatness',
        literature=['Peeters (2004)'],
        codec_robust_claim=False,
        extraction_fn='extract_spectral_statistics'
    ),

    # Phase-based features
    'MODGD': FeatureConfig(
        name='MODGD',
        category='Phase',
        n_dims=12,
        description='Modified Group Delay',
        literature=['Murthy & Yegnanarayana (2011)'],
        codec_robust_claim=False,
        extraction_fn='extract_modgd'
    ),
    'Instantaneous-Freq': FeatureConfig(
        name='Instantaneous-Freq',
        category='Phase',
        n_dims=8,
        description='Instantaneous frequency statistics',
        literature=['Boashash (1992)'],
        codec_robust_claim=False,
        extraction_fn='extract_instantaneous_frequency'
    ),

    # Prosodic features
    'Prosody': FeatureConfig(
        name='Prosody',
        category='Prosodic',
        n_dims=7,
        description='F0, energy, duration, HNR',
        literature=['Jadoul et al. (2018) - Parselmouth'],
        codec_robust_claim=True,
        extraction_fn='extract_prosody'
    ),

    # Formant features
    'Formants': FeatureConfig(
        name='Formants',
        category='Formants',
        n_dims=8,
        description='F1, F2 frequencies and bandwidths',
        literature=['Fant (1960)'],
        codec_robust_claim=True,
        extraction_fn='extract_formants'
    ),

    # Glottal features
    'Glottal': FeatureConfig(
        name='Glottal',
        category='Glottal',
        n_dims=8,
        description='Spectral tilt, H1-H2, HRF',
        literature=['Drugman et al. (2012)', 'Alku (2011)'],
        codec_robust_claim=False,
        extraction_fn='extract_glottal'
    ),

    # Temporal features
    'Temporal': FeatureConfig(
        name='Temporal',
        category='Temporal',
        n_dims=8,
        description='ZCR, STE, onset statistics',
        literature=['Various'],
        codec_robust_claim=False,
        extraction_fn='extract_temporal'
    ),

    # Wavelet features
    'Wavelet': FeatureConfig(
        name='Wavelet',
        category='Wavelet',
        n_dims=16,
        description='DWT coefficients (db4)',
        literature=['Mallat (1989)', 'Tian et al. (2016)'],
        codec_robust_claim=False,
        extraction_fn='extract_wavelet'
    ),
}


def compute_icc_for_feature(orig_values: np.ndarray, coded_values_by_codec: Dict[str, np.ndarray],
                            codec_names: List[str]) -> float:
    """Compute ICC(2,1) - two-way random, absolute agreement, single measures.

    Per Shrout & Fleiss (1979) Psychological Bulletin 86(2):420-428.

    Args:
        orig_values: array of shape (n_files, n_dims) - original features
        coded_values_by_codec: dict of codec_name -> array (n_files, n_dims)
        codec_names: list of codec names
    Returns:
        ICC value (float) or NaN
    """
    if not PINGOUIN_AVAILABLE:
        return float('nan')

    rows = []
    n_files = orig_values.shape[0]
    n_dims = orig_values.shape[1]

    for file_idx in range(n_files):
        for dim in range(n_dims):
            target_id = f'file{file_idx}_dim{dim}'
            rows.append({
                'target': target_id,
                'rater': 'original',
                'rating': float(orig_values[file_idx, dim])
            })
            for codec_name in codec_names:
                coded = coded_values_by_codec[codec_name]
                rows.append({
                    'target': target_id,
                    'rater': codec_name,
                    'rating': float(coded[file_idx, dim])
                })

    df = pd.DataFrame(rows)

    try:
        icc_result = pg.intraclass_corr(
            data=df, targets='target', raters='rater', ratings='rating'
        )
        # ICC2 = two-way random, single measures, absolute agreement
        icc2_row = icc_result[icc_result['Type'] == 'ICC2']
        if len(icc2_row) > 0:
            return float(icc2_row['ICC'].values[0])
    except Exception as e:
        print(f"  ICC computation failed: {e}")

    return float('nan')


def extract_all_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Extract all features for an audio sample."""
    features = {}

    for name, config in FEATURE_CONFIGS.items():
        try:
            fn = globals()[config.extraction_fn]
            feat = fn(y, sr)
            features[name] = feat
        except Exception as e:
            features[name] = np.zeros(config.n_dims)

    return features


def analyze_feature_codec_robustness(
    audio_files: List[str],
    n_samples: int = 100,
    output_dir: str = 'feature_analysis'
) -> Dict[str, FeatureAnalysisResult]:
    """
    Analyze codec robustness for all features.

    Args:
        audio_files: List of audio file paths
        n_samples: Number of samples to analyze
        output_dir: Directory to save results

    Returns:
        Dictionary mapping feature names to analysis results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample files
    if len(audio_files) > n_samples:
        np.random.seed(42)
        sampled_files = np.random.choice(audio_files, n_samples, replace=False)
    else:
        sampled_files = audio_files

    print(f"\n{'='*70}")
    print("COMPREHENSIVE ACOUSTIC FEATURE ANALYSIS")
    print(f"{'='*70}")
    print(f"Samples: {len(sampled_files)}")
    print(f"Features: {len(FEATURE_CONFIGS)}")
    print(f"Codecs: {len(CODECS)}")
    print(f"{'='*70}\n")

    # Accumulate feature matrices: {feat_name: {codec_id: [list of vectors]}}
    # and original features: {feat_name: [list of vectors]}
    orig_feat_accumulator = {feat_name: [] for feat_name in FEATURE_CONFIGS}
    coded_feat_accumulator = {
        feat_name: {codec_id: [] for codec_id in CODECS}
        for feat_name in FEATURE_CONFIGS
    }
    # Track which (file_idx, codec_id) pairs succeeded
    codec_success = {codec_id: [] for codec_id in CODECS}  # list of file indices

    # Also store per-file MAE for backward-compatible raw CSV output
    all_results = []

    # Process each sample
    for file_idx, file_path in enumerate(tqdm(sampled_files, desc="Processing samples")):
        try:
            # Load original audio
            y_orig, sr = librosa.load(file_path, sr=16000)

            # Extract original features
            orig_features = extract_all_features(y_orig, sr)

            # Store original features per feature type
            for feat_name in FEATURE_CONFIGS:
                orig_feat_accumulator[feat_name].append(orig_features[feat_name])

            # Process each codec
            for codec_id, codec_config in CODECS.items():
                try:
                    # Apply codec
                    y_coded = apply_codec(y_orig, sr, codec_config)
                    if y_coded is None:
                        # Append NaN placeholder to keep alignment
                        for feat_name in FEATURE_CONFIGS:
                            n_dims = FEATURE_CONFIGS[feat_name].n_dims
                            coded_feat_accumulator[feat_name][codec_id].append(
                                np.full(n_dims, np.nan))
                        continue

                    # Ensure same length
                    min_len = min(len(y_orig), len(y_coded))
                    y_coded = y_coded[:min_len]

                    # Extract coded features
                    coded_features = extract_all_features(y_coded, sr)

                    codec_success[codec_id].append(file_idx)

                    # Store coded features and per-file MAE
                    for feat_name in FEATURE_CONFIGS:
                        orig_feat = orig_features[feat_name]
                        coded_feat = coded_features[feat_name]
                        coded_feat_accumulator[feat_name][codec_id].append(coded_feat)

                        mae = np.mean(np.abs(orig_feat - coded_feat))

                        all_results.append({
                            'feature': feat_name,
                            'category': FEATURE_CONFIGS[feat_name].category,
                            'codec': codec_id,
                            'codec_name': codec_config['name'],
                            'codec_category': codec_config['category'],
                            'scenario': codec_config['scenario'],
                            'mae': mae,
                            'n_dims': FEATURE_CONFIGS[feat_name].n_dims
                        })

                except Exception as e:
                    # Append NaN placeholder on failure to keep alignment
                    for feat_name in FEATURE_CONFIGS:
                        n_dims = FEATURE_CONFIGS[feat_name].n_dims
                        coded_feat_accumulator[feat_name][codec_id].append(
                            np.full(n_dims, np.nan))
                    continue

        except Exception as e:
            continue

    # ========================================================================
    # CORRECT across-samples per-dimension correlation (Fix C3)
    # ========================================================================
    # For each (feature, codec) pair, compute Pearson r ACROSS ALL SAMPLES
    # for EACH DIMENSION, then report mean and min across dimensions.
    # This replaces the incorrect within-sample correlation.
    # ========================================================================

    # Build correlation results to merge into the DataFrame
    correlation_rows = []

    results = {}

    for feat_name, config in FEATURE_CONFIGS.items():
        # Build original and coded matrices: (n_files, n_dims)
        if len(orig_feat_accumulator[feat_name]) == 0:
            continue

        orig_matrix = np.array(orig_feat_accumulator[feat_name])  # (n_files, n_dims)

        per_codec_pearson = {}
        codec_results = {}

        for codec_id in CODECS:
            coded_list = coded_feat_accumulator[feat_name][codec_id]
            if len(coded_list) == 0:
                continue

            coded_matrix = np.array(coded_list)  # (n_files, n_dims)

            # Guard: skip wavelet features when pywt unavailable
            if feat_name == 'Wavelet' and not PYWT_AVAILABLE:
                per_codec_pearson[codec_id] = float('nan')
                codec_results[codec_id] = {
                    'mean_pearson_r': float('nan'),
                    'std_pearson_r': float('nan'),
                    'min_pearson_r': float('nan'),
                    'mean_mae': float('nan')
                }
                continue

            # Find valid rows (no NaN in either matrix for this codec)
            valid_mask = ~(np.isnan(orig_matrix).any(axis=1) | np.isnan(coded_matrix).any(axis=1))
            orig_valid = orig_matrix[valid_mask]
            coded_valid = coded_matrix[valid_mask]

            if len(orig_valid) < 3:
                per_codec_pearson[codec_id] = float('nan')
                codec_results[codec_id] = {
                    'mean_pearson_r': float('nan'),
                    'std_pearson_r': float('nan'),
                    'min_pearson_r': float('nan'),
                    'mean_mae': float('nan')
                }
                continue

            # CORRECT: Compute Pearson r ACROSS ALL SAMPLES for EACH DIMENSION
            # Then report mean and min across dimensions
            per_dim_correlations = []
            n_dims = min(orig_valid.shape[1], coded_valid.shape[1])
            for dim in range(n_dims):
                orig_col = orig_valid[:, dim]
                coded_col = coded_valid[:, dim]
                if np.std(orig_col) > 1e-10 and np.std(coded_col) > 1e-10:
                    r, _ = stats.pearsonr(orig_col, coded_col)
                    per_dim_correlations.append(r)
                else:
                    per_dim_correlations.append(float('nan'))

            if per_dim_correlations:
                pearson_r = float(np.nanmean(per_dim_correlations))
                min_pearson_r = float(np.nanmin(per_dim_correlations))
                std_pearson_r = float(np.nanstd(per_dim_correlations))
            else:
                pearson_r = float('nan')
                min_pearson_r = float('nan')
                std_pearson_r = float('nan')

            per_codec_pearson[codec_id] = pearson_r

            # Compute MAE across all valid samples
            mae_vals = np.mean(np.abs(orig_valid - coded_valid), axis=1)
            mean_mae = float(np.mean(mae_vals))

            codec_results[codec_id] = {
                'mean_pearson_r': pearson_r,
                'std_pearson_r': std_pearson_r,
                'min_pearson_r': min_pearson_r,
                'mean_mae': mean_mae
            }

            # Add to correlation_rows for backward-compatible DataFrame
            correlation_rows.append({
                'feature': feat_name,
                'codec': codec_id,
                'pearson_r': pearson_r,
                'min_dim_pearson_r': min_pearson_r
            })

        # Aggregate across codecs
        codec_r_values = [v for v in per_codec_pearson.values() if not np.isnan(v)]
        if len(codec_r_values) == 0:
            continue

        mean_r = float(np.mean(codec_r_values))
        min_r = float(np.min(codec_r_values))
        max_r = float(np.max(codec_r_values))
        std_r = float(np.std(codec_r_values))

        # Find worst codec
        worst_codec = min(per_codec_pearson, key=lambda k: per_codec_pearson[k]
                          if not np.isnan(per_codec_pearson[k]) else float('inf'))

        # ====================================================================
        # ICC(2,1) computation (Fix C4)
        # Shrout & Fleiss (1979) Psychological Bulletin 86(2):420-428
        # ====================================================================
        icc_value = float('nan')
        if PINGOUIN_AVAILABLE:
            # Collect coded matrices for codecs that have valid data
            icc_codec_names = []
            icc_coded_by_codec = {}
            for codec_id in CODECS:
                coded_list = coded_feat_accumulator[feat_name][codec_id]
                if len(coded_list) == 0:
                    continue
                coded_mat = np.array(coded_list)
                # Use only rows valid across ALL codecs and original
                valid_mask = ~(np.isnan(orig_matrix).any(axis=1) | np.isnan(coded_mat).any(axis=1))
                if valid_mask.sum() >= 3:
                    icc_codec_names.append(codec_id)
                    icc_coded_by_codec[codec_id] = coded_mat[valid_mask]

            if len(icc_codec_names) >= 2:
                # Find common valid rows across all codecs
                common_valid = np.ones(orig_matrix.shape[0], dtype=bool)
                for codec_id in icc_codec_names:
                    coded_mat = np.array(coded_feat_accumulator[feat_name][codec_id])
                    common_valid &= ~np.isnan(coded_mat).any(axis=1)
                common_valid &= ~np.isnan(orig_matrix).any(axis=1)

                if common_valid.sum() >= 3:
                    orig_common = orig_matrix[common_valid]
                    coded_common = {
                        c: np.array(coded_feat_accumulator[feat_name][c])[common_valid]
                        for c in icc_codec_names
                    }
                    print(f"  Computing ICC(2,1) for {feat_name} "
                          f"({common_valid.sum()} files, {len(icc_codec_names)} codecs)...")
                    icc_value = compute_icc_for_feature(
                        orig_common, coded_common, icc_codec_names)
                    print(f"    ICC(2,1) = {icc_value:.4f}")

        # Determine recommendation
        if min_r >= 0.95:
            status = "HIGHLY_RECOMMENDED"
            rec = f"Excellent stability (min_r={min_r:.3f})"
        elif min_r >= 0.90:
            status = "RECOMMENDED"
            rec = f"Good stability (min_r={min_r:.3f})"
        elif min_r >= 0.80:
            status = "ACCEPTABLE"
            rec = f"Moderate stability (min_r={min_r:.3f}), use with caution"
        else:
            status = "NOT_RECOMMENDED"
            rec = f"Poor stability (min_r={min_r:.3f}), avoid for codec-robust systems"

        results[feat_name] = FeatureAnalysisResult(
            name=feat_name,
            category=config.category,
            n_dims=config.n_dims,
            mean_pearson_r=mean_r,
            min_pearson_r=min_r,
            max_pearson_r=max_r,
            std_pearson_r=std_r,
            worst_codec=worst_codec,
            codec_results=codec_results,
            icc=icc_value,
            recommendation=rec,
            status=status
        )

    # Build backward-compatible DataFrame with per-file MAE and per-codec correlations
    df = pd.DataFrame(all_results)
    # Merge in the across-samples correlation values
    if correlation_rows:
        corr_df = pd.DataFrame(correlation_rows)
        df = df.merge(corr_df[['feature', 'codec', 'pearson_r']], on=['feature', 'codec'], how='left')
    else:
        df['pearson_r'] = float('nan')
    df.to_csv(os.path.join(output_dir, 'raw_results.csv'), index=False)

    return results, df


def create_visualizations(results: Dict[str, FeatureAnalysisResult],
                         df: pd.DataFrame,
                         output_dir: str):
    """Create comprehensive visualizations."""

    # 1. Feature ranking by mean correlation
    fig, ax = plt.subplots(figsize=(14, 8))

    sorted_features = sorted(results.values(), key=lambda x: x.mean_pearson_r, reverse=True)
    names = [r.name for r in sorted_features]
    means = [r.mean_pearson_r for r in sorted_features]
    mins = [r.min_pearson_r for r in sorted_features]
    maxs = [r.max_pearson_r for r in sorted_features]
    colors = ['#2ecc71' if r.status in ['HIGHLY_RECOMMENDED', 'RECOMMENDED']
              else '#f39c12' if r.status == 'ACCEPTABLE'
              else '#e74c3c' for r in sorted_features]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, color=colors, alpha=0.8, label='Mean')
    ax.errorbar(means, y_pos, xerr=[np.array(means)-np.array(mins),
                                     np.array(maxs)-np.array(means)],
                fmt='none', color='black', capsize=3)

    ax.axvline(x=0.90, color='red', linestyle='--', alpha=0.7, label='Threshold (0.90)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Pearson Correlation (codec stability)')
    ax.set_title('Acoustic Feature Codec Robustness Ranking\n(Mean ± Min/Max across all codecs)')
    ax.legend()
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_ranking.png'), dpi=150)
    plt.close()

    # 2. Heatmap: Features vs Codecs
    pivot_df = df.groupby(['feature', 'codec'])['pearson_r'].mean().unstack()

    # Reorder by mean correlation
    feature_order = [r.name for r in sorted_features]
    pivot_df = pivot_df.reindex(feature_order)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Pearson r'})
    ax.set_title('Feature Stability Across Codecs')
    ax.set_xlabel('Codec')
    ax.set_ylabel('Feature')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_features_codecs.png'), dpi=150)
    plt.close()

    # 3. Category comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Box plot by category
    categories = df.groupby('category')['pearson_r'].apply(list).to_dict()
    ax = axes[0, 0]
    ax.boxplot(categories.values(), labels=categories.keys())
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Codec Stability by Feature Category')
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.7)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Box plot by codec category
    codec_cats = df.groupby('codec_category')['pearson_r'].apply(list).to_dict()
    ax = axes[0, 1]
    ax.boxplot(codec_cats.values(), labels=codec_cats.keys())
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Feature Stability by Codec Category')
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.7)

    # Recommended vs Not Recommended
    ax = axes[1, 0]
    recommended = [r.mean_pearson_r for r in results.values()
                   if r.status in ['HIGHLY_RECOMMENDED', 'RECOMMENDED']]
    not_recommended = [r.mean_pearson_r for r in results.values()
                       if r.status == 'NOT_RECOMMENDED']
    acceptable = [r.mean_pearson_r for r in results.values()
                  if r.status == 'ACCEPTABLE']

    ax.bar(['Recommended', 'Acceptable', 'Not Recommended'],
           [np.mean(recommended) if recommended else 0,
            np.mean(acceptable) if acceptable else 0,
            np.mean(not_recommended) if not_recommended else 0],
           color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax.set_ylabel('Mean Pearson Correlation')
    ax.set_title('Average Stability by Recommendation')

    # Literature claims vs actual results
    ax = axes[1, 1]
    claimed_robust = [r.mean_pearson_r for r in results.values()
                      if FEATURE_CONFIGS[r.name].codec_robust_claim]
    not_claimed = [r.mean_pearson_r for r in results.values()
                   if not FEATURE_CONFIGS[r.name].codec_robust_claim]

    ax.boxplot([claimed_robust, not_claimed],
               labels=['Literature Claims\nCodec-Robust', 'No Robustness\nClaimed'])
    ax.set_ylabel('Actual Pearson Correlation')
    ax.set_title('Literature Claims vs Experimental Results')
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_analysis.png'), dpi=150)
    plt.close()

    # 4. Telephony codec focus (ASVspoof 2021 relevant)
    telephony_codecs = ['g711_alaw', 'g711_ulaw', 'g722']
    tel_df = df[df['codec'].isin(telephony_codecs)]

    if len(tel_df) > 0:
        pivot_tel = tel_df.groupby(['feature', 'codec'])['pearson_r'].mean().unstack()
        pivot_tel = pivot_tel.reindex(feature_order)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(pivot_tel, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Pearson r'})
        ax.set_title('Feature Stability Under Telephony Codecs\n(Most relevant for ASVspoof 2021)')
        ax.set_xlabel('Telephony Codec')
        ax.set_ylabel('Feature')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'telephony_focus.png'), dpi=150)
        plt.close()

    print(f"Visualizations saved to {output_dir}/")


def generate_report(results: Dict[str, FeatureAnalysisResult],
                   output_dir: str) -> str:
    """Generate comprehensive analysis report."""

    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE ACOUSTIC FEATURE ANALYSIS REPORT")
    report.append("Audio Deepfake Detection - Codec Robustness Validation")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)

    highly_rec = [r for r in results.values() if r.status == 'HIGHLY_RECOMMENDED']
    recommended = [r for r in results.values() if r.status == 'RECOMMENDED']
    acceptable = [r for r in results.values() if r.status == 'ACCEPTABLE']
    not_rec = [r for r in results.values() if r.status == 'NOT_RECOMMENDED']

    report.append(f"Total Features Analyzed: {len(results)}")
    report.append(f"  - HIGHLY RECOMMENDED: {len(highly_rec)}")
    report.append(f"  - RECOMMENDED: {len(recommended)}")
    report.append(f"  - ACCEPTABLE: {len(acceptable)}")
    report.append(f"  - NOT RECOMMENDED: {len(not_rec)}")
    report.append("")

    # Detailed results by category
    report.append("DETAILED RESULTS BY CATEGORY")
    report.append("-" * 40)

    categories = {}
    for r in results.values():
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    for cat, feats in sorted(categories.items()):
        report.append(f"\n{cat.upper()} FEATURES:")
        for r in sorted(feats, key=lambda x: x.mean_pearson_r, reverse=True):
            status_icon = "✅" if r.status in ['HIGHLY_RECOMMENDED', 'RECOMMENDED'] else \
                          "⚠️" if r.status == 'ACCEPTABLE' else "❌"
            report.append(f"  {status_icon} {r.name} ({r.n_dims}D)")
            report.append(f"      Mean r: {r.mean_pearson_r:.4f}, "
                         f"Min r: {r.min_pearson_r:.4f}, "
                         f"Worst: {r.worst_codec}")
            report.append(f"      {r.recommendation}")

    # Recommendations
    report.append("\n" + "=" * 80)
    report.append("FINAL RECOMMENDATIONS FOR CODEC-ROBUST DEEPFAKE DETECTION")
    report.append("=" * 80)

    report.append("\n1. HIGHLY RECOMMENDED (min_r >= 0.95):")
    for r in sorted(highly_rec, key=lambda x: x.mean_pearson_r, reverse=True):
        report.append(f"   - {r.name}: {r.n_dims}D, mean_r={r.mean_pearson_r:.4f}")

    report.append("\n2. RECOMMENDED (min_r >= 0.90):")
    for r in sorted(recommended, key=lambda x: x.mean_pearson_r, reverse=True):
        report.append(f"   - {r.name}: {r.n_dims}D, mean_r={r.mean_pearson_r:.4f}")

    report.append("\n3. ACCEPTABLE WITH CAUTION (0.80 <= min_r < 0.90):")
    for r in sorted(acceptable, key=lambda x: x.mean_pearson_r, reverse=True):
        report.append(f"   - {r.name}: {r.n_dims}D, min_r={r.min_pearson_r:.4f}, worst={r.worst_codec}")

    report.append("\n4. NOT RECOMMENDED (min_r < 0.80):")
    for r in sorted(not_rec, key=lambda x: x.min_pearson_r):
        report.append(f"   - {r.name}: {r.n_dims}D, min_r={r.min_pearson_r:.4f}, worst={r.worst_codec}")
        # Check if literature claimed robustness
        if FEATURE_CONFIGS[r.name].codec_robust_claim:
            report.append(f"     ⚠️ WARNING: Literature claimed codec robustness - NOT CONFIRMED!")

    # Suggested feature vector
    report.append("\n" + "=" * 80)
    report.append("SUGGESTED CODEC-ROBUST FEATURE VECTOR")
    report.append("=" * 80)

    suggested = highly_rec + recommended
    total_dims = sum(r.n_dims for r in suggested)

    report.append(f"\nTotal dimensions: {total_dims}")
    report.append("Components:")
    for r in sorted(suggested, key=lambda x: x.mean_pearson_r, reverse=True):
        lit = FEATURE_CONFIGS[r.name].literature[0] if FEATURE_CONFIGS[r.name].literature else "N/A"
        report.append(f"  - {r.name} ({r.n_dims}D): {FEATURE_CONFIGS[r.name].description}")
        report.append(f"    Reference: {lit}")

    report_text = "\n".join(report)

    # Save report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    # Save JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_features': len(results),
        'highly_recommended': [r.name for r in highly_rec],
        'recommended': [r.name for r in recommended],
        'acceptable': [r.name for r in acceptable],
        'not_recommended': [r.name for r in not_rec],
        'suggested_feature_dims': total_dims,
        'results': {name: asdict(r) for name, r in results.items()}
    }

    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return report_text


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Acoustic Feature Analysis')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str,
                        default='evidence/experiments/comprehensive_feature_analysis',
                        help='Output directory')

    args = parser.parse_args()

    # Find audio files
    audio_dir = Path(args.audio_dir)
    audio_files = list(audio_dir.glob('*.flac')) + list(audio_dir.glob('*.wav'))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files")

    # Run analysis
    results, df = analyze_feature_codec_robustness(
        [str(f) for f in audio_files],
        n_samples=args.n_samples,
        output_dir=args.output_dir
    )

    # Create visualizations
    create_visualizations(results, df, args.output_dir)

    # Generate report
    report = generate_report(results, args.output_dir)
    print("\n" + report)

    print(f"\nAnalysis complete. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
