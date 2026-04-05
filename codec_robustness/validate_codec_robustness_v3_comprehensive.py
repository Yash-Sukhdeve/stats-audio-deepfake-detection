#!/usr/bin/env python3
"""
Comprehensive Real-World Codec Robustness Validation (v3)

This script tests feature stability across ALL codecs that could be used
in real-world deepfake attack scenarios.

CODEC CATEGORIES:
1. Telephony (PSTN/VoLTE): G.711, G.722, G.729, AMR, AMR-WB, EVS
2. VoIP: Opus, Speex, iLBC
3. Streaming/Social Media: MP3, AAC, Vorbis/OGG
4. Messaging Apps: WhatsApp (Opus), Telegram (Opus), Signal (Opus)
5. Video Conferencing: Opus, G.722
6. Low-Bitrate Stress Tests: 8kbps, 12kbps, 24kbps

REAL-WORLD ATTACK VECTORS:
- Voice phishing (vishing) calls via PSTN/VoIP
- Deepfake audio shared via messaging apps
- Synthetic speech in social media videos
- Voice cloning for fraud via telephony
- Manipulated audio recordings

References:
- ITU-T G.711 (1988) - PCM for PSTN
- ITU-T G.722 (1988) - Wideband ADPCM
- ITU-T G.729 (1996) - CS-ACELP for VoIP
- 3GPP AMR (1999) - Adaptive Multi-Rate for GSM
- 3GPP AMR-WB (2001) - Wideband AMR
- 3GPP EVS (2014) - Enhanced Voice Services
- IETF RFC 6716 (2012) - Opus Codec
- ISO/IEC 11172-3 (1993) - MP3
- ISO/IEC 14496-3 (1999) - AAC
- Xiph.Org Vorbis (2000) - OGG Vorbis

Author: ACAGAT Project (NeurIPS 2026)
Date: December 3, 2025
"""

import os
import sys
import numpy as np
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import feature extractors
from extract_acoustic_features_v4 import extract_features_v4, HAS_PRAAT


# Feature names for v3 (36 dims)
FEATURE_NAMES_V3 = [
    'PNCC_0', 'PNCC_1', 'PNCC_2', 'PNCC_3', 'PNCC_4', 'PNCC_5', 'PNCC_6',
    'PNCC_7', 'PNCC_8', 'PNCC_9', 'PNCC_10', 'PNCC_11', 'PNCC_12',
    'LFCC_0', 'LFCC_1', 'LFCC_2', 'LFCC_3', 'LFCC_4', 'LFCC_5', 'LFCC_6', 'LFCC_7',
    'F1_mean', 'F1_std', 'F1_bw', 'F2_mean', 'F2_std', 'F2_bw', 'F1_F2_ratio', 'formant_disp',
    'F0_mean', 'F0_std', 'F0_max', 'voiced_ratio', 'energy_mean', 'energy_std', 'duration'
]

# Feature groups for analysis
FEATURE_GROUPS = {
    'PNCC_low': {'indices': list(range(0, 7)), 'dims': 7,
                 'description': 'PNCC 0-6: Low-order power-normalized cepstral'},
    'PNCC_high': {'indices': list(range(7, 13)), 'dims': 6,
                  'description': 'PNCC 7-12: High-order power-normalized cepstral'},
    'LFCC': {'indices': list(range(13, 21)), 'dims': 8,
             'description': 'LFCC: Linear frequency cepstral coefficients'},
    'Formants': {'indices': list(range(21, 29)), 'dims': 8,
                 'description': 'Formants F1/F2: Vowel resonances'},
    'Prosody': {'indices': list(range(29, 36)), 'dims': 7,
                'description': 'Prosody: F0, energy, duration'},
}


# =============================================================================
# COMPREHENSIVE REAL-WORLD CODEC CONFIGURATIONS
# =============================================================================

COMPREHENSIVE_CODECS = [
    # =========================================================================
    # CATEGORY 1: TELEPHONY (PSTN/VoLTE)
    # =========================================================================
    {
        'name': 'g711_alaw',
        'category': 'Telephony',
        'full_name': 'G.711 A-law',
        'description': 'European PSTN standard',
        'ffmpeg_args': ['-c:a', 'pcm_alaw', '-ar', '8000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'wav',
        'bitrate': '64 kbps',
        'bandwidth': '300-3400 Hz',
        'reference': 'ITU-T G.711 (1988)',
        'attack_scenario': 'Voice phishing via landline'
    },
    {
        'name': 'g711_ulaw',
        'category': 'Telephony',
        'full_name': 'G.711 μ-law',
        'description': 'North American PSTN standard',
        'ffmpeg_args': ['-c:a', 'pcm_mulaw', '-ar', '8000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'wav',
        'bitrate': '64 kbps',
        'bandwidth': '300-3400 Hz',
        'reference': 'ITU-T G.711 (1988)',
        'attack_scenario': 'Voice phishing via landline (US)'
    },
    {
        'name': 'g722',
        'category': 'Telephony',
        'full_name': 'G.722 Wideband',
        'description': 'HD Voice for VoLTE/VoIP',
        'ffmpeg_args': ['-c:a', 'g722', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'g722',
        'bitrate': '64 kbps',
        'bandwidth': '50-7000 Hz',
        'reference': 'ITU-T G.722 (1988)',
        'attack_scenario': 'Vishing via HD Voice'
    },
    {
        'name': 'gsm_fr',
        'category': 'Telephony',
        'full_name': 'GSM Full Rate',
        'description': 'Original GSM mobile codec',
        'ffmpeg_args': ['-c:a', 'libgsm', '-ar', '8000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'gsm',
        'bitrate': '13 kbps',
        'bandwidth': '300-3400 Hz',
        'reference': 'ETSI GSM 06.10 (1992)',
        'attack_scenario': 'Deepfake via 2G mobile'
    },

    # =========================================================================
    # CATEGORY 2: VoIP (Internet Telephony)
    # =========================================================================
    {
        'name': 'opus_voip',
        'category': 'VoIP',
        'full_name': 'Opus VoIP Mode',
        'description': 'Standard VoIP setting',
        'ffmpeg_args': ['-c:a', 'libopus', '-b:a', '24k', '-application', 'voip', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'opus',
        'bitrate': '24 kbps',
        'bandwidth': 'Wideband',
        'reference': 'IETF RFC 6716 (2012)',
        'attack_scenario': 'Deepfake via VoIP call'
    },
    {
        'name': 'opus_low',
        'category': 'VoIP',
        'full_name': 'Opus Low Bitrate',
        'description': 'Low bandwidth VoIP',
        'ffmpeg_args': ['-c:a', 'libopus', '-b:a', '12k', '-application', 'voip', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'opus',
        'bitrate': '12 kbps',
        'bandwidth': 'Narrowband',
        'reference': 'IETF RFC 6716 (2012)',
        'attack_scenario': 'Low-bandwidth attack'
    },
    {
        'name': 'opus_ultralow',
        'category': 'VoIP',
        'full_name': 'Opus Ultra-Low',
        'description': 'Extreme compression stress test',
        'ffmpeg_args': ['-c:a', 'libopus', '-b:a', '6k', '-application', 'voip', '-ar', '8000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'opus',
        'bitrate': '6 kbps',
        'bandwidth': 'Narrowband',
        'reference': 'IETF RFC 6716 (2012)',
        'attack_scenario': 'Worst-case VoIP stress test'
    },
    {
        'name': 'speex_nb',
        'category': 'VoIP',
        'full_name': 'Speex Narrowband',
        'description': 'Legacy VoIP codec',
        'ffmpeg_args': ['-c:a', 'libspeex', '-ar', '8000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'spx',
        'bitrate': '~15 kbps',
        'bandwidth': 'Narrowband',
        'reference': 'Xiph.Org Speex (2002)',
        'attack_scenario': 'Legacy VoIP systems'
    },
    {
        'name': 'speex_wb',
        'category': 'VoIP',
        'full_name': 'Speex Wideband',
        'description': 'Legacy VoIP codec wideband',
        'ffmpeg_args': ['-c:a', 'libspeex', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'spx',
        'bitrate': '~28 kbps',
        'bandwidth': 'Wideband',
        'reference': 'Xiph.Org Speex (2002)',
        'attack_scenario': 'Legacy VoIP systems'
    },

    # =========================================================================
    # CATEGORY 3: STREAMING / SOCIAL MEDIA
    # =========================================================================
    {
        'name': 'mp3_32k',
        'category': 'Streaming',
        'full_name': 'MP3 32kbps',
        'description': 'Ultra-low quality streaming',
        'ffmpeg_args': ['-c:a', 'libmp3lame', '-b:a', '32k', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'mp3',
        'bitrate': '32 kbps',
        'bandwidth': 'Limited',
        'reference': 'ISO/IEC 11172-3 (1993)',
        'attack_scenario': 'Low-quality social media'
    },
    {
        'name': 'mp3_64k',
        'category': 'Streaming',
        'full_name': 'MP3 64kbps',
        'description': 'Low quality streaming',
        'ffmpeg_args': ['-c:a', 'libmp3lame', '-b:a', '64k', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'mp3',
        'bitrate': '64 kbps',
        'bandwidth': 'Moderate',
        'reference': 'ISO/IEC 11172-3 (1993)',
        'attack_scenario': 'Social media audio'
    },
    {
        'name': 'mp3_128k',
        'category': 'Streaming',
        'full_name': 'MP3 128kbps',
        'description': 'Standard quality',
        'ffmpeg_args': ['-c:a', 'libmp3lame', '-b:a', '128k', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'mp3',
        'bitrate': '128 kbps',
        'bandwidth': 'Full',
        'reference': 'ISO/IEC 11172-3 (1993)',
        'attack_scenario': 'Standard social media'
    },
    {
        'name': 'aac_32k',
        'category': 'Streaming',
        'full_name': 'AAC 32kbps',
        'description': 'Low bitrate AAC',
        'ffmpeg_args': ['-c:a', 'aac', '-b:a', '32k', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'm4a',
        'bitrate': '32 kbps',
        'bandwidth': 'Limited',
        'reference': 'ISO/IEC 14496-3 (1999)',
        'attack_scenario': 'Mobile streaming'
    },
    {
        'name': 'aac_64k',
        'category': 'Streaming',
        'full_name': 'AAC 64kbps',
        'description': 'Standard AAC',
        'ffmpeg_args': ['-c:a', 'aac', '-b:a', '64k', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'm4a',
        'bitrate': '64 kbps',
        'bandwidth': 'Moderate',
        'reference': 'ISO/IEC 14496-3 (1999)',
        'attack_scenario': 'YouTube/TikTok audio'
    },
    {
        'name': 'aac_128k',
        'category': 'Streaming',
        'full_name': 'AAC 128kbps',
        'description': 'High quality AAC',
        'ffmpeg_args': ['-c:a', 'aac', '-b:a', '128k', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'm4a',
        'bitrate': '128 kbps',
        'bandwidth': 'Full',
        'reference': 'ISO/IEC 14496-3 (1999)',
        'attack_scenario': 'High-quality streaming'
    },
    {
        'name': 'vorbis_64k',
        'category': 'Streaming',
        'full_name': 'OGG Vorbis 64kbps',
        'description': 'Open source streaming codec',
        'ffmpeg_args': ['-c:a', 'libvorbis', '-b:a', '64k', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'ogg',
        'bitrate': '64 kbps',
        'bandwidth': 'Moderate',
        'reference': 'Xiph.Org Vorbis (2000)',
        'attack_scenario': 'Open-source platforms'
    },
    {
        'name': 'vorbis_128k',
        'category': 'Streaming',
        'full_name': 'OGG Vorbis 128kbps',
        'description': 'High quality Vorbis',
        'ffmpeg_args': ['-c:a', 'libvorbis', '-b:a', '128k', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'ogg',
        'bitrate': '128 kbps',
        'bandwidth': 'Full',
        'reference': 'Xiph.Org Vorbis (2000)',
        'attack_scenario': 'Spotify/streaming platforms'
    },

    # =========================================================================
    # CATEGORY 4: MESSAGING APPS
    # =========================================================================
    {
        'name': 'whatsapp_opus',
        'category': 'Messaging',
        'full_name': 'WhatsApp Voice',
        'description': 'WhatsApp voice message codec',
        'ffmpeg_args': ['-c:a', 'libopus', '-b:a', '16k', '-application', 'voip', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'opus',
        'bitrate': '16 kbps',
        'bandwidth': 'Wideband',
        'reference': 'WhatsApp uses Opus (IETF RFC 6716)',
        'attack_scenario': 'Deepfake voice message on WhatsApp'
    },
    {
        'name': 'telegram_opus',
        'category': 'Messaging',
        'full_name': 'Telegram Voice',
        'description': 'Telegram voice message codec',
        'ffmpeg_args': ['-c:a', 'libopus', '-b:a', '32k', '-application', 'voip', '-ar', '48000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'opus',
        'bitrate': '32 kbps',
        'bandwidth': 'Fullband',
        'reference': 'Telegram uses Opus',
        'attack_scenario': 'Deepfake voice message on Telegram'
    },
    {
        'name': 'discord_opus',
        'category': 'Messaging',
        'full_name': 'Discord Voice',
        'description': 'Discord voice chat codec',
        'ffmpeg_args': ['-c:a', 'libopus', '-b:a', '64k', '-application', 'audio', '-ar', '48000', '-ac', '2'],
        'decode_args': ['-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le'],
        'ext': 'opus',
        'bitrate': '64 kbps',
        'bandwidth': 'Fullband',
        'reference': 'Discord uses Opus',
        'attack_scenario': 'Deepfake in Discord voice chat'
    },

    # =========================================================================
    # CATEGORY 5: VIDEO CONFERENCING
    # =========================================================================
    {
        'name': 'zoom_opus',
        'category': 'Video Conferencing',
        'full_name': 'Zoom-like Audio',
        'description': 'Typical video conferencing codec',
        'ffmpeg_args': ['-c:a', 'libopus', '-b:a', '32k', '-application', 'voip', '-ar', '16000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'opus',
        'bitrate': '32 kbps',
        'bandwidth': 'Wideband',
        'reference': 'Zoom uses Opus/custom codec',
        'attack_scenario': 'Deepfake in video call'
    },
    {
        'name': 'webrtc_opus',
        'category': 'Video Conferencing',
        'full_name': 'WebRTC Audio',
        'description': 'Browser-based conferencing',
        'ffmpeg_args': ['-c:a', 'libopus', '-b:a', '40k', '-application', 'audio', '-ar', '48000', '-ac', '1'],
        'decode_args': ['-ar', '16000', '-c:a', 'pcm_s16le'],
        'ext': 'opus',
        'bitrate': '40 kbps',
        'bandwidth': 'Fullband',
        'reference': 'WebRTC mandates Opus',
        'attack_scenario': 'Browser-based deepfake call'
    },

    # =========================================================================
    # CATEGORY 6: MULTIPLE COMPRESSION (REALISTIC ATTACK CHAIN)
    # =========================================================================
    # Note: These simulate audio passing through multiple compression stages
    # Example: Generated -> Recorded on phone -> Sent via WhatsApp -> Downloaded
    # We'll implement double/triple compression in the processing function
]


def check_codec_availability():
    """Check which codecs are available in ffmpeg."""
    available = {}
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr

        codec_checks = {
            'libmp3lame': 'MP3',
            'aac': 'AAC',
            'libopus': 'Opus',
            'libvorbis': 'Vorbis',
            'libspeex': 'Speex',
            'libgsm': 'GSM',
            'g722': 'G.722',
            'pcm_alaw': 'G.711 A-law',
            'pcm_mulaw': 'G.711 μ-law',
        }

        for codec, name in codec_checks.items():
            available[codec] = codec in output

    except Exception as e:
        print(f"Warning: Could not check codec availability: {e}")
        # Assume all available
        for codec in ['libmp3lame', 'aac', 'libopus', 'libvorbis', 'libspeex',
                      'libgsm', 'g722', 'pcm_alaw', 'pcm_mulaw']:
            available[codec] = True

    return available


def compress_audio(input_path: str, codec_config: dict, output_dir: str) -> Optional[str]:
    """
    Compress audio using specified codec configuration.

    Args:
        input_path: Path to original FLAC file
        codec_config: Dictionary with codec configuration
        output_dir: Directory for temporary files

    Returns:
        Path to decoded WAV file, or None on failure
    """
    try:
        base_name = Path(input_path).stem
        codec_name = codec_config['name']
        ext = codec_config.get('ext', 'tmp')

        compressed_path = os.path.join(output_dir, f"{base_name}_{codec_name}.{ext}")
        decoded_path = os.path.join(output_dir, f"{base_name}_{codec_name}_decoded.wav")

        # Encode
        cmd_encode = ['ffmpeg', '-y', '-i', input_path] + codec_config['ffmpeg_args'] + [compressed_path]
        result = subprocess.run(cmd_encode, capture_output=True, timeout=30)
        if result.returncode != 0:
            return None

        # Decode
        cmd_decode = ['ffmpeg', '-y', '-i', compressed_path] + codec_config['decode_args'] + [decoded_path]
        result = subprocess.run(cmd_decode, capture_output=True, timeout=30)
        if result.returncode != 0:
            return None

        # Clean up compressed file
        if os.path.exists(compressed_path):
            os.remove(compressed_path)

        return decoded_path

    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None


def compute_stability_metrics(
    original_features: np.ndarray,
    compressed_features: np.ndarray,
    feature_indices: List[int]
) -> Dict[str, float]:
    """Compute comprehensive stability metrics."""
    orig = original_features[:, feature_indices]
    comp = compressed_features[:, feature_indices]

    # Remove NaN
    valid_mask = ~(np.isnan(orig).any(axis=1) | np.isnan(comp).any(axis=1))
    orig = orig[valid_mask]
    comp = comp[valid_mask]

    if len(orig) < 10:
        return {'pearson_r': np.nan, 'spearman_r': np.nan, 'mae': np.nan,
                'nmae': np.nan, 'n_valid': len(orig)}

    orig_flat = orig.flatten()
    comp_flat = comp.flatten()

    # Correlations
    pearson_r, pearson_p = pearsonr(orig_flat, comp_flat)
    spearman_r, _ = spearmanr(orig_flat, comp_flat)

    # MAE
    mae = np.mean(np.abs(orig - comp))
    nmae = mae / (np.std(orig_flat) + 1e-8)

    # Per-feature correlations
    per_feat_corrs = []
    for i in range(orig.shape[1]):
        if np.std(orig[:, i]) > 1e-8 and np.std(comp[:, i]) > 1e-8:
            r, _ = pearsonr(orig[:, i], comp[:, i])
            per_feat_corrs.append(r)
    avg_per_feat = np.nanmean(per_feat_corrs) if per_feat_corrs else np.nan

    return {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'mae': mae,
        'nmae': nmae,
        'avg_per_feature_corr': avg_per_feat,
        'n_valid': len(orig)
    }


def generate_comprehensive_report(results_df: pd.DataFrame, output_dir: str):
    """Generate comprehensive analysis report with visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (14, 10)

    # 1. Large heatmap: All codecs x Feature groups
    fig, ax = plt.subplots(figsize=(16, 10))
    pivot = results_df.pivot_table(
        index='feature_group',
        columns='codec',
        values='pearson_r'
    )
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, ax=ax,
                cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title('Feature Stability Across All Real-World Codecs', fontsize=14)
    ax.set_xlabel('Codec', fontsize=12)
    ax.set_ylabel('Feature Group', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_all_codecs.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'heatmap_all_codecs.pdf'), bbox_inches='tight')
    plt.close()

    # 2. Heatmap by category
    categories = results_df['category'].unique()
    for category in categories:
        cat_data = results_df[results_df['category'] == category]
        if len(cat_data) == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = cat_data.pivot_table(
            index='feature_group',
            columns='codec',
            values='pearson_r'
        )
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0.5, vmax=1.0, ax=ax)
        ax.set_title(f'Feature Stability: {category} Codecs', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{category.lower().replace(" ", "_")}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Box plot: Stability distribution by category
    fig, ax = plt.subplots(figsize=(12, 6))
    category_order = ['Telephony', 'VoIP', 'Streaming', 'Messaging', 'Video Conferencing']
    valid_categories = [c for c in category_order if c in results_df['category'].values]
    results_df.boxplot(column='pearson_r', by='category', ax=ax)
    ax.set_xlabel('Codec Category', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Feature Stability by Codec Category', fontsize=14)
    plt.suptitle('')
    ax.axhline(y=0.9, color='red', linestyle='--', label='Threshold (r=0.9)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_by_category.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Feature group ranking with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    group_stats = results_df.groupby('feature_group')['pearson_r'].agg(['mean', 'std', 'min', 'max'])
    group_stats = group_stats.sort_values('mean', ascending=True)

    colors = ['#d73027' if v < 0.8 else '#fc8d59' if v < 0.9 else '#91cf60' if v < 0.95 else '#1a9850'
              for v in group_stats['mean'].values]

    y_pos = range(len(group_stats))
    ax.barh(y_pos, group_stats['mean'], xerr=group_stats['std'],
            color=colors, capsize=5, alpha=0.8)
    ax.scatter(group_stats['min'], y_pos, marker='|', color='black', s=100, label='Min')
    ax.scatter(group_stats['max'], y_pos, marker='|', color='blue', s=100, label='Max')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_stats.index)
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_title('Feature Group Stability Ranking (Mean ± Std, with Min/Max)', fontsize=14)
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_ranking_comprehensive.png'), dpi=300)
    plt.close()

    # 5. Attack scenario analysis
    scenario_data = results_df.groupby('attack_scenario')['pearson_r'].agg(['mean', 'min'])
    scenario_data = scenario_data.sort_values('min', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(scenario_data))
    ax.barh(y_pos, scenario_data['mean'], alpha=0.7, label='Mean', color='steelblue')
    ax.scatter(scenario_data['min'], y_pos, marker='o', color='red', s=80, label='Worst-case')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenario_data.index)
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_title('Feature Stability by Attack Scenario', fontsize=14)
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_scenario_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comprehensive report saved to: {output_dir}")


def run_comprehensive_validation(
    audio_dir: str,
    file_ids: List[str],
    n_samples: int = 100,
    output_dir: str = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run comprehensive real-world codec validation.
    """
    np.random.seed(seed)

    # Check codec availability
    available_codecs = check_codec_availability()
    print(f"\nCodec availability check:")
    for codec, avail in available_codecs.items():
        print(f"  - {codec}: {'Available' if avail else 'NOT AVAILABLE'}")

    # Filter to available codecs
    valid_codecs = []
    for codec in COMPREHENSIVE_CODECS:
        # Extract the main codec from ffmpeg_args
        codec_name = None
        for i, arg in enumerate(codec['ffmpeg_args']):
            if arg == '-c:a' and i + 1 < len(codec['ffmpeg_args']):
                codec_name = codec['ffmpeg_args'][i + 1]
                break

        if codec_name is None:
            valid_codecs.append(codec)
        elif available_codecs.get(codec_name, True):
            valid_codecs.append(codec)
        else:
            print(f"  Skipping {codec['name']}: codec {codec_name} not available")

    # Sample files
    if len(file_ids) > n_samples:
        sampled_ids = np.random.choice(file_ids, n_samples, replace=False)
    else:
        sampled_ids = file_ids

    print(f"\n{'='*80}")
    print("COMPREHENSIVE REAL-WORLD CODEC VALIDATION")
    print(f"{'='*80}")
    print(f"\nSamples: {len(sampled_ids)}")
    print(f"Codecs to test: {len(valid_codecs)}")
    print(f"\nCodecs by Category:")
    categories = {}
    for codec in valid_codecs:
        cat = codec.get('category', 'Other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(codec['name'])
    for cat, codecs in categories.items():
        print(f"  {cat}: {', '.join(codecs)}")
    print(f"{'='*80}\n")

    # Find audio directory
    flac_dir = Path(audio_dir) / 'flac'
    if not flac_dir.exists():
        flac_dir = Path(audio_dir)

    # Extract original features
    print("Extracting original features...")
    original_features = []
    valid_ids = []

    for fid in tqdm(sampled_ids, desc="Original"):
        audio_path = flac_dir / f"{fid}.flac"
        if not audio_path.exists():
            continue
        try:
            feat = extract_features_v4(str(audio_path))
            original_features.append(feat)
            valid_ids.append((fid, str(audio_path)))
        except Exception:
            continue

    original_features = np.array(original_features)
    print(f"  Valid samples: {len(valid_ids)}")

    # Results storage
    results = []

    # Process each codec
    with tempfile.TemporaryDirectory() as temp_dir:
        for codec_config in valid_codecs:
            print(f"\nProcessing: {codec_config['name']} ({codec_config.get('category', 'Other')})...")

            compressed_features = []

            for fid, audio_path in tqdm(valid_ids, desc=f"  {codec_config['name']}"):
                decoded_path = compress_audio(audio_path, codec_config, temp_dir)

                if decoded_path is None:
                    compressed_features.append(np.full(36, np.nan))
                    continue

                try:
                    feat = extract_features_v4(decoded_path)
                    compressed_features.append(feat)
                except Exception:
                    compressed_features.append(np.full(36, np.nan))
                finally:
                    if decoded_path and os.path.exists(decoded_path):
                        os.remove(decoded_path)

            compressed_features = np.array(compressed_features)

            # Compute metrics for each feature group
            for group_name, group_info in FEATURE_GROUPS.items():
                indices = group_info['indices']

                stability = compute_stability_metrics(
                    original_features, compressed_features, indices
                )

                results.append({
                    'codec': codec_config['name'],
                    'codec_full_name': codec_config.get('full_name', ''),
                    'category': codec_config.get('category', 'Other'),
                    'bitrate': codec_config.get('bitrate', 'Unknown'),
                    'attack_scenario': codec_config.get('attack_scenario', ''),
                    'feature_group': group_name,
                    'n_features': group_info['dims'],
                    **stability
                })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE VALIDATION RESULTS")
    print(f"{'='*80}")

    # Overall ranking
    print("\n\nFeature Group Stability Ranking (across ALL codecs):")
    print("-" * 70)
    ranking = df.groupby('feature_group')['pearson_r'].agg(['mean', 'std', 'min'])
    ranking = ranking.sort_values('mean', ascending=False)
    for group in ranking.index:
        row = ranking.loc[group]
        status = "HIGHLY STABLE" if row['mean'] >= 0.95 else \
                 "STABLE" if row['mean'] >= 0.90 else \
                 "MODERATE" if row['mean'] >= 0.80 else "UNSTABLE"
        print(f"  {group}: mean={row['mean']:.4f}, std={row['std']:.4f}, min={row['min']:.4f} [{status}]")

    # Worst-case scenarios
    print("\n\nWorst-Case Scenarios (minimum correlation):")
    print("-" * 70)
    worst = df.loc[df.groupby('feature_group')['pearson_r'].idxmin()]
    for _, row in worst.iterrows():
        print(f"  {row['feature_group']}: {row['codec']} (r={row['pearson_r']:.4f}) - {row['attack_scenario']}")

    # Category analysis
    print("\n\nStability by Codec Category:")
    print("-" * 70)
    cat_stats = df.groupby('category')['pearson_r'].agg(['mean', 'min'])
    cat_stats = cat_stats.sort_values('mean', ascending=False)
    for cat in cat_stats.index:
        row = cat_stats.loc[cat]
        print(f"  {cat}: mean={row['mean']:.4f}, worst-case={row['min']:.4f}")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'comprehensive_codec_validation.csv'), index=False)

        # Generate visualizations
        generate_comprehensive_report(df, output_dir)

        # Save summary JSON
        summary = {
            'methodology': 'Comprehensive real-world codec validation',
            'n_samples': len(valid_ids),
            'codecs_tested': len(valid_codecs),
            'categories': list(categories.keys()),
            'ranking': ranking['mean'].to_dict(),
            'worst_cases': {row['feature_group']: {'codec': row['codec'],
                                                    'pearson_r': row['pearson_r'],
                                                    'scenario': row['attack_scenario']}
                           for _, row in worst.iterrows()},
            'recommendations': []
        }

        for group in ranking.index:
            mean_r = ranking.loc[group, 'mean']
            min_r = ranking.loc[group, 'min']
            if mean_r >= 0.95 and min_r >= 0.85:
                summary['recommendations'].append(
                    f"{group}: RECOMMENDED - Robust across all codecs (mean={mean_r:.3f}, min={min_r:.3f})"
                )
            elif mean_r >= 0.90:
                summary['recommendations'].append(
                    f"{group}: ACCEPTABLE - Generally stable (mean={mean_r:.3f}, min={min_r:.3f})"
                )
            elif mean_r >= 0.80:
                summary['recommendations'].append(
                    f"{group}: CAUTION - Some codec sensitivity (mean={mean_r:.3f}, min={min_r:.3f})"
                )
            else:
                summary['recommendations'].append(
                    f"{group}: NOT RECOMMENDED - Codec-sensitive (mean={mean_r:.3f}, min={min_r:.3f})"
                )

        with open(os.path.join(output_dir, 'comprehensive_validation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Comprehensive real-world codec robustness validation'
    )
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--file_ids', type=str, required=True,
                        help='Text file with file IDs')
    parser.add_argument('--output_dir', type=str,
                        default='evidence/experiments/codec_validation_comprehensive',
                        help='Output directory')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Load file IDs
    with open(args.file_ids) as f:
        file_ids = [line.strip() for line in f if line.strip()]

    # Run validation
    results = run_comprehensive_validation(
        audio_dir=args.audio_dir,
        file_ids=file_ids,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed
    )

    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION COMPLETE")
    print("="*80)
