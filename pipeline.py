"""
ViralClip Studio — Core Pipeline Module (Windows & Cross-Platform Compatible)
Handles:
1. YouTube Topic Search + Most Replayed Heatmap (`heatMarkers`) & Transcript Peak Extraction
2. Frame-accurate section clipping via yt-dlp & FFmpeg
3. 9:16 Smart Reframing (OpenCV face-tracked crop), Word-by-Word Animated `.ass` Subtitles (faster-whisper),
   Voiceover (edge-tts with Windows asyncio SelectorEventLoop support), and Audio/Video Quality Polish
4. Auto-fetching & Merging Secondary Sub-Clips from YouTube (Retention videos below OR Relevant B-roll alongside)
"""

import asyncio
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yt_dlp


@dataclass
class PeakSegment:
    start_time: float
    end_time: float
    score: float  # 0.0 to 1.0 from YouTube Most Replayed heatmap
    transcript_snippet: str = ""
    keywords: List[str] = field(default_factory=list)


@dataclass
class VideoCandidate:
    video_id: str
    title: str
    url: str
    channel: str
    view_count: int
    duration: float
    thumbnail: str
    heatmap: List[Dict[str, float]] = field(default_factory=list)
    peaks: List[PeakSegment] = field(default_factory=list)


RETENTION_PRESETS = {
    "Subway Surfers Gameplay": "Subway Surfers no copyright gameplay 4k 60fps",
    "Minecraft Parkour": "Minecraft parkour gameplay no copyright 4k",
    "GTA V Stunt Racing": "GTA 5 mega ramp stunt race gameplay no commentary",
    "Satisfying Soap / Kinetic Sand": "oddly satisfying kinetic sand soap cutting no music",
    "Hydraulic Press": "hydraulic press satisfying compilation no commentary",
}

STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "your", "have", "more",
    "about", "what", "when", "where", "which", "their", "there", "been",
    "were", "they", "will", "would", "could", "should", "into", "just",
    "like", "some", "than", "then", "them", "these", "those", "very",
    "because", "really", "actually", "going", "think", "know", "people",
}


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def extract_keywords(text: str, top_n: int = 4) -> List[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def _parse_vtt_or_json3_subtitles(info: dict, start_sec: float, end_sec: float) -> str:
    """Extracts chapter or description context within [start_sec, end_sec]."""
    chapters = info.get("chapters") or []
    for ch in chapters:
        ch_start = ch.get("start_time", 0)
        ch_end = ch.get("end_time", 0)
        if ch_start <= end_sec and ch_end >= start_sec:
            return f"Chapter: {ch.get('title', '')}"
    desc = (info.get("description") or "").strip().splitlines()
    return desc[0][:140] if desc else info.get("title", "")


def find_top_heatmap_peaks(
    heatmap: List[Dict[str, float]],
    duration: float,
    clip_length: float = 35.0,
    top_k: int = 3,
    info: Optional[dict] = None,
) -> List[PeakSegment]:
    """
    Finds the top non-overlapping high-retention windows using YouTube's Most Replayed `heatmap` data.
    If a video lacks a heatmap, falls back to chapter markers or early-hook window.
    """
    if not heatmap:
        fallback_start = min(15.0, max(0.0, duration * 0.1))
        fallback_end = min(duration, fallback_start + clip_length)
        snippet = _parse_vtt_or_json3_subtitles(info or {}, fallback_start, fallback_end)
        return [
            PeakSegment(
                start_time=round(fallback_start, 1),
                end_time=round(fallback_end, 1),
                score=0.65,
                transcript_snippet=snippet,
                keywords=extract_keywords(snippet or (info or {}).get("title", "")),
            )
        ]

    sorted_bins = sorted(heatmap, key=lambda b: b.get("value", 0.0), reverse=True)
    peaks: List[PeakSegment] = []

    for b in sorted_bins:
        peak_center = (b.get("start_time", 0.0) + b.get("end_time", 0.0)) / 2.0
        win_start = max(0.0, peak_center - clip_length * 0.35)
        win_end = min(duration, win_start + clip_length)

        overlap = any(
            not (win_end < p.start_time or win_start > p.end_time) for p in peaks
        )
        if not overlap:
            snippet = _parse_vtt_or_json3_subtitles(info or {}, win_start, win_end)
            peaks.append(
                PeakSegment(
                    start_time=round(win_start, 1),
                    end_time=round(win_end, 1),
                    score=round(float(b.get("value", 0.0)), 3),
                    transcript_snippet=snippet,
                    keywords=extract_keywords(snippet + " " + (info or {}).get("title", "")),
                )
            )
        if len(peaks) >= top_k:
            break

    return sorted(peaks, key=lambda p: p.score, reverse=True)


def search_trending_videos_with_heatmaps(
    topic: str, max_results: int = 5, target_clip_duration: float = 35.0
) -> List[VideoCandidate]:
    """
    Step 1: Searches YouTube for videos matching `topic`, fetches full metadata including
    YouTube's `heatmap` (Most Replayed graph), and computes the best timestamp intervals.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
    }
    query = f"ytsearch{max_results}:{topic}"
    candidates: List[VideoCandidate] = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_res = ydl.extract_info(query, download=False)
        entries = search_res.get("entries") or []
        for entry in entries:
            if not entry:
                continue
            duration = float(entry.get("duration") or 0)
            if duration < 20:
                continue
            heatmap = entry.get("heatmap") or []
            peaks = find_top_heatmap_peaks(
                heatmap, duration, clip_length=target_clip_duration, top_k=3, info=entry
            )
            candidates.append(
                VideoCandidate(
                    video_id=entry.get("id", ""),
                    title=entry.get("title", "Untitled"),
                    url=entry.get("webpage_url", f"https://www.youtube.com/watch?v={entry.get('id')}"),
                    channel=entry.get("uploader", "Unknown Channel"),
                    view_count=int(entry.get("view_count") or 0),
                    duration=duration,
                    thumbnail=entry.get("thumbnail", ""),
                    heatmap=heatmap,
                    peaks=peaks,
                )
            )

    candidates.sort(
        key=lambda c: (1 if c.heatmap else 0, c.view_count),
        reverse=True,
    )
    return candidates


def download_video_clip(
    video_url: str,
    start_time: float,
    end_time: float,
    output_path: Path,
) -> Path:
    """
    Step 2: Downloads only the selected timestamp interval [start_time, end_time] from YouTube
    at up to 1080p MP4 using yt-dlp's section downloader + FFmpeg.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "outtmpl": str(output_path),
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        "force_keyframes_at_cuts": True,
        "merge_output_format": "mp4",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return output_path


def smart_reframe_crop_coordinates(video_path: Path, target_aspect: float = 9 / 16) -> str:
    """
    Step 3a: Uses OpenCV face detection across sampled frames to find the primary speaker's
    horizontal center and returns an FFmpeg crop filter expression.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            return f"crop=ih*{target_aspect}:ih"

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        centers_x: List[float] = []
        sample_step = max(1, total_frames // 20)

        for idx in range(0, total_frames, sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                centers_x.append(x + w / 2.0)
        cap.release()

        crop_w = min(width, int(height * target_aspect))
        crop_h = height
        if centers_x:
            median_x = sorted(centers_x)[len(centers_x) // 2]
            x_offset = int(max(0, min(width - crop_w, median_x - crop_w / 2)))
        else:
            x_offset = int((width - crop_w) / 2)

        return f"crop={crop_w}:{crop_h}:{x_offset}:0"
    except Exception:
        return f"crop=ih*{target_aspect}:ih"


def generate_word_level_ass_subtitles(
    video_path: Path, ass_output_path: Path, model_size: str = "base"
) -> Tuple[Path, str]:
    """
    Step 3b: Transcribes the clipped video using `faster-whisper` with word-level timestamps
    and generates a TikTok/Shorts-style dynamic pop-in `.ass` subtitle file.
    Automatically uses CUDA on Windows if an NVIDIA GPU is available, falling back to int8 CPU.
    """
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_size, device="auto", compute_type="int8")
    except Exception:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(str(video_path), word_timestamps=True)

    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralSub,Arial Black,72,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,860,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def _ass_time(sec: float) -> str:
        hrs = int(sec // 3600)
        mins = int((sec % 3600) // 60)
        secs = sec % 60
        return f"{hrs:d}:{mins:02d}:{secs:05.2f}"

    events: List[str] = []
    full_text_parts: List[str] = []

    for seg in segments:
        full_text_parts.append(seg.text.strip())
        words = seg.words or []
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            chunk = words[i : i + chunk_size]
            if not chunk:
                continue
            c_start = _ass_time(chunk[0].start)
            c_end = _ass_time(chunk[-1].end)
            text = " ".join(w.word.strip().upper() for w in chunk)
            events.append(f"Dialogue: 0,{c_start},{c_end},ViralSub,,0,0,0,,{text}")

    ass_output_path.write_text(ass_header + "\n".join(events), encoding="utf-8")
    return ass_output_path, " ".join(full_text_parts)


async def _synthesize_edge_tts(text: str, voice: str, out_file: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_file))


def generate_voiceover_mp3(text: str, out_file: Path, voice: str = "en-US-ChristopherNeural") -> Path:
    """Step 3c: Synthesizes an AI voiceover track (Windows asyncio compatible)."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_synthesize_edge_tts(text, voice, out_file))
    return out_file


def search_subclip_recommendations(
    mode: str,
    main_transcript: str,
    retention_preset_query: Optional[str] = None,
    custom_query: Optional[str] = None,
    max_results: int = 4,
) -> List[VideoCandidate]:
    """
    Step 4a: Auto-fetches secondary sub-clip candidates from YouTube:
    - If `mode == "retention"`: Searches high-retention gameplay/satisfying videos to run below the main clip.
    - If `mode == "contextual"`: Extracts keywords from `main_transcript` and searches relevant B-roll/context videos.
    """
    if custom_query and custom_query.strip():
        query_str = custom_query.strip()
    elif mode == "retention":
        query_str = retention_preset_query or RETENTION_PRESETS["Subway Surfers Gameplay"]
    else:
        kw = extract_keywords(main_transcript, top_n=4)
        query_str = " ".join(kw) + " footage b-roll" if kw else "cinematic b-roll footage"

    return search_trending_videos_with_heatmaps(query_str, max_results=max_results, target_clip_duration=30.0)


def _ffmpeg_escape_filter_path(path: Path) -> str:
    """
    Safely escapes Windows drive letters and backslashes (e.g. `C:\\Users\\...\\captions.ass`)
    for FFmpeg's `ass=` filter parser: `C\\:/Users/.../captions.ass`.
    """
    posix_str = path.resolve().as_posix()
    return posix_str.replace(":", "\\:").replace("'", "\\'")


def render_final_composite_video(
    main_clip_path: Path,
    output_path: Path,
    sub_clip_path: Optional[Path] = None,
    layout_mode: str = "split_bottom",  # "split_bottom", "side_by_side", or "single_9_16"
    ass_subtitle_path: Optional[Path] = None,
    voiceover_path: Optional[Path] = None,
    enhance_quality: bool = True,
) -> Path:
    """
    Step 3 & 4 Final Render:
    Reframes, applies color/contrast + loudness normalization, burns subtitles, mixes optional AI voiceover,
    and merges the secondary sub-clip (either stacked below for retention or alongside/overlaid for relevance).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eq_filter = ",eq=contrast=1.06:saturation=1.12,unsharp=3:3:0.5:3:3:0.0" if enhance_quality else ""

    if sub_clip_path and sub_clip_path.exists() and layout_mode == "split_bottom":
        top_crop = smart_reframe_crop_coordinates(main_clip_path, target_aspect=1080 / 960)
        vf_chain = (
            f"[0:v]{top_crop},scale=1080:960,setsar=1{eq_filter}[top];"
            f"[1:v]crop=ih*(1080/960):ih,scale=1080:960,setsar=1[bottom];"
            f"[top][bottom]vstack=inputs=2[stacked]"
        )
    elif sub_clip_path and sub_clip_path.exists() and layout_mode == "side_by_side":
        vf_chain = (
            f"[0:v]scale=540:960:force_original_aspect_ratio=increase,crop=540:960{eq_filter}[left];"
            f"[1:v]scale=540:960:force_original_aspect_ratio=increase,crop=540:960[right];"
            f"[left][right]hstack=inputs=2,pad=1080:1920:0:480:black[stacked]"
        )
    else:
        full_crop = smart_reframe_crop_coordinates(main_clip_path, target_aspect=9 / 16)
        vf_chain = f"[0:v]{full_crop},scale=1080:1920,setsar=1{eq_filter}[stacked]"

    last_video_label = "[stacked]"
    if ass_subtitle_path and ass_subtitle_path.exists():
        escaped_ass = _ffmpeg_escape_filter_path(ass_subtitle_path)
        vf_chain += f";{last_video_label}ass='{escaped_ass}'[vout]"
        last_video_label = "[vout]"

    cmd = ["ffmpeg", "-y", "-i", str(main_clip_path)]
    if sub_clip_path and sub_clip_path.exists() and layout_mode in ("split_bottom", "side_by_side"):
        cmd.extend(["-stream_loop", "-1", "-i", str(sub_clip_path)])

    if voiceover_path and voiceover_path.exists():
        vo_input_idx = 2 if (sub_clip_path and sub_clip_path.exists() and layout_mode in ("split_bottom", "side_by_side")) else 1
        cmd.extend(["-i", str(voiceover_path)])
        af_chain = (
            f"[0:a]loudnorm=I=-16:TP=-1.5:LRA=11,volume=0.85[a_main];"
            f"[{vo_input_idx}:a]loudnorm=I=-14:TP=-1.5:LRA=11,volume=1.15[a_vo];"
            f"[a_main][a_vo]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        full_filter = f"{vf_chain};{af_chain}"
        audio_map = "[aout]"
    else:
        full_filter = f"{vf_chain};[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
        audio_map = "[aout]"

    cmd.extend(
        [
            "-filter_complex",
            full_filter,
            "-map",
            last_video_label,
            "-map",
            audio_map,
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    )

    subprocess.run(cmd, check=True)
    return output_path
