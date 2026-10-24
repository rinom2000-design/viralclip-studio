"""
ViralClip Studio — Interactive 4-Step Sequential Web UI (Streamlit, Windows & Cross-Platform)
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    RETENTION_PRESETS,
    PeakSegment,
    VideoCandidate,
    download_video_clip,
    format_timestamp,
    generate_voiceover_mp3,
    generate_word_level_ass_subtitles,
    render_final_composite_video,
    search_subclip_recommendations,
    search_trending_videos_with_heatmaps,
)

WORKSPACE_DIR = Path(__file__).parent / "media_workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="ViralClip Studio — AI Heatmap Clipper & Editor",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 ViralClip Studio (Windows Edition)")
st.caption(
    "Sequential Pipeline: **1. Search & Most-Replayed Heatmap Peaks** → **2. Clip Selection** → "
    "**3. 9:16 Reframe, Subtitles, Voiceover & Polish** → **4. Auto-Fetched Retention / Contextual Sub-Clip Merge**"
)

if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "clipped_main_path" not in st.session_state:
    st.session_state.clipped_main_path = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "ass_path" not in st.session_state:
    st.session_state.ass_path = None
if "subclip_candidates" not in st.session_state:
    st.session_state.subclip_candidates = []
if "final_video_path" not in st.session_state:
    st.session_state.final_video_path = None

# =====================================================================
# STEP 1: TOPIC SEARCH & MOST REPLAYED GRAPH ANALYSIS
# =====================================================================
st.header("Step 1: Topic Search & 'Most Replayed' Peak Finder")
col_q, col_n, col_dur, col_btn = st.columns([4, 1, 1, 1.5])
with col_q:
    topic_query = st.text_input(
        "Enter Topic or Niche Keyword",
        placeholder="e.g., AI agents startup, productivity habits, SpaceX Starship...",
    )
with col_n:
    max_videos = st.number_input("Max Videos", min_value=2, max_value=12, value=5)
with col_dur:
    clip_dur = st.number_input("Clip Length (s)", min_value=15, max_value=90, value=35)
with col_btn:
    st.write("")
    st.write("")
    search_clicked = st.button("🔍 Find Trending Peaks", type="primary", use_container_width=True)

if search_clicked and topic_query.strip():
    with st.spinner("Searching YouTube & extracting 'Most Replayed' retention heatmaps..."):
        st.session_state.candidates = search_trending_videos_with_heatmaps(
            topic_query.strip(), max_results=int(max_videos), target_clip_duration=float(clip_dur)
        )

if st.session_state.candidates:
    video_titles = [
        f"{idx + 1}. {c.title} ({c.view_count:,} views) {'🔥 Heatmap' if c.heatmap else '⚡ Auto-Hook'}"
        for idx, c in enumerate(st.session_state.candidates)
    ]
    chosen_idx = st.selectbox(
        "Select a Trending Video to Inspect:",
        range(len(video_titles)),
        format_func=lambda i: video_titles[i],
    )
    cand: VideoCandidate = st.session_state.candidates[chosen_idx]
    st.session_state.selected_video = cand

    c_info, c_chart = st.columns([1.3, 2.2])
    with c_info:
        if cand.thumbnail:
            st.image(cand.thumbnail, use_container_width=True)
        st.markdown(f"**[{cand.title}]({cand.url})**")
        st.caption(
            f"Channel: {cand.channel} • Views: {cand.view_count:,} • Duration: {format_timestamp(cand.duration)}"
        )

    with c_chart:
        if cand.heatmap:
            df_hm = pd.DataFrame(cand.heatmap)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_hm["start_time"],
                    y=df_hm["value"],
                    fill="tozeroy",
                    mode="lines",
                    name="Most Replayed Score",
                    line=dict(color="#FF4B4B", width=2),
                )
            )
            for idx, pk in enumerate(cand.peaks):
                fig.add_vrect(
                    x0=pk.start_time,
                    x1=pk.end_time,
                    fillcolor="#00CC96",
                    opacity=0.25,
                    line_width=1,
                    annotation_text=f"Peak #{idx+1} ({int(pk.score*100)}%)",
                )
            fig.update_layout(
                title="YouTube 'Most Replayed' Audience Retention Graph (heatMarkers)",
                xaxis_title="Timestamp (seconds)",
                yaxis_title="Replay Intensity (0-1)",
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "This video does not expose public heatmap markers yet; fallback hook windows have been calculated."
            )

    # =====================================================================
    # STEP 2: CHOOSE & CLIP THE VIDEO SEGMENT
    # =====================================================================
    st.header("Step 2: Choose & Clip Your Target Segment")
    peak_options = [
        f"Peak #{i+1}: {format_timestamp(p.start_time)} – {format_timestamp(p.end_time)} "
        f"(Replay Score: {int(p.score * 100)}%) | {p.transcript_snippet[:75]}"
        for i, p in enumerate(cand.peaks)
    ]
    selected_peak_idx = st.radio(
        "Recommended High-Retention Segments:",
        range(len(peak_options)),
        format_func=lambda i: peak_options[i],
    )
    chosen_peak: PeakSegment = cand.peaks[selected_peak_idx]

    col_s, col_e, col_clip_btn = st.columns([1, 1, 1.5])
    with col_s:
        custom_start = st.number_input(
            "Start Timestamp (sec)",
            value=float(chosen_peak.start_time),
            min_value=0.0,
            max_value=float(cand.duration),
        )
    with col_e:
        custom_end = st.number_input(
            "End Timestamp (sec)",
            value=float(chosen_peak.end_time),
            min_value=1.0,
            max_value=float(cand.duration),
        )
    with col_clip_btn:
        st.write("")
        st.write("")
        if st.button("✂️ Download & Clip Selected Window", type="primary", use_container_width=True):
            with st.spinner(
                f"Clipping {format_timestamp(custom_start)} to {format_timestamp(custom_end)} from YouTube..."
            ):
                out_raw = (
                    WORKSPACE_DIR
                    / f"main_clip_{cand.video_id}_{int(custom_start)}_{int(custom_end)}.mp4"
                )
                st.session_state.clipped_main_path = download_video_clip(
                    cand.url, custom_start, custom_end, out_raw
                )
                st.session_state.transcript_text = chosen_peak.transcript_snippet

if st.session_state.clipped_main_path and Path(st.session_state.clipped_main_path).exists():
    st.success(f"Main Clip Ready: `{st.session_state.clipped_main_path.name}`")
    st.video(str(st.session_state.clipped_main_path))

    # =====================================================================
    # STEP 3: REFRAME, SUBTITLES, VOICEOVER & QUALITY EDITS
    # =====================================================================
    st.header("Step 3: Reframe, Subtitles, Voiceover & Polish Settings")
    c_sub, c_vo, c_pol = st.columns(3)
    with c_sub:
        enable_subs = st.checkbox(
            "Burn Animated Word-by-Word Subtitles (faster-whisper)", value=True
        )
        whisper_model = st.selectbox("Whisper Model Size", ["tiny", "base", "small"], index=1)
        if st.button("📝 Transcribe & Preview Subtitles"):
            with st.spinner("Running faster-whisper word-level transcription..."):
                ass_file = WORKSPACE_DIR / "captions.ass"
                ass_path, full_txt = generate_word_level_ass_subtitles(
                    Path(st.session_state.clipped_main_path), ass_file, model_size=whisper_model
                )
                st.session_state.ass_path = ass_path
                st.session_state.transcript_text = full_txt
                st.success("Subtitles generated!")

    with c_vo:
        enable_vo = st.checkbox("Add AI Voiceover Hook / Commentary (edge-tts)", value=False)
        vo_script = st.text_area(
            "Voiceover Script (optional)",
            placeholder="Watch how this changes everything in 30 seconds...",
            disabled=not enable_vo,
        )
        vo_voice = st.selectbox(
            "Voice Profile",
            [
                "en-US-ChristopherNeural",
                "en-US-GuyNeural",
                "en-US-AriaNeural",
                "en-GB-RyanNeural",
            ],
            disabled=not enable_vo,
        )

    with c_pol:
        enable_polish = st.checkbox(
            "Apply Color Contrast, Unsharp Mask & -16 LUFS Audio Normalization", value=True
        )
        if st.session_state.transcript_text:
            st.text_area(
                "Detected Clip Transcript / Context:",
                value=st.session_state.transcript_text,
                height=110,
            )

    # =====================================================================
    # STEP 4: SECONDARY SUB-CLIP RECOMMENDATION & MERGE (RETENTION OR RELEVANT)
    # =====================================================================
    st.header("Step 4: Recommend & Auto-Fetch Secondary Sub-Clip from YouTube")
    sub_mode = st.radio(
        "How should the secondary clip work?",
        [
            "Retention Video Below (Split-Screen 9:16 — Satisfying / Gameplay clip stacked below main video)",
            "Relevant Context Video Alongside (Auto-matched to your main clip's topic/transcript)",
            "No Secondary Clip (Full-Screen 9:16 Face-Tracked Main Clip Only)",
        ],
    )

    if not sub_mode.startswith("No Secondary"):
        is_retention = sub_mode.startswith("Retention")
        col_preset, col_custom, col_fetch = st.columns([1.5, 2, 1.2])
        with col_preset:
            if is_retention:
                preset_label = st.selectbox("Retention Category:", list(RETENTION_PRESETS.keys()))
                preset_q = RETENTION_PRESETS[preset_label]
            else:
                st.info("Uses keywords from your main clip to find relevant B-roll on YouTube.")
                preset_q = None
        with col_custom:
            custom_sub_q = st.text_input(
                "Or Custom Sub-Clip Search Query (optional):",
                placeholder="Leave blank to use preset / automatic transcript keywords",
            )
        with col_fetch:
            st.write("")
            st.write("")
            if st.button("🎯 Recommend Sub-Clips", use_container_width=True):
                with st.spinner("Auto-fetching high-retention secondary clips from YouTube..."):
                    st.session_state.subclip_candidates = search_subclip_recommendations(
                        mode="retention" if is_retention else "contextual",
                        main_transcript=st.session_state.transcript_text or topic_query,
                        retention_preset_query=preset_q,
                        custom_query=custom_sub_q,
                        max_results=4,
                    )

        chosen_sub_candidate: VideoCandidate | None = None
        if st.session_state.subclip_candidates:
            sub_titles = [
                f"{i+1}. {sc.title} ({sc.view_count:,} views) — Peak: {format_timestamp(sc.peaks[0].start_time)}–{format_timestamp(sc.peaks[0].end_time)}"
                for i, sc in enumerate(st.session_state.subclip_candidates)
            ]
            sub_idx = st.selectbox(
                "Recommended Secondary Clips from YouTube:",
                range(len(sub_titles)),
                format_func=lambda i: sub_titles[i],
            )
            chosen_sub_candidate = st.session_state.subclip_candidates[sub_idx]
            st.caption(
                f"Selected Sub-Clip URL: [{chosen_sub_candidate.title}]({chosen_sub_candidate.url})"
            )

    if st.button("🚀 Render Final Polished Short Video", type="primary", use_container_width=True):
        with st.spinner(
            "Processing smart reframe, subtitles, audio polish, and compositing final 9:16 video..."
        ):
            main_p = Path(st.session_state.clipped_main_path)

            ass_p = st.session_state.ass_path
            if enable_subs and not ass_p:
                ass_p, full_txt = generate_word_level_ass_subtitles(
                    main_p, WORKSPACE_DIR / "captions.ass", model_size=whisper_model
                )
                st.session_state.ass_path = ass_p
                st.session_state.transcript_text = full_txt

            vo_p = None
            if enable_vo and vo_script.strip():
                vo_p = generate_voiceover_mp3(
                    vo_script.strip(), WORKSPACE_DIR / "voiceover.mp3", voice=vo_voice
                )

            sub_p = None
            layout = "single_9_16"
            if not sub_mode.startswith("No Secondary") and st.session_state.subclip_candidates:
                sc = chosen_sub_candidate
                if sc and sc.peaks:
                    sp = sc.peaks[0]
                    sub_p = download_video_clip(
                        sc.url,
                        sp.start_time,
                        sp.end_time,
                        WORKSPACE_DIR / f"sub_clip_{sc.video_id}.mp4",
                    )
                    layout = "split_bottom" if sub_mode.startswith("Retention") else "side_by_side"

            final_out = WORKSPACE_DIR / "final_viral_short.mp4"
            st.session_state.final_video_path = render_final_composite_video(
                main_clip_path=main_p,
                output_path=final_out,
                sub_clip_path=sub_p,
                layout_mode=layout,
                ass_subtitle_path=ass_p if enable_subs else None,
                voiceover_path=vo_p,
                enhance_quality=enable_polish,
            )

if st.session_state.final_video_path and Path(st.session_state.final_video_path).exists():
    st.balloons()
    st.subheader("✅ Final Polished Video Ready")
    c_vid, c_dl = st.columns([1, 1.5])
    with c_vid:
        st.video(str(st.session_state.final_video_path))
    with c_dl:
        with open(st.session_state.final_video_path, "rb") as f:
            st.download_button(
                "⬇️ Download Final MP4 (1080x1920)",
                data=f,
                file_name="final_viral_short.mp4",
                mime="video/mp4",
                type="primary",
            )
