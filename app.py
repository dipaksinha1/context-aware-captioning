"""Run with: streamlit run app.py (after pip install -e .)."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import mimetypes
import os

import pandas as pd
import streamlit as st

from context_caption.captions import apply_edits
from context_caption.exporters import timeline, to_json, to_srt, to_vtt
from context_caption.media import MAX_BYTES, SUPPORTED
from context_caption.pipeline import Settings, process
from context_caption.player import player_html
from context_caption.schema import Project
from context_caption.sounds import apply_sound_edits, format_sound

ROOT = Path(__file__).parent
st.set_page_config(page_title="Context Caption", page_icon="💬", layout="wide")
st.markdown("""<style>
.block-container{max-width:1250px;padding-top:2rem}
[data-testid="stMetric"]{background:white;border:1px solid #e2e8f0;padding:16px;border-radius:12px}
h1{letter-spacing:-.04em}
</style>""", unsafe_allow_html=True)
st.caption("COSC-5926  /  ACCESSIBLE MULTI-SPEAKER MEDIA")
st.title("Every voice. In context.")
st.write("Turn recordings into speaker-aware captions. Review the words, name the voices, and export.")

with st.sidebar:
    st.subheader("Processing settings")
    model = st.selectbox("Whisper model", ["base", "tiny", "small", "medium", "large-v3"])
    device = st.selectbox("Compute device", ["cpu", "cuda"])
    language = st.text_input("Language code (blank = detect)", placeholder="en")
    count = st.number_input("Known speaker count (0 = detect)", min_value=0, max_value=20, value=0)
    batch = st.select_slider("Batch size", [1, 2, 4, 8, 16], value=4)
    emotions = st.checkbox("Estimate English text emotion", value=False)
    sounds = st.checkbox("Detect background sounds", value=False,
                         help="Adds bracketed cues such as [dog barking]. Slower on CPU.")
    token = st.text_input("Hugging Face read token", type="password", value=os.getenv("HF_TOKEN", ""))
    st.caption("Token stays in this session and is excluded from project exports.")
    st.link_button("Accept diarization model access", "https://huggingface.co/pyannote/speaker-diarization-community-1")
    st.caption("CPU works on macOS, Windows, and Linux. NVIDIA CUDA is optional.")
    if st.button("Clear session", width="stretch"):
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()

upload_tab, reopen_tab = st.tabs(["New recording", "Open saved project"])
with upload_tab:
    col1, col2 = st.columns([3, 1])
    with col1:
        upload = st.file_uploader("Upload audio or video", type=sorted(s[1:] for s in SUPPORTED))
        consent = st.checkbox("I have permission to process the voices in this recording.")
    with col2:
        st.write("Explore first")
        if st.button("Load sample project", width="stretch"):
            st.session_state.project = Project.from_dict(json.loads(
                (ROOT / "examples/demo_project.json").read_text(encoding="utf-8")
            ))
            sample = ROOT / "examples/demo.wav"
            st.session_state.media = sample.read_bytes() if sample.exists() else None
            st.session_state.mime = "audio/wav"
            st.session_state.revision = st.session_state.get("revision", 0) + 1
    if st.button("Generate captions", type="primary", disabled=not(upload and consent)):
        try:
            if upload.size > MAX_BYTES:
                raise ValueError("Upload limit is 200 MiB.")
            settings = Settings(model=model, device=device, batch_size=batch,
                                language=language.strip() or None,
                                min_speakers=int(count) or None, max_speakers=int(count) or None,
                                emotions=emotions, sounds=sounds, token=token)
            settings.validate()
            bar = st.progress(0, text="Starting")
            with TemporaryDirectory(prefix="caption-upload-") as directory:
                path = Path(directory) / ("input" + Path(upload.name).suffix.lower())
                path.write_bytes(upload.getvalue())
                project = process(path, settings, lambda value, label: bar.progress(value, text=label))
            st.session_state.project = project
            st.session_state.media = upload.getvalue()
            st.session_state.mime = mimetypes.guess_type(upload.name)[0] or "audio/wav"
            st.session_state.revision = st.session_state.get("revision", 0) + 1
        except Exception:
            # Avoid rendering upstream exception strings that may include signed URLs/tokens.
            st.error("Processing failed. Check ML installation, FFmpeg, model access, and memory. "
                     "Try CPU, the tiny model, and batch size 1. Your previous project is retained.")
with reopen_tab:
    saved = st.file_uploader("Project JSON", type=["json"])
    if st.button("Open project", disabled=saved is None):
        try:
            if saved.size > 10 * 1024 * 1024:
                raise ValueError("Project JSON limit is 10 MiB.")
            candidate = Project.from_dict(json.loads(saved.getvalue()))
            st.session_state.project = candidate
            st.session_state.media = None
            st.session_state.revision = st.session_state.get("revision", 0) + 1
        except (ValueError, TypeError, KeyError):
            st.error("Could not open this project. Use a valid Context Caption JSON export.")
    attachment = st.file_uploader("Attach original media for playback (optional)",
                                  type=sorted(s[1:] for s in SUPPORTED), key="attach_media")
    if st.button("Attach media", disabled=attachment is None):
        st.session_state.media = attachment.getvalue()
        st.session_state.mime = mimetypes.guess_type(attachment.name)[0] or "audio/wav"

project = st.session_state.get("project")
if project is None:
    st.info("Upload a recording or load the sample to begin. Models download only during real processing.")
    st.stop()

revision = st.session_state.get("revision", 0)
if project.metadata.get("mode") == "demo":
    st.info("Sample project: predefined captions and sound cues over synthetic speech and effects. "
            "No model inference was performed.")
if not project.captions:
    st.warning("No speech was detected. Try a recording with clearer speech.")
    if not project.sounds:
        st.stop()
metrics = st.columns(4)
metrics[0].metric("Caption cues", len(project.captions))
metrics[1].metric("Speaker labels", len(project.speakers))
metrics[2].metric("Sound cues", len(project.sounds))
metrics[3].metric("Language", project.language.upper())

preview, editor, words, export = st.tabs(["Playback", "Review & edit", "Word timestamps", "Export"])
with editor:
    st.subheader("Name the voices")
    with st.form(f"names-{revision}"):
        names = {}
        for speaker, name in project.speakers.items():
            names[speaker] = st.text_input(speaker, value=name, key=f"name-{revision}-{speaker}")
        if st.form_submit_button("Save speaker names"):
            if all(n.strip() for n in names.values()):
                project.speakers = {key: name.strip() for key, name in names.items()}
                st.success("Speaker names saved.")
            else:
                st.error("Speaker names cannot be blank.")
    st.subheader("Correct the captions")
    st.caption("Times are seconds. Original words and their alignment remain available after edits. "
               "Text edits clear emotion estimates; playback then shows the edited cue without word highlighting.")
    rows = [{"id": c.id, "start": c.start, "end": c.end,
             "speaker": c.speaker, "text": c.text} for c in project.captions]
    frame = st.data_editor(pd.DataFrame(rows, columns=["id", "start", "end", "speaker", "text"]),
        hide_index=True, disabled=["id"],
        key=f"editor-{revision}", width="stretch",
        column_config={"speaker": st.column_config.SelectboxColumn(options=list(project.speakers)),
                       "start": st.column_config.NumberColumn(min_value=0, format="%.3f"),
                       "end": st.column_config.NumberColumn(min_value=0, format="%.3f"),
                       "text": st.column_config.TextColumn(width="large")})
    st.subheader("Background sounds")
    st.caption("Describe non-speech sounds in lowercase present tense, such as dog barking or "
               "door slams. Brackets are added for you. Add rows for sounds that were missed; "
               "delete rows that are wrong or unimportant. Save caption edits stores both tables.")
    sound_rows = pd.DataFrame(
        [{"id": s.id, "start": s.start, "end": s.end, "text": s.text} for s in project.sounds],
        columns=["id", "start", "end", "text"],
    ).astype({"id": "Int64", "start": "float64", "end": "float64", "text": "object"})
    sound_frame = st.data_editor(sound_rows, hide_index=True, disabled=["id"], num_rows="dynamic",
        key=f"sounds-{revision}", width="stretch",
        column_config={"start": st.column_config.NumberColumn(min_value=0, format="%.3f"),
                       "end": st.column_config.NumberColumn(min_value=0, format="%.3f"),
                       "text": st.column_config.TextColumn("sound", width="large")})
    # One save for both tables: saving resets the editors, so a second button would drop
    # whatever was still unsaved in the other table.
    if st.button("Save caption edits", type="primary"):
        try:
            cleaned = sound_frame.astype(object).where(sound_frame.notna(), None)
            st.session_state.project = apply_sound_edits(
                apply_edits(project, frame.to_dict("records")), cleaned.to_dict("records"))
            st.session_state.revision = revision + 1
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
with preview:
    show_emotions = st.checkbox("Show available emotion cues", value=False)
    show_sounds = st.checkbox("Show background sound cues", value=True)
    st.caption("Speaker names accompany colors. Text emotion is an uncertain inference, not a measurement of feelings. "
               "Sounds appear in brackets, apart from speech.")
    if st.session_state.get("media"):
        st.iframe(player_html(st.session_state.media, st.session_state.mime,
                                    project, show_emotions, show_sounds), height=650,
                  alt="Recording with synchronized speaker captions and sound cues")
    else:
        st.info("Attach the original recording in Open saved project to enable synchronized playback.")
    query = st.text_input("Search transcript")
    for is_sound, c in timeline(project, show_sounds):
        text = format_sound(c) if is_sound else c.text
        if not query or query.casefold() in text.casefold():
            who = "Sound" if is_sound else project.speakers[c.speaker]
            st.write(f"{c.start:.2f}–{c.end:.2f}s · **{who}**")
            st.text(text)
with words:
    st.caption("These are original model/sample words. Empty times mean alignment was unavailable.")
    st.dataframe(pd.DataFrame([
        {"cue": c.id, **w.__dict__} for c in project.captions for w in c.words
    ]), hide_index=True, width="stretch")
with export:
    include_emotions = st.checkbox("Include available emotion cues in subtitles")
    include_sounds = st.checkbox("Include background sound cues in subtitles", value=True)
    cols = st.columns(3)
    cols[0].download_button("Download SRT", to_srt(project, include_emotions, include_sounds),
                          "captions.srt", "text/plain", width="stretch")
    cols[1].download_button("Download VTT", to_vtt(project, include_emotions, include_sounds),
                          "captions.vtt", "text/vtt", width="stretch")
    cols[2].download_button("Save project JSON", to_json(project),
                          "caption-project.json", "application/json", width="stretch")
    st.caption("JSON preserves names, edits, original words, sound cues, and model metadata. "
               "Media and tokens are excluded. SRT carries speaker names; VTT adds color styling "
               "where supported. Sound cues export as their own bracketed cues, such as [dog barking].")
    st.json(project.metadata, expanded=False)
