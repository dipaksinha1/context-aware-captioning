# Validation record

Authoring environment: Python 3.12.14, Streamlit 1.66.1, Linux, FFmpeg available.

- Lint: `python -m ruff check .` — passed.
- Test suite: `python -m unittest discover -s tests -v` — 23 tests passed on Python 3.10 and 3.13.
- UI flow: Streamlit AppTest loaded the sample, renamed a speaker, and saved captions without exceptions.
- Media: FFmpeg converted a real stereo WAV to 16 kHz mono; corrupt media was rejected.
- Demo: FFmpeg/Flite generated two synthetic voices. Captions/times are predefined sample annotations.
- Adapter: stubbed WhisperX/pyannote/Transformers modules exercised orchestration, no-speech behavior,
  emotion addition, and nonfatal optional emotion failure. No real model inference was performed.
- Source compatibility: WhisperX 3.8.6 wheel inspected for the adapter signatures used by this project.
- Browser end-to-end test (Playwright + Chromium, Streamlit 1.66.1, Python 3.12): 29 checks passed —
  sample load, metrics, real audio playback with word highlighting, seek buttons, gap placeholder,
  transcript search, speaker rename and blank-name rejection, caption text edit, invalid timing
  rejection, SRT/VTT/JSON downloads, Clear session, JSON reopen, media attach, invalid JSON rejection,
  upload/consent gating, and the no-token error path. No browser JavaScript errors.
- ML extra install: `pip install -e '.[ml]'` resolves on Python 3.12 (torch 2.8.0, WhisperX 3.8.6,
  pyannote.audio 4.0.7, Transformers 4.57.6). Adapter calls were checked against the installed
  WhisperX/pyannote signatures, and real `whisperx.assign_word_speakers` output was run through the
  caption builder.
- Real-model smoke test: pending a Hugging Face token, accepted model conditions, and network access to
  huggingface.co (blocked in the cloud test environment; the pipeline stops at the Whisper model download).
- Supported Python versions: 3.10–3.13, as in the proposal. Python 3.10 uses Streamlit 1.65.0
  (1.66 requires >=3.11); 3.11+ uses Streamlit 1.66.1.
- CI matrix: Python 3.10, 3.11, 3.12, and 3.13; see the Actions tab for results of the compatibility fix.
- GitHub repository: `dipaksinha1/context-aware-captioning`.
- GitHub CI: workflow supplied; inspect the Actions tab for the current run status.

No ASR, diarization, emotion accuracy, or HCI study outcomes are claimed.

CI setup also validates the wheel-bundled FFmpeg executable before running media tests.
The workflow uses Node.js 24 actions (`checkout@v6`, `setup-python@v6`) on Ubuntu 24.04.
