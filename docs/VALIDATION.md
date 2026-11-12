# Validation record

Authoring environment: Python 3.12.14, Streamlit 1.66.1, Linux, FFmpeg available.

- Lint: `python -m ruff check .` — passed.
- Test suite: `python -m unittest discover -s tests -v` — 20 tests passed on Python 3.10 and 3.13.
- UI flow: Streamlit AppTest loaded the sample, renamed a speaker, and saved captions without exceptions.
- Media: FFmpeg converted a real stereo WAV to 16 kHz mono; corrupt media was rejected.
- Demo: FFmpeg/Flite generated two synthetic voices. Captions/times are predefined sample annotations.
- Adapter: stubbed WhisperX/pyannote/Transformers modules exercised orchestration, no-speech behavior,
  emotion addition, and nonfatal optional emotion failure. No real model inference was performed.
- Source compatibility: WhisperX 3.8.6 wheel inspected for the adapter signatures used by this project.
- Browser visual/playback test: not completed; the browser binary download failed in this environment.
- Real-model smoke test: pending a Hugging Face token, accepted model conditions, and ML installation.
- Supported Python versions: 3.10–3.13, as in the proposal. Python 3.10 uses Streamlit 1.65.0
  (1.66 requires >=3.11); 3.11+ uses Streamlit 1.66.1.
- CI matrix: Python 3.10, 3.11, 3.12, and 3.13; see the Actions tab for results of the compatibility fix.
- GitHub repository: `dipaksinha1/context-aware-captioning`.
- GitHub CI: workflow supplied; inspect the Actions tab for the current run status.

No ASR, diarization, emotion accuracy, or HCI study outcomes are claimed.

CI setup also validates the wheel-bundled FFmpeg executable before running media tests.
The workflow uses Node.js 24 actions (`checkout@v6`, `setup-python@v6`) on Ubuntu 24.04.

## Background sound cues (added after the record above)

Environment for this change: Python 3.13.18, Streamlit 1.66.1, Linux, 4 CPU cores, FFmpeg available.
For the real-model checks only: PyTorch 2.15.0, Transformers 4.57.6, model
`MIT/ast-finetuned-audioset-10-10-0.4593` on CPU.

- Lint passed; 30 tests passed on Python 3.13 locally. Other Python versions: see CI.
- Browser (Chromium via Playwright): loaded the sample; the player showed `[doorbell rings]` alone,
  `[beeping]` together with speech, and nothing in a gap; seek buttons listed both sound cues. In
  Review & edit, typed `[Doorbell chimes]` over a cue and added a `rain falling` row; after saving,
  the project had three sound cues, shown once-bracketed.
- Stubbed detector tests cover windowing, silence skipping, class grouping, and pipeline failure
  handling. They say nothing about detection quality.

### Real-model check (small, not a benchmark)

The detector code in `sounds.py` was run unchanged on recorded audio:

| Check | Threshold 0.20 | 0.25 (default) | 0.30 |
| --- | --- | --- | --- |
| 252 isolated 5 s clips, 42 sound types: expected cue present | 177 (70%) | 170 (67%) | 160 (63%) |
| Same clips: some other cue also shown | 63 (25%) | 44 (17%) | 34 (13%) |
| 8.0 min of read speech with no added sounds: cues shown | 3 | 2 | 2 |
| Six sounds mixed into speech at speech loudness: found / wrong cues | 6 / 0 | 4 / 0 | 4 / 0 |
| Six other clips of the same types at half speech loudness: found / wrong cues | 4 / 0 | 4 / 0 | 4 / 0 |

- Clips: the first six per category from one shard of the ESC-50 copy at `ashraq/esc50`; speech: the
  73 utterances of `hf-internal-testing/librispeech_asr_dummy`. Both converted to 16 kHz mono.
- "Expected cue" is this project's wording for the clip's ESC-50 category, for example `dog` →
  `[dog barking]`. "Other cue" counts near-misses too, such as `[crying]` on a baby crying.
- All six tested clips got the expected cue for dog, door knock, coughing, clapping, siren, cat,
  rooster, toilet flush, and others; rain 5 of 6. None of six did for wind or crackling fire.
- Mixed tracks used dog, door knock, rain, breaking glass, siren, and clapping. Rain and breaking
  glass were the ones missed. At the default, seven of the eight found sounds were cued within
  1 s of their true start; one applause cue started 4 s late.
- The 2 cues on speech-only audio at the default were `[insects buzzing]` and `[coughing]`; whether
  the reader actually coughed was not checked.
- Three vocabulary entries (breaking → `[crash]`, throat clearing → `[coughing]`, plop →
  `[water dripping]`) were added after looking at these same clips, so the first two rows are
  optimistic for new recordings.
- Speed: 0.61 s per two-second window, one window per second of audio.
- The sample recording: the detector returned `[doorbell rings]` at 3.0–5.0 s (the effect is at
  3.4–5.2 s) and `[beeping]` at 8.0–10.0 s (the effect is at 9.5–10.7 s, so the cue was 1.5 s
  early), and no cue for the speech. The cues stored in the sample remain predefined.
- Digital silence: with exact zeros around it, the same doorbell scored 0.06 and was not captioned;
  with a faint noise floor it scored 0.34. The sample therefore carries faint room tone. Audio with
  true digital silence between sounds (some screen recordings, synthetic audio) may lose cues.

Not checked: a door slam (no such recording was available); sound detection together with real
WhisperX and pyannote in one run (WhisperX was not installed here); GPU; music; long recordings.
