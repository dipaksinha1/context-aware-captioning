# Validation record

Authoring environment: Python 3.12.14, Streamlit 1.66.1, Linux, FFmpeg available.

- Lint: `python -m ruff check .` — passed.
- Test suite: `python -m unittest discover -s tests -v` — 23 tests passed on Python 3.10 and 3.13;
  32 tests (including reference scoring) passed on Python 3.12 after adding `context-caption-eval`.
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
- Real-model run (cloud session, 4 vCPU Xeon 2.1 GHz, CPU only, Python 3.13): torch 2.8.0 (PyPI build),
  WhisperX 3.8.6, faster-whisper 1.2.1, pyannote.audio 4.0.7, Transformers 4.57.6, NLTK 3.10.4.
  Input `examples/test_conversation.mp4`, 41 s, three synthetic Piper voices; WhisperX `base`, int8,
  batch 4, language auto-detected (en, 0.99), pyannote community-1, emotion on.
  - Processing time with cached models: 34–43 s (real-time factor about 0.85–1.05); first run with
    model downloads 58 s.
  - Scored with `context-caption-eval` against the script (collar 0.25 s):

    | Speaker count | WER | DER | Missed / confusion | Word speaker accuracy |
    | --- | --- | --- | --- | --- |
    | Auto-detect | 2.9% | 38.1% | 1.14 s / 9.32 s of 27.48 s | 64.4% |
    | Fixed at 3 | 2.9% | 25.7% | 1.40 s / 5.66 s of 27.48 s | 73.3% |

  - Auto-detection found two speakers and merged the two female voices. With three fixed, errors were
    quick replies after 0.3–0.5 s pauses joined to the previous speaker ("Great. What did you find?",
    "Yes.") and one sentence split across speakers. Word errors: "Good." heard as "Dude,", punctuation.
    Word-timestamp error is not measurable: the reference has utterance times only.
  - Emotion is unreliable as expected: "Yes. Dude," (an ASR error) was labelled disgust (0.77), and a
    misattributed line inherited the previous speaker's joy label.
  - Silent 10 s WAV: zero captions, exit 0. Random-bytes MP4: "FFmpeg could not decode an audio track".
  - Fixes needed to run: Silero VAD downloads from GitHub at run time and failed behind the proxy, so the
    pipeline now uses WhisperX's bundled pyannote VAD; NLTK `punkt_tab` was installed manually (README).
- Browser end-to-end on real models (Playwright + Chromium, Streamlit 1.66.1, file watcher off):
  20 checks passed — upload/consent gating, processing in the UI (34–43 s, 17 cues, 3 speakers, EN),
  media playback with word highlighting, seek, transcript search, speaker rename, a caption correction
  ("Dude," to "Good.") retaining original text and invalidating its alignment, SRT/VTT/JSON export,
  Clear session, JSON reopen with media attached. No browser JavaScript errors.
- Supported Python versions: 3.10–3.13, as in the proposal. Python 3.10 uses Streamlit 1.65.0
  (1.66 requires >=3.11); 3.11+ uses Streamlit 1.66.1.
- CI matrix: Python 3.10, 3.11, 3.12, and 3.13; see the Actions tab for results of the compatibility fix.
- GitHub repository: `dipaksinha1/context-aware-captioning`.
- GitHub CI: workflow supplied; inspect the Actions tab for the current run status.

- Reference scoring: unit tests cover WER counts, DER components, collar, label mapping, edited and
  untimed words, and the CLI; it was applied to the real-model run above.

- Player timing (headless Chromium, full 41 s of the test video): with `timeupdate`-only rendering,
  78 of 103 timed words were ever highlighted and highlights lagged word starts by a median 116 ms
  (max 250 ms). Rendering on animation frames while playing highlighted 103 of 103, median lag 8 ms
  (max 17 ms).
- Four-voice podcast `examples/podcast_policy.mp4` (110 s, Piper voices amy/ryan/alba/alan, animated
  illustrated presenters, two overlapping lines, one line raised +7 dB, one aside −8 dB, one line −6 dB,
  one +5 dB), WhisperX `base`, CPU, emotion on, about 95 s to process:

  | Speaker count | WER | DER | Missed / confusion | Word speaker accuracy |
  | --- | --- | --- | --- | --- |
  | Fixed at 4 | 3.9% | 13.0% | 5.08 s / 6.11 s of 85.96 s | 90.9% |
  | Auto-detect | 3.9% | 13.0% | same (4 speakers found) | 90.9% |

  Word errors are mostly names and compounds: "Riverton's" → "Rivertons'", "Elena" → "Elina",
  "car per" → "corporate", "burden" → "bird", "Roundtable" → "Round Table". Crosstalk: Marcus's "Exactly."
  over Tom was attributed correctly; Dana's quiet "Right." under Elena was attributed to Elena. Elena's
  "Really? …" and "Maybe. …" lines went to Dana, and one of her sentences switches to Dana midway. Emotion
  cues: 30 neutral, 5 fear, 4 surprise, 1 sadness, 3 below threshold. An earlier take of the same script
  without volume changes scored 4.9% WER, 10.2% DER, 95.2% word speaker accuracy.
- Captions on the video (Chromium, podcast video in the player): the stage matched the picture
  (604×340); captions stayed inside the picture 62 px above the bottom, hid with no active cue and with
  CC off; "Fullscreen with captions" made the stage the fullscreen element with captions re-anchored to
  the letterboxed picture and text scaled up (27.5 px at 860 px wide); F left fullscreen; highlighting
  continued during playback; no JavaScript errors.
- Word loudness: unit tests check levels against sine waves of known amplitude (0.5 → −9.0 dBFS,
  0.05 → −29.0 dBFS), silence, untimed words, size scaling and clamping, the player toggle, JSON round trip
  and older projects; the stubbed pipeline test checks a silent word (−90 dBFS floor) and a −12 dBFS tone.
  On the podcast, the median word is −23.2 dBFS; Elena's raised line renders at 1.07–1.30× (−15 to
  −21 dBFS, one unstressed word at 0.80×) and her soft line at 0.75–1.0× (−23 to −35 dBFS). In Chromium,
  hovering shows each level, the toggle returns all words to 1.0×, and attaching media to the sample project
  (saved without levels) measured them (−13 to −20 dBFS). No JavaScript errors.
- Demo videos (1280×800, recorded with `scripts/record_app_demo.js --config examples/<name>.demo.json`;
  audio aligned by detecting highlight changes in the recorded frames; the script verifies the caption
  correction lands on the intended row and that the app's output matches the stored model output):

  | Video | Length | Processing in UI | Highlight vs. audio word start (median, p10, p90) |
  | --- | --- | --- | --- |
  | `docs/demo/podcast_demo.mp4` | 3:23 | 108 s (shown 10×) | 27 ms, 2 ms, 51 ms over 285 changes |
  | `docs/demo/app_demo.mp4` | 2:03 | 38 s (shown 8×) | 45 ms, −36 ms, 90 ms over 110 changes |

  Both were recorded with loudness-sized words; their exported projects carry `level_db` for every timed word.

  A take right after a container restart took 193 s to process (cold caches) and was discarded.
- Emotion emoji: unit tests cover SRT/VTT output and unknown labels; in Chromium with Noto Color Emoji,
  the player label ("Raj · 😄 joy") and transcript lines rendered with no JavaScript errors.

Results above come from one synthetic recording and do not establish accuracy on real conversation.
No HCI study outcomes are claimed.

CI setup also validates the wheel-bundled FFmpeg executable before running media tests.
The workflow uses Node.js 24 actions (`checkout@v6`, `setup-python@v6`) on Ubuntu 24.04.
