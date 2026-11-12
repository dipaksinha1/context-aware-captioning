# Context Caption

**Context-aware captioning for multi-speaker media**

COSC-5926 — Advanced Human-Computer Interaction

A local Streamlit prototype that combines WhisperX transcription and forced word alignment,
pyannote speaker diarization, optional bracketed background-sound cues such as `[dog barking]`,
and optional transcript-based emotion cues in an editable caption workflow.

## What you can do

- Upload audio/video; convert it to 16 kHz mono audio using FFmpeg.
- Transcribe speech and inspect original word timestamps and alignment scores.
- Split captions on word-level speaker changes, with consistent speaker labels and colors.
- Play media with synchronized captions, highlighted words, and click-to-seek controls.
- Rename speakers, edit text/timing/speaker assignments, and search the transcript.
- Optionally detect non-speech sounds — a door slamming, rain, a dog barking — and caption them
  in brackets, apart from speech: `[door slams]`, `[rain falling]`, `[dog barking]`.
- Correct, add, or delete sound cues by hand, with or without the detector.
- Optionally estimate English text emotion per speaker turn without modifying spoken content.
- Export speaker-aware SRT, styled VTT, or a JSON project retaining original words and edits.
- Reopen a JSON project and reattach its original media.
- Try a predefined two-voice synthetic demo without downloading ML models or providing a token.

## Start with the demo

The package supports Python 3.10–3.13, as specified in the project proposal. Python 3.11+ installs
Streamlit 1.66.1; Python 3.10 installs Streamlit 1.65.0, the last release supporting 3.10.
CI checks all four supported Python versions. Python 3.12 is the simplest match to local validation.

```bash
cd context-caption
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m streamlit run app.py
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with
`.venv\Scripts\Activate.ps1` in PowerShell instead of the `source` command.

Open `http://localhost:8501` and select **Load sample project**. The supplied WAV contains
synthetic voices from FFmpeg's Flite filter. Its captions and word times are predefined for UI testing:
**they are not results of ASR or diarization**. The sample also contains a synthesized doorbell and a
synthesized beep with predefined `[doorbell rings]` and `[beeping]` cues, to show how sound cues look beside
speech; they are not detector output either. Regeneration is optional; see `examples/create_demo.py`.

## Enable actual transcription and diarization

1. Install FFmpeg and confirm `ffmpeg -version` works.
   - macOS with Homebrew: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install ffmpeg`
   - Windows: use an FFmpeg build linked from <https://ffmpeg.org/download.html> and add its `bin` to PATH.
2. In your active virtual environment, run:

   ```bash
   python -m pip install -e '.[ml]'
   ```

   This installs WhisperX 3.8.6, including its compatible PyTorch/pyannote dependencies, and Transformers.
   The demo UI and core tests do not require this extra. A resolved cross-platform ML lockfile is not
   included; native wheels depend on operating system and GPU. Keep this environment separate from
   existing MLX/LoRA installations.
3. Sign into Hugging Face and accept the access conditions at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>.
4. Create a read token with permission to access that model. Enter it in the sidebar or set `HF_TOKEN`
   in your shell. Do not paste tokens into GitHub, captions, notebooks, or the README.
5. Start the app. Upload a short, clear recording with two consenting speakers, confirm permission,
   select **CPU / base / batch size 4**, and click **Generate captions**.
6. Review names, text, and times before exporting. Model downloads require internet on first use.

macOS, including Apple Silicon, uses the **CPU** backend here. This project does not use MLX or MPS.
For NVIDIA GPUs, install a PyTorch/CUDA runtime supported by your WhisperX installation before selecting
CUDA. Use the official [WhisperX setup guide](https://github.com/m-bain/whisperX#setup-) for platform details.
Start with `tiny` and batch size `1` if memory is limited. Allow 10–20 GB for dependencies and model caches.

## Background sound cues

Tick **Detect background sounds** in the sidebar (or pass `--sounds`) to caption sounds that are not
speech. The format follows subtitling practice for Deaf and hard-of-hearing viewers (SDH):

```text
4
00:00:12,000 --> 00:00:13,500
[door slams]

5
00:00:13,200 --> 00:00:16,900
Alex: Who's there?

6
00:00:15,000 --> 00:00:21,000
[dog barking]
```

- **Square brackets** mark a sound. Speech never starts with a bracket; it starts with a speaker name.
- **Lowercase, present tense.** A sustained sound takes the *-ing* form (`[rain falling]`,
  `[dog barking]`); an abrupt one takes the plain verb (`[door slams]`, `[glass shatters]`).
- **Its own cue.** A sound is a separate timed cue, never mixed into someone's words, so it can overlap
  speech and can be turned off for playback or export without touching the transcript.
- **Shown when it starts.** A cue marks the onset and stays up for 1.5–6 seconds. Rain that lasts
  five minutes is announced once, not held on screen; a sound that returns after a pause of more
  than three seconds is announced again.

Detection uses the public [Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
trained on AudioSet. It needs the ML extra but no token; the first run downloads about 350 MB.
Only a curated list of 80 sound types is captioned (`VOCABULARY` in `sounds.py`), each with
fixed wording. Speech classes are excluded on purpose.

Expect a first draft, not a finished track. In a small check on recorded clips
([docs/VALIDATION.md](docs/VALIDATION.md)), about two in three isolated sounds got the expected cue
and about one clip in six also got a wrong one; distinct sounds such as a dog, a knock, a siren, or
applause were found under speech, while rain and breaking glass under speech were missed at the
default setting. The **Sound confidence needed** slider (`--sound-threshold`, default 0.25) trades
missed sounds against wrong cues. On a four-core CPU, detection took about 0.6 seconds per second
of audio.

Every sound cue can be edited under **Review & edit → Background sounds**: change the wording or
times, delete a wrong cue, or add a row for a sound the detector missed, then **Save caption edits**. Type the description
without brackets; they are added on display and export. The original detected wording and its
confidence stay in the project JSON.

## Command line

After installing the ML extra and setting `HF_TOKEN`:

```bash
context-caption --input recording.mp4 --device cpu --model base --min-speakers 2 --max-speakers 2 --output outputs
```

Add `--sounds` for bracketed background-sound cues (`--sound-threshold 0.2` to catch quieter
sounds), `--emotions` for English text emotion
estimates, or `--language en` to fix the language.
The token is read from the environment rather than a CLI argument.

Re-export the supplied JSON without ML models:

```bash
context-caption --project examples/demo_project.json --output outputs/demo
```

This produces `captions.srt`, `captions.vtt`, and `captions.json`. Outputs are ignored by Git.

## Project structure

```text
app.py                         Streamlit upload/review/export interface
src/context_caption/
  schema.py                    Project, caption, and word data structures
  media.py                     Bounded FFmpeg conversion
  pipeline.py                  WhisperX → alignment → pyannote → optional emotion
  captions.py                  Speaker-aware cue building and atomic edits
  sounds.py                    Background-sound detection, cue wording, and sound-cue edits
  player.py                    Synchronized HTML media player
  exporters.py                 SRT, VTT, and JSON exports
  cli.py                       Local processing and project re-export CLI
examples/                      Synthetic demonstration project and audio
tests/                        Core, media, pipeline, and Streamlit tests
docs/                          Architecture, evaluation, and GitHub setup
.github/workflows/ci.yml        Automated lint and tests
```

## Checks

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m unittest discover -s tests -v
```

Tests exercise caption behavior, sound-cue building and formatting, import/export, security escaping,
media conversion, the adapter with stubbed ML modules, and the Streamlit editing flow. **Stubbed ML tests do not validate model quality**.
The optional real-model smoke test in [docs/EVALUATION.md](docs/EVALUATION.md) needs your model access.
CI installs the base/dev package, not GPU dependencies or gated models.
It uses Ubuntu 24.04 and an FFmpeg executable bundled in the `imageio-ffmpeg==0.6.0` wheel,
so media tests do not depend on Ubuntu package-mirror speed. Local users should still install
FFmpeg as described above.

## Important behavior and limits

- Diarization labels cluster voices within one recording. They do not establish real-world identities
  or link people across recordings. Manual speaker names are review annotations.
- Overlapping speech, accents, noise, short utterances, and rapid turns can reduce quality.
- A word without alignment remains explicitly untimed (`null`). It is not silently dropped or assigned
  a fabricated timestamp. Untimed groups use the original segment bounds for display and need review.
- Cue length targets are 84 characters and six seconds. A single long token or fallback segment can
  exceed those targets; this is a prototype, not a broadcast subtitle compliance validator.
- Editing text or cue times disables word highlighting for that cue; original word timings remain
  available in JSON and the Word timestamps tab. Text edits clear stale emotion predictions.
- Emotion is inferred from English text, not vocal prosody or a person's internal state. Consecutive
  cues by the same speaker are classified together as one speaker turn, so short cues share their turn's
  label. Only predictions at confidence >= 0.60 are shown; low confidence stays unlabeled. Emotion is
  hidden by default. Non-English recordings skip emotion and report that status.
- Sound cues come from an audio tagger scoring two-second windows, so cue times are approximate
  and quiet, brief, or unusual sounds can be missed or misnamed. The wording is fixed per sound
  type and does not say who or what made the sound. Similar sounds are confused: wind or
  thunder can be captioned `[explosion]`, a vacuum cleaner `[aircraft overhead]`. Sounds
  surrounded by exact digital silence, as in some synthetic audio, score poorly. Detection adds
  about 0.6 s of CPU time per second of audio.
  Captioning guides ask for sounds that matter to the story, not every sound: delete cues that
  do not help. Sound detection is off by default.
- Speaker colors come from a six-color palette. With more than six speakers, colors repeat; the
  speaker name is always shown with the color, so labels stay distinguishable.
- Words that diarization cannot attribute are labeled UNKNOWN rather than guessed from the nearest
  speaker. Assign them in Review & edit.
- SRT uses plain speaker names. VTT adds color classes; external players differ in styling support.
  Sound cues carry no speaker name or color in either format.
- Project JSON is schema version 2 (adds `sounds`). Version 1 files still open; projects saved by
  this version do not open in older copies of the app.
- Upload limit: 200 MiB; processing limit: 60 minutes. FFmpeg conversion times out after five minutes.
  Model inference is synchronous and may take much longer on CPU. There is no background queue/cancel.
- Playback embeds media in the local app and uses extra memory. Prefer short clips for class demos.
  Caption edits or other reruns reset playback position.
- Uploaded/converted files use temporary directories and are removed after processing. Media bytes and
  captions remain in the current session until cleared; save project JSON and media separately to resume.
  Model caches persist locally. No analytics or application-level external media API is included;
  third-party model libraries have their own download/telemetry behavior.
- No authentication or multi-user isolation is provided. Bind to localhost; this prototype is not ready
  to expose to the public internet.

## GitHub

Repository: **[dipaksinha1/context-aware-captioning](https://github.com/dipaksinha1/context-aware-captioning)**.

```bash
git clone https://github.com/dipaksinha1/context-aware-captioning.git
cd context-aware-captioning
```

Authenticate with your GitHub account if the repository is private.
See [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md) for development and upload instructions.
The repository excludes access tokens, uploaded recordings, and downloaded model weights.

## References

### Literature (from the project proposal)

1. Ubur, S. D., et al. (2026). Speaker-Aware Affective Captioning for Multi-Speaker STEM Talk in
   Inclusive Classrooms. *CHI EA 2026*. <https://doi.org/10.1145/3772363.3798718>
2. Amorim, J., Pimentel, J., & Rudzicz, F. (2026). On the Limitations of Speaker Diarization.
   *Expert Systems, 43*(3), e70221. <https://doi.org/10.1111/exsy.70221>
3. Ubur, S. D., et al. (2025). Interpretive Caption: Real-Time Vocal Emotion Cues for DHH Users.
   *ASSETS 2025*. <https://doi.org/10.1145/3663547.3759697>
4. McDonnell, E. J., et al. (2024). "Caption It in an Accessible Way That Is Also Enjoyable."
   *CHI 2024*. <https://doi.org/10.1145/3613904.3642177>
5. de Lacerda Pataca, C., et al. (2024). Caption Royale: Exploring the Design Space of Affective
   Captions from the Perspective of Deaf and Hard-of-Hearing Individuals. *CHI 2024*.
   <https://doi.org/10.1145/3613904.3642258>
6. Bain, M., Huh, J., Han, T., & Zisserman, A. (2023). WhisperX: Time-Accurate Speech Transcription
   of Long-Form Audio. *INTERSPEECH 2023*. <https://doi.org/10.21437/Interspeech.2023-78>

### Software and models

- [WhisperX official repository and Python API](https://github.com/m-bain/whisperX)
- [pyannote.audio official repository](https://github.com/pyannote/pyannote-audio)
- [Speaker diarization community model](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [English emotion model card](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- [Audio Spectrogram Transformer, AudioSet model card](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

### Caption style for sounds

- [DCMP Captioning Key: sound effects and music](https://dcmp.org/learn/602) — bracketed lowercase
  descriptions; *-ing* form for sustained sounds, plain verb for abrupt ones.
- [Netflix Timed Text Style Guide: brackets for sound effects](https://backlothelp.netflix.com/hc/en-us/articles/115000239632-Timed-Text-Style-Guide-Supplementals-Marketing-Assets)
- [Streamlit documentation](https://docs.streamlit.io/)

The implementation uses these existing models; it does not train a new speech recognition model.
Model licenses and access conditions apply separately. No license for redistribution of this project's
original source has been selected yet; choose one before sharing it publicly.
