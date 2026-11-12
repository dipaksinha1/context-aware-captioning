# Architecture and proposal coverage

The prototype follows the seven-step workflow in the supplied COSC-5926 proposal.

| Proposal step | Implementation | Review evidence |
| --- | --- | --- |
| Upload/preprocessing | `media.normalize_audio` converts to 16 kHz mono PCM; bounded file size/duration | FFmpeg integration tests |
| Speech transcription | `pipeline.process` loads WhisperX and transcribes in batches | Adapter contract test; real-model smoke test pending |
| Speaker diarization | WhisperX `DiarizationPipeline` wraps pyannote community-1 | Adapter test; real-model quality pending |
| Word/speaker alignment | Forced alignment followed by `assign_word_speakers`; cues split by word speaker | Core speaker-turn and untimed-word tests |
| Background sounds (added after the proposal) | `sounds.score_windows` tags half-overlapping 2 s windows with the AudioSet AST model; `build_sounds` turns scores into bracketed cues | Cue-building tests; stubbed detector test; see VALIDATION for the real-model check |
| Optional emotion | English DistilRoBERTa classifier run per speaker turn (consecutive same-speaker cues); separate cue metadata; confidence threshold | Optional branch adapter test; quality pending |
| Synchronized display | Streamlit and an embedded HTML player using the same media clock for cues and words | UI flow test; interactive browser verification pending |
| Subtitle export | Speaker-prefixed SRT, styled VTT, and JSON for reopening | Serialization, escaping, and CLI export tests |

## Data flow

1. The browser sends an uploaded recording to the local Streamlit process.
2. A temporary file is passed to FFmpeg; only its normalized waveform enters the ML pipeline.
3. WhisperX produces transcript segments and the detected language.
4. Forced alignment produces word times and optional alignment confidence scores.
5. pyannote clusters speaker turns; WhisperX assigns speakers by temporal overlap.
6. The cue builder groups contiguous words, splitting at speaker changes and length targets.
7. The optional sound tagger scores the same waveform in two-second windows. Scores for a curated
   vocabulary become separate sound cues; the transcript is not touched. This runs even when no speech
   was found.
8. The optional emotion model classifies each speaker turn (consecutive cues by one speaker) and copies
   the label to that turn's cues in separate fields; it never rewrites transcript text.
9. The project is stored in session memory. Edits are applied to a deep copy and validated atomically.
10. Exports serialize reviewed cues, merging speech and sound cues in time order. JSON additionally
    retains original words, original caption text, and original sound wording with its score.

## Deliberate decisions

**UNKNOWN labels:** no nearest-speaker filling is enabled. When evidence is absent, the interface
shows UNKNOWN rather than presenting a guessed identity as established.

**Original alignment:** edits do not attempt to estimate new word timestamps. Word highlighting is
disabled after text/time edits; the old alignment remains an inspectable record.

**Emotion:** a text classifier is an approximation of sentiment expressed in words. It cannot reliably
infer sarcasm, intonation, or feelings. It is English-only and optional; its failure does not discard captions.

**Sound cues:** sounds are a separate list, not captions with a pretend speaker, so speaker naming,
emotion turns, and caption edits are unaffected. Text is stored bare (`dog barking`) and bracketed in one
place, `format_sound`, so every surface agrees. Only allow-listed AudioSet classes produce cues: an
unlisted class is ignored instead of shown under a raw label. AudioSet is multi-label, so scores are
independent sigmoids and at most two sounds are kept per window. A sound heard in consecutive overlapping
windows is placed in the part those windows share, which is tighter than their union. A cue marks the
onset: it is held 1.5–6 s, and repeats of the same sound within three seconds join one cue. Near-silent
windows are skipped. Detection failure leaves captions intact, like emotion.

**Resource use:** ASR and alignment models are released before the next heavy stage. CPU defaults to int8,
CUDA to float16. Models are loaded per run rather than globally cached with a user's access token.

**Playback:** a single HTML media element drives all active cues, including simultaneous cues. Labels
accompany colors, keyboard-accessible seek buttons jump to cue starts, and gaps clear captions.
All transcript and speaker strings are inserted using DOM `textContent`; embedded JSON escapes HTML
script delimiters. VTT text also escapes markup.

## Deferred production work

Background jobs/cancellation; model dependency locks for each supported platform; streaming and long-file
chunking; durable project database; authentication; real speech benchmark evaluation; accessible usability
study; overlapping-speaker presentation research; timestamp re-alignment after edits; sound-event benchmark
evaluation, finer sound onset timing, and wording that names the source of a sound; and scalable media
serving instead of base64 embedding. None is claimed as implemented.
