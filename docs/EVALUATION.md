# Evaluation and class demonstration

## Current validation boundary

Local checks cover deterministic logic, real FFmpeg conversion, stubbed ML orchestration, and the
Streamlit sample workflow. They do not establish ASR word error rate, diarization error rate, emotion
accuracy, real-model runtime, or accessibility conformance. No measured model results are reported.

## First real-model smoke test

1. Install the ML extra, FFmpeg, and accept the gated model conditions as described in the README.
2. Record a 30–60 second WAV with two consenting adults, alternating turns in a quiet room.
3. Write a manual reference transcript with speaker names and approximate word boundaries.
4. Process on CPU with `tiny` or `base`, fixed speaker count 2, emotion disabled.
5. Verify that both speakers are represented and retain their labels when returning to the conversation.
6. Inspect Word timestamps. Verify captions near the start, middle, and end by listening and seeking.
7. Correct one caption and rename both speakers. Save JSON, reopen it, and compare exported subtitles.
8. Enable emotion on an English sample separately. Verify that transcript content is unchanged.
9. Enable sound detection on a recording with a known sound at a known time (a door, a dog, rain).
   Check the cue wording, that it is bracketed and separate from speech, and how far its start is from
   the sound. Note sounds that were missed and cues for sounds that are not there.
10. Repeat with no speech and with an invalid/corrupted media file. Confirm errors or empty results are clear.
11. Record package versions, hardware, model names, settings, media duration, and elapsed processing time.

Keep actual recordings and tokens out of the repository. Use `python -m pip freeze > outputs/environment.txt`
in the active environment to retain the exact dependencies used for your evaluation.

## Suggested test matrix

| Condition | Purpose |
| --- | --- |
| Two speakers, clean alternating turns | Baseline consistency and timestamp checks |
| Three speakers | Label count and re-entry consistency |
| Short utterances and rapid turns | Cue boundaries and ambiguous speaker attribution |
| Overlapping speech | Identify limitations rather than assuming complete recovery |
| Background noise | ASR and diarization robustness |
| Distinct sounds under and between speech | Sound-cue wording, timing, misses, and false cues |
| Speech only, quiet room | Sound detector should add no cues |
| Numbers, currencies, and proper nouns | Untimed-token handling and transcription errors |
| Non-English recording | Alignment availability and emotion skipped status |
| Silence | Empty transcript behavior |

## Metrics to measure

- **Word error rate:** `(substitutions + deletions + insertions) / reference_word_count` after documenting
  normalization of punctuation, casing, and numbers. Evaluate against a manually checked reference.
- **Diarization error rate:** `(missed_speech + false_alarm_speech + speaker_confusion) / reference_speech`.
  Use a standard library such as pyannote.metrics; declare overlap handling and scoring collar.
  Map arbitrary predicted labels to reference speakers before comparisons.
- **Timestamp quality:** median and 95th-percentile absolute word-start and word-end error on aligned
  matched words; report untimed-word rate separately rather than excluding failures silently.
- **Sound cues:** event-based precision, recall, and F1 against hand-marked sound events, stating the
  onset tolerance used (for example ±1 s); report false cues per minute on speech-only recordings.
- **Runtime:** wall-clock processing seconds / recording seconds (real-time factor), with hardware noted.
- **Editing effort:** corrections per minute of media, task completion time, and number of speaker relabels.

The proposal does not prescribe a benchmark dataset. Begin with consented local recordings for a smoke
test; select a licensed multi-speaker benchmark and document its source and scoring protocol before
claiming general accuracy. Training/validation/test splits are unnecessary for this inference-only demo;
keep any samples used to tune thresholds separate from final evaluation recordings.

## HCI study plan

Recruit volunteers with informed consent; include Deaf/hard-of-hearing users if appropriate for the
course and its research requirements. Do not claim a user study has been performed.

Compare ordinary captions with speaker-aware captions using the same recording. Counterbalance the
order. Ask participants to identify who spoke, locate a sentence, correct a word, rename a speaker,
and export subtitles. Record success, errors, time, perceived readability, and whether emotion cues
and bracketed sound cues helped or confused understanding. Make emotion optional and avoid using color as the sole identifier.

## Five-minute class walkthrough

1. Explain the accessibility goal and how transcription differs from diarization.
2. Load the synthetic sample and demonstrate speaker colors, synchronized cues, word highlighting,
   and the bracketed `[doorbell rings]` and `[beeping]` sound cues.
3. Rename a speaker; edit text and explain why its original alignment is retained separately.
4. Export SRT/VTT/JSON, then reopen the JSON and attach audio.
5. Show a separately processed real recording if model access is ready; label demo versus model results.
6. Discuss overlap, noise, uncertain emotion, and the evaluation plan.
