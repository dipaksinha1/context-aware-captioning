import argparse
import json
import os
from pathlib import Path

from .exporters import to_json, to_srt, to_vtt
from .schema import Project
from .sounds import THRESHOLD


def main(argv=None):
    parser = argparse.ArgumentParser(description="Speaker-aware captions from local media")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Audio/video recording")
    source.add_argument("--project", type=Path, help="Re-export an existing project JSON")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--language", help="ISO code; default auto detection")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--emotions", action="store_true")
    parser.add_argument("--sounds", action="store_true",
                        help="Detect background sounds and caption them in brackets")
    parser.add_argument("--sound-threshold", type=float, default=THRESHOLD,
                        help="Confidence needed for a sound cue (default %(default)s); "
                             "lower finds quieter sounds but adds more wrong cues")
    args = parser.parse_args(argv)
    try:
        if args.project:
            project = Project.from_dict(json.loads(args.project.read_text(encoding="utf-8")))
        else:
            from .pipeline import Settings, process
            project = process(args.input, Settings(
                model=args.model, device=args.device, language=args.language,
                min_speakers=args.min_speakers, max_speakers=args.max_speakers,
                emotions=args.emotions, sounds=args.sounds,
                sound_threshold=args.sound_threshold, token=os.getenv("HF_TOKEN", ""),
            ), lambda _, message: print(message))
        args.output.mkdir(parents=True, exist_ok=True)
        for suffix, content in {
            "json": to_json(project), "srt": to_srt(project, args.emotions),
            "vtt": to_vtt(project, args.emotions),
        }.items():
            (args.output / f"captions.{suffix}").write_text(content, encoding="utf-8")
        print(f"Exported {len(project.captions)} captions and {len(project.sounds)} "
              f"sound cues to {args.output}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
