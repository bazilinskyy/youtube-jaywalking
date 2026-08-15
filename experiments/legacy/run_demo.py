# run_demo.py  (throwaway script, not part of the pipeline)
import argparse
from helper_script import Youtube_Helper


def main():
    parser = argparse.ArgumentParser(description="Run jaywalking detection demo")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS (default: 25)")
    parser.add_argument("--title", default="demo", help="Video title for CSV output")
    args = parser.parse_args()

    helper = Youtube_Helper()
    helper.set_video_title(args.title)
    helper.tracking_mode(
        input_video_path=args.input,
        output_video_path=args.output,
        video_fps=args.fps,
    )


if __name__ == "__main__":
    main()
