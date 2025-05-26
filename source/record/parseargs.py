import argparse

# Shared arguments (not the main parser)
shared_parser = argparse.ArgumentParser(add_help=False)
shared_parser.add_argument("--camera", type=int, help="The camera stream to capture", required=True)
shared_parser.add_argument("--fps", type=int, help="The FPS at which to capture the stream", default=30)
shared_parser.add_argument("--width", type=int, help="The width of the frame to capture", default=1920)
shared_parser.add_argument("--height", type=int, help="The height of the frame to capture", default=1080)

# Main parser (top-level)
main_parser = argparse.ArgumentParser()
subparser = main_parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

# Subcommands
record_parser = subparser.add_parser("record", parents=[shared_parser], help="Record camera stream")
record_parser.add_argument("--output", type=str, required=True, help="The output folder for the videos")

preview_parser = subparser.add_parser("preview", parents=[shared_parser], help="Preview camera stream")

# Parse
args = main_parser.parse_args()
