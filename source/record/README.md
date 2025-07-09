# Record

This folder contains all the code used in this project to record videos.

## Installation

This folder use the [UV package manager](https://docs.astral.sh/uv/getting-started/installation/).

To run it, you can use the command `uv run <python file>.py`

## Usage

The main usage of this folder is to record video. To do so, you can use : `uv run record.py <record/preview> <camera_id>`

To get more detail on the usage of the script, you can use the command `uv run record.py --help`


By default, it will record by step of 5 minutes per files, in order to avoid data loss in case of catastrophic failure.

If the program crash, the last file will be finished cleanly before exiting.

This script will saves a file in parallel to the video containing the timestamps at which each frame is recorded, this is done in order to simulate a live recording later.

## Folder structure

- `record.py`: Main script to record video.
- `sync.py`: *!NOT FINISHED!* This script is an early try to syncronized recording of two USB cameras. For now, it does not work higher than 15FPS.
- `sync_and_merge.py`: This scripts contains all the functions needed to merge mutliples small videos file (and their timestamps) into a single big one
- `test.linuxpy.py`: *!NOT FINISHED!* This script is a test to record sychronized video feeds from two USB cameras using linuxpy. It does not work for now.
- `test.html`: This is a simple file to show a clock on a screen in order to test syncronization of USB cameras. You need a high refresh rate screen to test this.
- `record_timestamp.py` - This is a helper file to record timstamp on keypress. It was used in order to generate data for comparaison with the automatic timing system
