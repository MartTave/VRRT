# Stereo

This folder contains everything related to the stereo vision part of the project.

This part was firstly planned to be used in the depth estimation part of the project, but it was replaced with monocular depth estimation models for multipes reasons described in the reports.

The folder is still here because one might want to use the stereo depth estimation instead of the monocular one under certains conditions.


## Installation

This folder use the [UV package manager](https://docs.astral.sh/uv/getting-started/installation/).
So to run the project, you can use the command : `uv run <pythonfile>.py`

## Usage

### Taking pictures

To record picture for the stereo calibration, you can use : `uv run take_picture.py`

When you see the preview, you can press :

- `s` (as in stereo) to take a picture from both cameras at the same time and save them in the corresponding folder
- `l` (as in left) to take a picture from the left camera and save it in the corresponding folder
- `r` (as in right) to take a picture from the right camera and save it in the corresponding folder
- `a` (as in all) to take a picture from both camera at the same time, and save it in the corresponding folders
- `q` to quit

### Calibration

To calibrate a stereo setup. First take some pictures by using the script described above. And then run `uv run calibration.py`

This script will by default read all the pics present in the stereo folder. Find the charuco board corners, calibrate each cameras using all the points detected.
Then it will attempts to calibrate the stereo setup using only the points present in both images for each picture.

The result will be saved under the `calibrations` folder.

### Rectification

If you want to rectify a set of pictures. You can use the `rectify.py` file, or take inspiration from it.


### Depth map

To get a depth map, you can use the `depth_map.py` file.

This file is not final, it is mainly a helper to try to find the right parameters for the stereo setup and the scene.
