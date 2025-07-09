# VRRT - Video-base Real-time Race Timing

This repository contains the solution for the VRRT bachelor thesis.

## Structure

This repository is structured  as follow :

- `data` -- This folder contains all the data necessary to do the project. For now, the data is not open source, this can change in the future
- `models` -- This folder contains all the ultralytics YOLO models. Some downloaded as is from their website, some fine-tuned for the task
- `results` -- This folder contains all the results for the project. For now, the data is not open source, this can change in the future
- `source` -- This folder contains all the sub-projects. Each sub-project contains a README with more details
- `tools` -- This folder contains the scripts created during this project
- `trackers` -- This folder contains the different trackers configuration used in pair with the YOLO model

The `data`, `models`, `results` and `trackers` are linked in the subfolder to allow sharing of folders between the differents subfolders

## Installation

For the installation of the sub-projects, you can refeir to their README. But they all use the [UV](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Sub-folders

### Bib detection

The `bib_detection` sub folder contains all the code for fine-tuning a YOLOv11 model, completing and inspecting a YOLOv11 dataset.

### Stereo

The `stereo` sub folder contains all the code for calibrating a stereo setup, rectifying images, estimating depth using diverse algorithms.

### Record

The `record` sub folder contains all the code for recording video using a single camera or a stereo setup. This include a 'best-effort' syncronization between two cameras for a makeshift stereo setup.

This is the code used to gather most of the data used in this project.

### Project

The `project` sub folder is the main project folder. This contains all the final pipeline, using the byproduct of the others subfolders (the data from `./record`, the fine-tuned model from `./bib_detection`)
