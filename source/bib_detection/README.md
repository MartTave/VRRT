# Bib detection

This folder contains the solution to the bib-detection problem.

## Data source

The data used in this folder comes from the following sources :

- [Roboflow - BIB_detect](https://universe.roboflow.com/gohar-w4jjb/bib_detect)
- [Roboflow - bib number labeling](https://universe.roboflow.com/vrrt/bib-number-labeling-3z7gm-wekvr) -> this datasets was built using mutliples other data sources, please read the project description itself for more informations

## Installation

This project use [the UV package manager](https://docs.astral.sh/uv/getting-started/installation/).

To run it, you simply have to run `uv run <pythonfile>.py`

If you don't want to use UV. You can install the dependencies present in the `pyproject.toml` files and then run the scripts normally (but you shouldn't).

## Usage

### Dataset management

To annotate a dataset, create a dataset based on video detection or analzyse one, you can look into the files : `annotate.py`, `annotate_video.py`, `inspect_dataset.py`

Launching them with UV. You can refer to their comments to have more details on how to use them.

### Training

To train or fine-tune a model. You can refer to the code in the `train.py` file.

## Folder structure

- `./data` - This folder is a symlink of the `../../data`
- `./models` - This folder is a symlink of the `../../models` folder
- `annotates.py` - This file is used to annotate data with a model resuts. It would be used to add a label to an existing dataset
- `annotates_vidceo.py` - This file is used to annotate a video with some models results. It would be used to merge two models into a single video dataset
- `inference.py` - This file is used as a quick way to test the results of a training.
- `inspect_dataset.py` - This file is used to inspect a dataset to ensure consistancy in the data.
- `train.py` - This file is used to fine-tune a pretrained YOLO model using a local dataset.
