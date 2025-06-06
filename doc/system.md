# System overview


## Person detection

The first part of the system is the person detector. For this, I had many choices with similar performances and results. The main differences were ease of use and ease of installation. The solution I found pretty quickly was [Ultralytics YOLOV11](https://github.com/ultralytics/ultralytics). This is more than a model in the sense that it is a library that alllows for quick inference, many pre-trained model, an easy way to do fine-tuning (which will come in handy in the bib detection segment)

## Bib detection

## Bib number reading

## Race finish timing

The goal of this part of the solution is to have a way to detect when a person cross a predefined line. In order to do this, the solutions explored are the following :
- Monocular Depth estimation
- Multi-camera structure from motion
- Standard stereo camera setup

## Monocular Depth estimation

This solution involve estimating the depth map of an scene using a single picture. This normally involve deep learning, as the features needed to acheive this are too complicated for a standard approach.

This type of solution can give phenomenal results like [Depth pro](https://github.com/apple/ml-depth-pro) from Apple or [DepthAnythingV2](https://github.com/LiheYoung/Depth-Anything). Those models would be a good options, but they require pretty heavy hardware in order to give accurate results, in the sense that many models that gives depth estimation exists, in all form factor (for real time applications, for high precision, large, medium, small, etc...)
So it seems that the sweet spot between precision and inference speed does not exists for my use case. As in order to achieve meaningfull precision, we need to infer the full solution (not only this model) to at least ~10FPS. So this solution is not viable for now, but might be in the very near future.

## Multi-camera structure from motion

This solution seems to be the most powerfull one, as it can be pretty precise, adjustable (in the sens that more cameras mean more precision). But it is notoriously hard to setup, and it may be too hard for my use case. So it may be a future upgrade path, but for now it will not be explored more
