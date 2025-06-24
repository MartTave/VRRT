This page will contains research and results found leading to the multiples decision taken during the project. It is split between the different parts of the final solution

# Person detection

I went quickly for the yolo model serie, for obvious advantages like :
- Ease of use
- Many model size
- Fine tuning possible
- Good results
- tracking and re-identification integrated

# Bib detection
For the bib detection, I found many examples, some old, some more recent ones.

[oldest](https://people.csail.mit.edu/talidekel/RBNR.html) from MIT. Good results, but no deep learning, so less flexibility.
[As old](https://github.com/gheinrich/bibnumber), seems to be globally the same pipeline as the MIT one

Many solutions used a fine-tuned yolo model :
- https://github.com/Lwhieldon/BibObjectDetection
- https://github.com/ericBayless/bib-detector

I decided to try to fine tune a yolo model to detect bib number. My objective is to combine bib and person detection in a single model to reduce inference time

# Bib reading
For the reading of the bib number, the simplest and quickest solution was to use a OCR engine like tesseract. This engine will be fed the cropped region containing a bib. It will then read the text, and some post-processing will be applied in order to reduce the false detections.

# Depth estimation
This part was the most challenging one to choose the solution for, as some solutions needed specific hardware, which made pivoting harder. So more research and testing was needed before taking a decision.

Possible solutions :
Monocular depth estimation
Classic stereo vision
Deep learning matching for stereo vision
Epipolar geometry.

At first, deep look into monocular depth estimation -> simplest, could give good enough results without too much hassle.

Moncular depth estimation : 
- https://github.com/dwofk/fast-depth
- https://github.com/LiheYoung/Depth-Anything
- https://github.com/Ecalpal/RT-MonoDepth
- https://github.com/CompVis/depth-fm
- https://github.com/apple/ml-depth-pro
- https://github.com/zhyever/PatchRefiner
- https://github.com/isl-org/MiDaS
- https://arxiv.org/abs/2405.10885

This is just a sample, it is exploding. Either too unprecise, too slow or I just didn't succeed in implementing it. For now, the sweet spot between precision and performance is not good enough for our use case. But it is very close. Maybe in a few years, this will be good<

Deep learning stereo matching :
- https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching
- https://github.com/gangweiX/CGI-Stereo
- https://github.com/NVlabs/FoundationStereo/
- https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_stereo_depth?tab=readme-ov-file
- https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation
- https://github.com/megvii-research/CREStereo
- https://github.com/kbatsos/Real-Time-Stereo

This is more promising, to do more testing, I needed the stereo bar setup in order to have some data that matches the scene that my project will be used on, as the results seems to depend a lot on the scene filmed.


# Problems encountered

Finetuning -> no dataset containing both bib number and person. So one model to do both would need data labeling

Stereo vision -> Synchro of dual USB camera is very hard/impossible
