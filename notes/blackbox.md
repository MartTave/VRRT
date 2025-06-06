# 02.06.25
Main goal currently is to find a way to make the timing work.

For now I have explored :
- [Foundation Stereo](https://nvlabs.github.io/FoundationStereo/)
- [Depth anything](https://github.com/DepthAnything/Depth-Anything-V2)
- [Basic stereo vision](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)

The foundation stereo model seems too big for what I want to do -> so not fast enough
I have tried setting up a stereo vision, with no success for now, the code seems to work. But either the physical setup or the calibrations picture are are not good enough. Because my results are way off

I have just started to look into epipolar geometry. For now it seems like a solution that would work, but if I can avoid going full in on this without any insurance, that would be good, as it seems pretty complicated.

I will try investing more time into basic stereo vision, and a little more on some lighter monocular depth estimation

It seems like fast depth is the way to go. But the website to download the weights is down right now. Don't know if or when it's gonna go up again
I found a spinoff, don't know if it's any good but it got the weights

# 05.06.25

So it seems that a monocular depth estimation model won't be enough. Either too unprecise, or too big to fit in real time on the jetson. So stereo vision it is

Stereo calibration is finally working, now I need to do a serious stand for the cameras in order to have a more volatile setup

The url for the jetson python library is [this](https://developer.download.nvidia.com/compute/redist/jp/v61/)
