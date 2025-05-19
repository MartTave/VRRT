# Unfiltered research

### Maybe contain what I want ?

[secondscount](https://secondscount.com/)

### This seems easier, but may be inspiring...

[photofinish-app](https://photofinish-app.com/en/)

### Maybe ? On ice, and seems to be for on person at a time

[paper](https://pubmed.ncbi.nlm.nih.gov/24936905/)

[detailed paper](https://journals.lww.com/nsca-jscr/fulltext/2014/09000/a_simple_video_based_timing_system_for_on_ice_team.37.aspx)

### Ideas from chatGPT and me

- Time syncing if using multiple devices - NTP or GPS time

- [YOLOv4 + CUDA bib detection](https://github.com/Lwhieldon/BibObjectDetection)

- [mit paper on bib detection and recognition](https://people.csail.mit.edu/talidekel/RBNR.html#:~:text=amateur%C2%A0photographers,on%20three%20newly%20collected%20datasets)

- [Bib number detection](https://github.com/gheinrich/bibnumber) - **Seems to ignore single digit bib number**

- [Bib detector - YOLOv4-tiny](https://github.com/ericBayless/bib-detector) - Seems to use a two step approach : detect bib number first, and then try to read it

- [Nvidia stereo vision](https://nvlabs.github.io/FoundationStereo/), [Github](https://github.com/NVlabs/FoundationStereo/)

- [May be good processing ?](https://github.com/ConsistentlyInconsistentYT/Pixeltovoxelprojector) [related youtube](https://youtu.be/m-b51C82-UE?si=Kx2jYuIh7PbDCHFp).

- They say that it's impossible, I don't believe them.... [here](https://behindthefinishline.com/blog/what-timing-technology-is-right-for-your-race#camera-vision). But they have a point, if used as a standalone system, I need a way to allow for human intervention the system is not confident/bib number can't be recognized

### Ideas

Two cameras will be necessary, but maybe a third to detect the line itself ? Or two, one to detect bib, and the other one to detect the timing precisly ? I think it can be flexible on that

Global shutter might be a good idea, might help to avoid motion blur for fast moving object.

We may need to have a big baseline (distance between the two cameras) in order to have a good precision at range of 5-20 meters. Apparently, focal length is a criteria too for precision for stereo setup

We may have performances problem if running some heavy models, like for example the foundations stereo (from nvidia), apparently we nee dto expect between 10-25 FPS for processing. We might need to find a solution to ease this a little

To sync the start, it may be hard. I need to look into how they do it for now ?

### Solutions already existing

- Seems to be like what I need to do [here](https://www.innovativetimingsystems.com/trackandfield)

- The famous [FinishLynx](https://finishlynx.com/), very expensive, and pretty old techno (~2013)

- Seems pretty good as they do ID and timing with a camera [eagleeye](https://www.eagleeyetrack.com/collections/videos-1)

- Seems like a solution like I need to do : [FlashTiming](https://flashtiming.com/)
