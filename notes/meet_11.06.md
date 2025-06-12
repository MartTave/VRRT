# Meeting 11.06

## Current situation

Stereo calibration working ! 

Was blocked because hardware manager was not there !

Explored many algorithms for the stereo matching -> I should be able to find the one

Explored Mono depth algorithms -> not good enough for now, so out of the picture. Tested many things

Stereo algorithms. Wanna tests before exploring too much, because it seems like a deep rabbithole...

Asked people for help -> Jerome Darbellay -> Can I ask him for help ?? 

The stereo setup is being done now



## What's the near future plan

Prepare the recording for saturday -> **When are we making a reunion with the client ?** I want to show him where I'm at as the solution is becoming more clear

I want to take good data saturday, to have good way to test my solutions

This means trying to sycronize at best usb cameras. Will not be perfect, but I think it will be good enough

I'm in parrelel trying to train a little better a model to recognize bib, found a better dataset (smaller but more quality)

I think i'm gonna need to label some data. To have person and bib number class in the same dataset, to not forget about person when training. This can be either an already label for bib number dataset on which I add the persons -> this could be quite fast as I can use a model to annotate, and then just check the results.
Or label one from scratch...

I am gonna need to label the data for the race this weekend in order to have a ground truth to compare the results -> I may be able to use the data from the race directly !!!


## My questions

Am I in good way ? Am I doing enough ? I'm thinking I do not have many results for the past week or so, it is alarming for me.

I'm hoping that this weekend will fix this by giving a clean way to test my entire pipeline, which will lead to meaningful statics.




