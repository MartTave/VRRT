#!/bin/bash

ffmpeg -i $1 -vf "fps=10,scale=1280:720:flags=lanczos,palettegen" temp_palette.png
ffmpeg -i $1 -i temp_palette.png -filter_complex "fps=$3,scale=1280:720:flags=lanczos[x];[x][1:v]paletteuse" $2
