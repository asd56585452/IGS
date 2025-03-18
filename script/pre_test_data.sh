#!/bin/bash

# Check if the user provided a video path as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <videopath>"
  exit 1
fi

# Assign the first argument to the videopath variable
videopath="$1"

# Run the Python scripts with the provided path
python pre_input.py --videopath "$videopath"
python my_convert.py -s "$videopath/colmap_0"
python my_copy_cams.py --source "$videopath/colmap_0" --scene "$videopath"
python my_convert_frames.py -s "$videopath"