#!/bin/bash

# Default values
START_FRAME=0
END_FRAME=60

# Check if the user provided a video path as an argument. The first argument must be the videopath.
if [ -z "$1" ]; then
  echo "Usage: $0 <videopath> [--start_frame <start>] [--end_frame <end>]"
  exit 1
fi
VIDEOPATH="$1"
shift # consume videopath argument

# Parse optional arguments for start and end frames
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start_frame) START_FRAME="$2"; shift ;;
        --end_frame) END_FRAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python scripts with the provided path and frames
echo "pre_input.py"
python pre_input.py --videopath "$VIDEOPATH" --startframe "$START_FRAME" --endframe "$END_FRAME"

echo "my_convert.py"
python my_convert.py -s "$VIDEOPATH/colmap_0"

echo "my_copy_cams.py"
python my_copy_cams.py --source "$VIDEOPATH/colmap_0" --scene "$VIDEOPATH"

echo "my_convert_frames.py"
# Calculate the number of frames to process based on the interval
NUM_FRAMES=$((END_FRAME - START_FRAME))
python my_convert_frames.py -s "$VIDEOPATH" --endframe "$NUM_FRAMES"