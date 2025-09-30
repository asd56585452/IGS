#!/bin/bash

# Default values
startframe=0
endframe=60
videopath=""
skip_extraction_arg=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --videopath) videopath="$2"; shift ;;
        --startframe) startframe="$2"; shift ;;
        --endframe) endframe="$2"; shift ;;
        --skip_extraction) skip_extraction_arg="--skip_extraction" ;; # 新增這行：如果看到 --skip_extraction，就準備好對應的參數
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if videopath is provided
if [ -z "$videopath" ]; then
  echo "Usage: $0 --videopath <path> [--startframe <start>] [--endframe <end>] [--skip_extraction]"
  exit 1
fi

# Calculate the length of the frame range
framelength=$((endframe - startframe))

# Run the Python scripts with the provided path and frame range
echo "pre_input.py"
# 將 $skip_extraction_arg 傳遞給 python 腳本
python pre_input.py --videopath "$videopath" --startframe "$startframe" --endframe "$endframe" $skip_extraction_arg

echo "my_convert.py"
python my_convert.py -s "$videopath/colmap_0"
echo "my_copy_cams.py"
python my_copy_cams.py --source "$videopath/colmap_0" --scene "$videopath"
echo "my_convert_frames.py"
python my_convert_frames.py -s "$videopath"  --endframe "$framelength"