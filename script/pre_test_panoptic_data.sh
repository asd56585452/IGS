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
        --skip_extraction) skip_extraction_arg="--skip_extraction" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if videopath is provided
if [ -z "$videopath" ]; then
  echo "Usage: $0 --videopath <path> [--startframe <start>] [--endframe <end>] [--skip_extraction]"
  exit 1
fi

# 確保路徑結尾有斜線
[[ "${videopath}" != */ ]] && videopath="${videopath}/"

# --- 修改後的流程 ---

# 步驟 1: 抽取所有影格並準備 COLMAP 輸入資料夾
echo "Step 1: Running pre_input.py to extract frames and prepare directories..."
python pre_input.py --videopath "$videopath" --startframe "$startframe" --endframe "$endframe" $skip_extraction_arg

# 步驟 2: 處理所有 Panoptic 資料，包含 OpenCV undistortion 和 COLMAP
echo "Step 2: Running process_panoptic.py for all frames..."
python process_panoptic.py --videopath "$videopath" --startframe "$startframe" --endframe "$endframe"

echo "--- All processing steps completed. ---"