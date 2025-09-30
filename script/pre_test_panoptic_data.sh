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


# Calculate the length of the frame range
framelength=$((endframe - startframe))

# --- 修改後的流程 ---

# 步驟 1: 抽取影格並準備 COLMAP 輸入資料夾
echo "Step 1: Running pre_input.py to extract frames and prepare directories..."
python pre_input.py --videopath "$videopath" --startframe "$startframe" --endframe "$endframe" $skip_extraction_arg

# 步驟 2: 處理 Panoptic 資料 (僅針對 colmap_0)，使用已知的相機參數
echo "Step 2: Running process_panoptic.py for the first frame (colmap_0)..."
python process_panoptic.py --videopath "$videopath"

# 步驟 3: 複製相機參數到所有其他的 frame
echo "Step 3: Running my_copy_cams.py to distribute camera models..."
python my_copy_cams.py --source "${videopath}colmap_0" --scene "$videopath"

# 步驟 4: 為剩餘的 frames 產生 undistorted 影像
echo "Step 4: Running my_convert_frames.py to undistort images for all frames..."
python my_convert_frames.py -s "$videopath" --endframe "$framelength"

echo "--- All processing steps completed. ---"