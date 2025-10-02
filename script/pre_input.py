# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os 
import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle
import sys 
import argparse
sys.path.append(".")
# from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
# from thirdparty.colmap.pre_colmap import *
# from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
import multiprocessing as mp

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def extractframes(videopath, start_frame, end_frame):
    savepath = videopath.replace(".mp4", "")
    
    # Check if frames are already extracted in the 0-indexed format
    all_frames_exist = True
    for i in range(end_frame - start_frame):
        if not os.path.exists(os.path.join(savepath, str(i) + ".png")):
            all_frames_exist = False
            break
    if all_frames_exist:
        print(f"Frames 0 to {end_frame - start_frame - 1} already exist in {savepath}, skipping extraction.")
        return

    if not os.path.exists(savepath) :
        os.makedirs(savepath)

    # Use ffmpeg to extract the specified frame range and re-index them from 0
    # The select filter is 0-indexed and inclusive.
    # setpts=PTS-STARTPTS resets the timestamp to make the output start from frame 0.
    # -vsync vfr is important to handle variable frame rate videos correctly.
    print(f"Extracting frames {start_frame} to {end_frame - 1} from {videopath} and re-indexing from 0.")
    do_system(f"ffmpeg -i {videopath} -vf \"select='between(n,{start_frame},{end_frame-1})',setpts=PTS-STARTPTS\" -vsync vfr -start_number 0 {savepath}/%d.png")
    return




def preparecolmapdynerf(folder, offset=0):
    print(offset)
    folderlist = glob.glob(folder + "hd_**/")+glob.glob(folder + "cam**/")
    imagelist = []
    savedir = os.path.join(folder, "colmap_" + str(offset))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savedir = os.path.join(savedir, "input")
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for folder_path in folderlist :
        # The source images are now 0.png, 1.png, etc., corresponding to the offset
        imagepath = os.path.join(folder_path, str(offset) + ".png")
        imagesavepath = os.path.join(savedir, folder_path.split("/")[-2] + ".png")

        shutil.copy(imagepath, imagesavepath)


    



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=300, type=int)
    parser.add_argument("--skip_extraction", action='store_true', help="If set, skip the frame extraction process.")

    args = parser.parse_args()
    videopath = args.videopath

    startframe = args.startframe
    endframe = args.endframe


    if startframe >= endframe:
        print("start frame must smaller than end frame")
        quit()
    if startframe < 0:
        print("start frame cannot be negative")
        quit()
    if not os.path.exists(videopath):
        print("path not exist")
        quit()
    
    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    
    
    #### step1
    if not args.skip_extraction:
        print(f"Start extracting frames from videos, range {startframe} to {endframe}")
        videoslist = glob.glob(videopath + "*.mp4")
        for v in tqdm.tqdm(videoslist):
            extractframes(v, startframe, endframe) # 之前的 extractframes 函式會自動檢查，這裡再加上手動跳過更保險
        print("Extract frames down")
    else:
        print("--- Skipped frame extraction as requested by the '--skip_extraction' flag. ---")



    

    # # ## step2 prepare colmap input 
    res = []
    p = mp.Pool(100)
    # Loop over the new, 0-indexed frame numbers
    for offset in range(endframe - startframe):
        res.append(p.apply_async(preparecolmapdynerf, args=(videopath,offset)))
    p.close()
    p.join()
    print("prepare input down")