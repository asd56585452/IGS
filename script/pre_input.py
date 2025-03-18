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
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
from thirdparty.colmap.pre_colmap import *
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
import multiprocessing as mp

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def extractframes(videopath):
    ctr=0
    for i in range(300):
        if os.path.exists(os.path.join(videopath.replace(".mp4", ""), str(i) + ".png")):
            ctr += 1

    if ctr == 300 or ctr == 150: # 150 for 04_truck 
        print("already extracted all the frames, skip extracting")
        return
    savepath = videopath.replace(".mp4", "")
    if not os.path.exists(savepath) :
        os.makedirs(savepath)
    do_system(f"ffmpeg -i {videopath} -start_number 0 {savepath}/%d.png")
    return




def preparecolmapdynerf(folder, offset=0):
    print(offset)
    folderlist = glob.glob(folder + "cam**/")
    imagelist = []
    savedir = os.path.join(folder, "colmap_" + str(offset))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savedir = os.path.join(savedir, "input")
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for folder in folderlist :
        imagepath = os.path.join(folder, str(offset) + ".png")
        imagesavepath = os.path.join(savedir, folder.split("/")[-2] + ".png")

        shutil.copy(imagepath, imagesavepath)


    



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=300, type=int)

    args = parser.parse_args()
    videopath = args.videopath

    startframe = args.startframe
    endframe = args.endframe


    if startframe >= endframe:
        print("start frame must smaller than end frame")
        quit()
    if startframe < 0 or endframe > 300:
        print("frame must in range 0-300")
        quit()
    if not os.path.exists(videopath):
        print("path not exist")
        quit()
    
    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    
    
    #### step1
    print("start extracting 300 frames from videos")
    videoslist = glob.glob(videopath + "*.mp4")
    for v in tqdm.tqdm(videoslist):
        extractframes(v)
    print("extract frames down")

    

    # # ## step2 prepare colmap input 
    res = []
    p = mp.Pool(100)
    for offset in range(startframe, endframe):
        res.append(p.apply_async(preparecolmapdynerf, args=(videopath,offset)))
    p.close()
    p.join()
    print("prepare input down")




