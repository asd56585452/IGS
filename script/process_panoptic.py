import os
import subprocess
import argparse
import shutil
import glob
import numpy as np
import json
import sys
import multiprocessing as mp
from tqdm import tqdm

# 假設 pre_colmap.py 與此腳本在同一目錄
try:
    from pre_colmap import COLMAPDatabase
except ImportError:
    print("Error: pre_colmap.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def convert_panoptic_to_colmap_db(path, hd_cameras, offset=0):
    projectfolder = os.path.join(path, "colmap_" + str(offset))
    manualfolder = os.path.join(projectfolder, "manual")
    os.makedirs(manualfolder, exist_ok=True)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    db_path = os.path.join(projectfolder, "input.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    for i, cam_info in enumerate(hd_cameras):
        R = np.array(cam_info['R'])
        t = np.array(cam_info['t']).flatten()
        K = np.array(cam_info['K'])
        dist = np.array(cam_info['distCoef']).flatten()
        T = t
        W, H = cam_info['resolution']
        focal_x, focal_y, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        params = np.array([focal_x, focal_y, cx, cy, dist[0], dist[1], dist[2], dist[3]])
        qvec = rotmat2qvec(R)
        image_id = i + 1
        camera_id = i + 1
        
        pngname = f"hd_{cam_info['name']}.png"

        line = f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} {camera_id} {pngname}\n\n"
        imagetxtlist.append(line)
        
        camera_model_id = 4  # OPENCV
        db.add_camera(model=camera_model_id, width=W, height=H, params=params, camera_id=camera_id)
        param_str = " ".join(map(str, params))
        cameraline = f"{camera_id} OPENCV {W} {H} {param_str}\n"
        cameratxtlist.append(cameraline)
        db.add_image(name=pngname, camera_id=camera_id, prior_q=qvec, prior_t=T, image_id=image_id)

    db.commit()
    db.close()

    with open(savetxt, "w") as f:
        f.writelines(imagetxtlist)
    with open(savecamera, "w") as f:
        f.writelines(cameratxtlist)
    with open(savepoints, "w") as f:
        pass

def run_colmap_for_frame(path, offset, hd_cameras):
    folder = os.path.join(path, f"colmap_{offset}")
    print(f"Processing {folder}...")
    
    convert_panoptic_to_colmap_db(path, hd_cameras, offset)
    
    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel_base = os.path.join(folder, "distorted/sparse")
    manualinputfolder = os.path.join(folder, "manual")
    os.makedirs(distortedmodel_base, exist_ok=True)

    print("Running feature extractor...")
    feature_extractor_cmd = f"colmap feature_extractor --database_path {dbfile} --image_path {inputimagefolder}"
    subprocess.run(feature_extractor_cmd, shell=True, check=True)

    print("Running exhaustive matcher...")
    feature_matcher_cmd = f"colmap exhaustive_matcher --database_path {dbfile}"
    subprocess.run(feature_matcher_cmd, shell=True, check=True)

    print("Running point triangulator...")
    point_triangulator_cmd = f"colmap point_triangulator --database_path {dbfile} --image_path {inputimagefolder} --output_path {distortedmodel_base} --input_path {manualinputfolder}"
    subprocess.run(point_triangulator_cmd, shell=True, check=True)
    
    # --- 新增的修正 ---
    # 為 distorted/sparse 模型建立 '0' 子資料夾並移入檔案
    distortedmodel_final = os.path.join(distortedmodel_base, "0")
    os.makedirs(distortedmodel_final, exist_ok=True)
    files_to_move_distorted = ['cameras.bin', 'images.bin', 'points3D.bin']
    for file_name in files_to_move_distorted:
        src = os.path.join(distortedmodel_base, file_name)
        dst = os.path.join(distortedmodel_final, file_name)
        if os.path.exists(src):
            shutil.move(src, dst)
    # --- 修正結束 ---

    print("Running image undistorter...")
    # 注意: image_undistorter 的 input_path 應該是 distorted/sparse/0
    img_undistorter_cmd = f"colmap image_undistorter --image_path {inputimagefolder} --input_path {distortedmodel_final} --output_path {folder} --output_type COLMAP"
    subprocess.run(img_undistorter_cmd, shell=True, check=True)
    
    # 整理最終的 sparse 資料夾
    sparse_folder = os.path.join(folder, "sparse")
    final_sparse_folder = os.path.join(sparse_folder, "0")
    os.makedirs(final_sparse_folder, exist_ok=True)
    
    files_to_move = ['cameras.bin', 'images.bin', 'points3D.bin']
    for file_name in files_to_move:
        src = os.path.join(sparse_folder, file_name)
        dst = os.path.join(final_sparse_folder, file_name)
        if os.path.exists(src):
            shutil.move(src, dst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Panoptic data using known camera intrinsics/extrinsics.")
    parser.add_argument("--videopath", type=str, required=True, help="Path to the scene directory (e.g., ./data/panoptic/kitchen).")
    parser.add_argument("--startframe", type=int, default=0)
    parser.add_argument("--endframe", type=int, default=60)
    args = parser.parse_args()

    calibration_file = glob.glob(os.path.join(args.videopath, "calibration*.json"))
    if not calibration_file:
        print(f"Error: Calibration file not found in {args.videopath}")
        sys.exit(1)
        
    with open(calibration_file[0]) as f:
        calibration_data = json.load(f)
    
    hd_cameras = sorted([cam for cam in calibration_data['cameras'] if cam.get('type') == 'hd'], key=lambda x: x['name'])
    
    print("--- Running COLMAP for the first frame (colmap_0) ---")
    run_colmap_for_frame(args.videopath, 0, hd_cameras)
    
    print("--- Panoptic data processing for the first frame is complete. ---")