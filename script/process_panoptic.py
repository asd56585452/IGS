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
import cv2  # 引入 OpenCV

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

def undistort_and_crop_images(path, hd_cameras, offset, final_resolution=(1920, 1080)):
    """
    使用 OpenCV 對影像進行 Undistort，強制 cx, cy 置中，並裁切黑邊。
    採用 remap 方式確保影像和相機參數的幾何一致性。
    """
    project_folder = os.path.join(path, f"colmap_{offset}")
    original_input_folder = os.path.join(project_folder, "input_distorted")
    undistorted_input_folder = os.path.join(project_folder, "images")
    
    if os.path.exists(os.path.join(project_folder, "input")):
        os.rename(os.path.join(project_folder, "input"), original_input_folder)
    
    os.makedirs(undistorted_input_folder, exist_ok=True)

    hd_cameras_undistorted = []

    FINAL_W, FINAL_H = final_resolution

    print(f"Undistorting images for colmap_{offset}...")
    for cam_info in tqdm(hd_cameras):
        K = np.array(cam_info['K'])
        dist = np.array(cam_info['distCoef']).flatten()
        W, H = cam_info['resolution']
        
        img_name = f"hd_{cam_info['name']}.png"
        img_path = os.path.join(original_input_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue
            
        img = cv2.imread(img_path)

        # 步驟 1: 計算最佳裁切區域(roi)和一個暫時性的 new_K
        temp_new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), alpha=0)
        x, y, w, h = roi
        
        # 如果 roi 無效 (例如寬高為0)，則跳過此圖片
        if w <= 0 or h <= 0:
            print(f"Warning: Invalid ROI for {img_name}. Skipping.")
            continue

        # 步驟 2: 定義我們最終想要的、主點置中的目標相機矩陣 (target_K)
        # 我們使用 temp_new_K 的焦距，但強制 cx, cy 置中於裁切後尺寸的中心
        w = int(min(temp_new_K[0, 2],w-temp_new_K[0, 2])*2)
        h = int(min(temp_new_K[1, 2],h-temp_new_K[1, 2])*2)
        ws = FINAL_W / w
        hs = FINAL_H / h
        target_K = np.array([
            [ws * temp_new_K[0, 0], 0, ws * w / 2.0],
            [0, hs * temp_new_K[1, 1], hs * h / 2.0],
            [0, 0, 1]
        ])
        w = FINAL_W
        h = FINAL_H

        # 步驟 3: 計算從 target_K 到原始影像的映射
        # 注意：這裡的尺寸 W, H 是原始影像尺寸
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, target_K, (w, h), cv2.CV_32FC1)

        # 步驟 4: 應用映射
        # 這裡直接生成尺寸為 (w, h) 的影像，其幾何屬性符合 target_K
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        
        # 由於 remap 已經產生了正確尺寸的影像，理論上不再需要裁切
        # 但為了確保萬無一失，可以保留裁切步驟，此時 x,y 應為0,0
        # cropped_img = undistorted_img[0:h, 0:w]
        
        # 儲存處理後的影像
        cv2.imwrite(os.path.join(undistorted_input_folder, img_name), undistorted_img)
        
        # --- 更新相機參數 ---
        new_W, new_H = w, h
        
        # 建立新的 cam_info
        cam_info_undistorted = cam_info.copy()
        cam_info_undistorted['K'] = target_K.tolist()
        cam_info_undistorted['distCoef'] = [0.0, 0.0, 0.0, 0.0, 0.0]
        cam_info_undistorted['resolution'] = [new_W, new_H]
        hd_cameras_undistorted.append(cam_info_undistorted)

    return hd_cameras_undistorted


def convert_panoptic_to_colmap_db(path, hd_cameras_undistorted, offset=0):
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

    for i, cam_info in enumerate(hd_cameras_undistorted):
        R = np.array(cam_info['R'])
        t = np.array(cam_info['t']).flatten()
        K = np.array(cam_info['K'])
        
        T = t
        W, H = cam_info['resolution']
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        # --- 修正 ---
        # 使用 PINHOLE 模型，它需要 4 個參數: fx, fy, cx, cy
        params = np.array([fx, fy, cx, cy])
        
        qvec = rotmat2qvec(R)
        image_id = i + 1
        camera_id = i + 1
        
        pngname = f"hd_{cam_info['name']}.png"

        line = f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} {camera_id} {pngname}\n\n"
        imagetxtlist.append(line)
        
        # --- 修正 ---
        camera_model_id = 1 # PINHOLE
        db.add_camera(model=camera_model_id, width=W, height=H, params=params, camera_id=camera_id)
        param_str = " ".join(map(str, params))
        # --- 修正 ---
        cameraline = f"{camera_id} PINHOLE {W} {H} {param_str}\n"
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

def run_colmap_for_frame(path, offset, hd_cameras_undistorted):
    folder = os.path.join(path, f"colmap_{offset}")
    print(f"Processing COLMAP for {folder}...")
    
    convert_panoptic_to_colmap_db(path, hd_cameras_undistorted, offset)
    
    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "images")
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
    
    final_sparse_folder = os.path.join(folder, "sparse/0")
    os.makedirs(final_sparse_folder, exist_ok=True)
    files_to_move = ['cameras.bin', 'images.bin', 'points3D.bin']
    for file_name in files_to_move:
        src = os.path.join(distortedmodel_base, file_name)
        dst = os.path.join(final_sparse_folder, file_name)
        if os.path.exists(src):
            shutil.move(src, dst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Panoptic data for all frames using OpenCV for undistortion.")
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
    
    hd_cameras_original = sorted([cam for cam in calibration_data['cameras'] if cam.get('type') == 'hd'], key=lambda x: x['name'])
    
    for frame_offset in range(0,args.endframe - args.startframe):
        print(f"\n--- Processing frame offset: {frame_offset} ---")
        
        hd_cameras_undistorted = undistort_and_crop_images(args.videopath, hd_cameras_original, frame_offset)
        
        if not hd_cameras_undistorted:
            print(f"Warning: No valid cameras after undistortion for frame {frame_offset}. Skipping COLMAP.")
            continue
            
        run_colmap_for_frame(args.videopath, frame_offset, hd_cameras_undistorted)
    
    print("\n--- All panoptic data processing is complete. ---")