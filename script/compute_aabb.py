import numpy as np
import struct
import argparse
import json
import os

# COLMAP point3D.bin 檔案的讀取函式
# 參考自: https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
def read_points3d_binary(path_to_model_file):
    """
    讀取 points3D.bin 並回傳一個 (N, 3) 的 NumPy 陣列，包含所有點的 XYZ 座標。
    """
    points = []
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_points):
            # 讀取每筆點資料的固定長度部分
            # point3D_id(Q), XYZ(3d), RGB(3B), error(d)
            point_data = struct.unpack('<Qddd3Bd', fid.read(8 + 3*8 + 3*1 + 8))
            
            # 我們只需要 XYZ 座標
            xyz = point_data[1:4]
            points.append(xyz)
            
            # 讀取並跳過 track data (image_id, point2D_idx)
            track_len = struct.unpack('<Q', fid.read(8))[0]
            fid.read(8 * track_len) # 每個 track 元素是 2 * uint32_t = 8 bytes

    print(f"成功讀取 {len(points)} 個 3D 點。")
    return np.array(points)

def compute_aabb(points, percentile=100, padding_factor=0.05):
    """
    從點雲計算 AABB。
    :param points: (N, 3) 的 NumPy 陣列
    :param percentile: 用於過濾離群點的百分位數 (100 表示不過濾)
    :param padding_factor: 在計算出的邊界上增加的邊距比例
    :return: aabb 列表 [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    """
    if points.shape[0] == 0:
        return [[-1, -1, -1], [1, 1, 1]]

    if percentile < 100:
        print(f"使用 {percentile} 和 {100-percentile} 百分位數過濾離群點...")
        min_bound = np.percentile(points, (100 - percentile) / 2, axis=0)
        max_bound = np.percentile(points, 100 - (100 - percentile) / 2, axis=0)
    else:
        print("使用絕對最小/最大值計算邊界...")
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)

    # 計算邊距
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    padded_size = size * (1 + padding_factor)
    
    # 計算帶邊距的 aabb
    final_min_bound = center - padded_size / 2
    final_max_bound = center + padded_size / 2
    
    aabb = [final_min_bound.tolist(), final_max_bound.tolist()]
    
    print("\n計算完成:")
    print(f"  - 最小邊界 (min_bound): {final_min_bound}")
    print(f"  - 最大邊界 (max_bound): {final_max_bound}")
    
    return aabb


def main():
    parser = argparse.ArgumentParser(description="從 COLMAP 的 points3D.bin 檔案計算 AABB。")
    parser.add_argument("--input", type=str, required=True,
                        help="輸入的 points3D.bin 檔案路徑。")
    parser.add_argument("--output", type=str, default="bbox.json",
                        help="輸出的 bbox.json 檔案路徑。")
    parser.add_argument("--percentile", type=float, default=99.9,
                        help="用於過濾離群點的百分位數。例如 99.9 表示移除最極端的 0.1%% 的點。設為 100 則不進行過濾。")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="在計算出的 AABB 基礎上增加的邊距比例 (例如 0.05 表示 5%%)。")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"錯誤: 輸入檔案不存在! {args.input}")
        return

    # 1. 讀取點雲
    points_xyz = read_points3d_binary(args.input)
    
    # 2. 計算 AABB
    aabb = compute_aabb(points_xyz, args.percentile, args.padding)
    
    # 3. 準備輸出
    output_data = {
        "aabb": aabb
        # 你可以根據需要加入 scale 等其他參數
        # "scale": 1.0 
    }
    
    # 4. 儲存到 JSON 檔案
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nAABB 已成功儲存至: {args.output}")


if __name__ == "__main__":
    main()