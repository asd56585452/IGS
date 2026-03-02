
import struct
import numpy as np
import sys

def read_ply_bbox(ply_path):
    with open(ply_path, 'rb') as f:
        # Read header
        header_end = False
        properties = []
        num_vertices = 0
        while not header_end:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property float'):
                properties.append(line.split()[-1]) # e.g. x, y, z, nx, ny, nz
            elif line == 'end_header':
                header_end = True
        
        print(f"Num vertices: {num_vertices}")
        print(f"Properties: {properties}")
        
        # We only care about x, y, z which remain typically at the beginning
        # Standard PLY binary little endian
        # We'll read all data and extract x, y, z
        
        # Determine property size (assuming all float32 for simplicity which is standard for point_cloud.ply in 3DGS)
        # Actually 3DGS/colmap ply often has x,y,z,nx,ny,nz,f_dc_0, f_dc_1, ...
        # But we only need the first 3 floats.
        
        # Let's count how many properties there are
        num_props = len(properties)
        # Each property is float32 (4 bytes)
        vertex_size = 4 * num_props
        
        # Read data
        data = f.read(num_vertices * vertex_size)
        if len(data) != num_vertices * vertex_size:
            print(f"Warning: Expected {num_vertices * vertex_size} bytes, got {len(data)}")
        
        # Parse using numpy
        # Create a dtype
        dtype_list = [(p, 'f4') for p in properties]
        arr = np.frombuffer(data, dtype=dtype_list)
        
        x = arr['x']
        y = arr['y']
        z = arr['z']
        
        min_bound = [float(np.min(x)), float(np.min(y)), float(np.min(z))]
        max_bound = [float(np.max(x)), float(np.max(y)), float(np.max(z))]
        
        print(f"Min: {min_bound}")
        print(f"Max: {max_bound}")
        
        return min_bound, max_bound

def update_bbox_json(json_path, scene_name, min_b, max_b):
    import json
    import os
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}
        
    data[scene_name] = [min_b, max_b]
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated {json_path} for scene {scene_name}")

if __name__ == "__main__":
    ply_file = "/home/tsaichenghan/IGS/dataset/4K_Actor1_Greeting/colmap_0/test/point_cloud/iteration_6000_compress/point_cloud.ply"
    json_file = "/home/tsaichenghan/IGS/dataset/bbox.json"
    
    min_b, max_b = read_ply_bbox(ply_file)
    center_b = [(min_b[i] + max_b[i]) / 2 for i in range(3)]
    scale_b = [(max_b[i] - min_b[i]) / 2 for i in range(3)]
    scale_b = [scale_b[i]*2 for i in range(3)]
    min_b = [center_b[i] - scale_b[i] for i in range(3)]
    max_b = [center_b[i] + scale_b[i] for i in range(3)]
    update_bbox_json(json_file, "4K_Actor1_Greeting", min_b, max_b)
