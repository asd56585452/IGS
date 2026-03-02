import os
import shutil
import glob
from pathlib import Path

def convert_hifi4g_to_igs(base_dir):
    """
    Converts HiFi4G dataset format to IGS format using symlinks.
    
    Source structure:
        base_dir/sparse/0/ (camera info)
        base_dir/images/{frame}/{cam_num}.png (images)
        
    Target structure:
        base_dir/colmap_{frame}/sparse/0 (symlink to base_dir/sparse/0)
        base_dir/colmap_{frame}/image/{cam_num}.png (symlink to images)
    """
    base_path = Path(base_dir)
    images_root = base_path / "image_white_undistortion"
    sparse_source = base_path / "image_white_undistortion" / "colmap" / "sparse" / "0"
    
    if not images_root.exists():
        print(f"Error: Images root not found at {images_root}")
        return
    
    if not sparse_source.exists():
        print(f"Error: Sparse source not found at {sparse_source}")
        return

    # Get all frame directories (e.g., '30', '45', etc.)
    # We look for directories in the images folder
    frame_dirs = [d for d in images_root.iterdir() if d.is_dir() and d.name.isdigit()]
    frame_dirs = sorted(frame_dirs, key=lambda x: int(x.name))
    
    print(f"Found {len(frame_dirs)} frames to process.")
    
    for frame_dir in frame_dirs:
        frame_name = frame_dir.name
        # Target directory: colmap_{frame}
        colmap_dir = base_path / f"colmap_{frame_name}"
        
        # Create colmap_{frame}/sparse/0
        target_sparse_dir = colmap_dir / "sparse"
        target_sparse_0_dir = target_sparse_dir / "0"
        
        # Create colmap_{frame}/image
        target_image_dir = colmap_dir / "images"
        
        # 1. Setup Directories
        # We need to make sure 'sparse' exists to link '0' inside it
        # Or we can link the 'sparse' folder itself if it contains '0' directly?
        # The user said: /home/.../colmap_{frame}/sparse/0放相機資訊
        # So we create colmap_{frame}/sparse and link '0' to the source '0'.
        
        try:
            target_sparse_dir.mkdir(parents=True, exist_ok=True)
            target_image_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Failed to create directories for frame {frame_name}: {e}")
            continue

        # 2. Symlink Sparse Data
        # We link target_sparse_0_dir -> sparse_source
        if not target_sparse_0_dir.exists():
            try:
                # symlink(source, link_name) -> link_name points to source
                # Use absolute paths to be safe
                os.symlink(sparse_source.absolute(), target_sparse_0_dir.absolute())
                # print(f"Linked sparse: {target_sparse_0_dir} -> {sparse_source}")
            except OSError as e:
                print(f"Failed to symlink sparse for frame {frame_name}: {e}")
        
        # 3. Symlink Images
        # Source images: images/{frame}/*.png
        # Target images: colmap_{frame}/image/*.png
        source_images = list(frame_dir.glob("*.png"))
        if not source_images:
             # Try other extensions just in case
             source_images = list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.jpeg"))
        
        count = 0
        for src_img in source_images:
            dst_img = target_image_dir / src_img.name
            if not dst_img.exists():
                try:
                    os.symlink(src_img.absolute(), dst_img.absolute())
                    count += 1
                except OSError as e:
                    print(f"Failed to link image {src_img.name}: {e}")
        
        print(f"Frame {frame_name}: Linked sparse info and {count} images to {colmap_dir}")

    print("Conversion complete.")

if __name__ == "__main__":
    dataset_path = "/home/tsaichenghan/IGS/dataset/4K_Actor1_Greeting"
    convert_hifi4g_to_igs(dataset_path)
