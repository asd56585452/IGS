import multiprocessing as mp
import os
from tqdm import tqdm
import logging
import json

GPU_NUM = 4
ROOT_DIR ="YOUR ROOT DIR"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def build_3dgs(queue, task):
    #remeber to change corresponding iterations!!!!!

    gpu_id = queue.get()
    folder_path = os.path.join(ROOT_DIR, task)
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {folder_path} -l N3D --iterations 7000 -m {folder_path}/test_4kiter --quiet --sh_degree 1 -r 2 --eval" # 
    logging.info(f"Running command on GPU {gpu_id}: {command}")
    folder_path = os.path.join(ROOT_DIR, task,"test_4kiter")

    compress_command = f"CUDA_VISIBLE_DEVICES={gpu_id} python compress.py -m {folder_path} --iterations 15000 --iteration 4000"
    render_command = f"CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {folder_path} --iteration 14000_compress"
    logging.info(f"Running command on GPU {gpu_id}: {compress_command},{render_command}")

    try:
        os.system(command)
        os.system(compress_command)
        os.system(render_command)
    except Exception as e:
        logging.error(f"Error executing command: {e}")
    finally:
        queue.put(gpu_id)


if __name__ == "__main__":
    name_list= ["scene_1", "scene_2", "scene_3"]

    task_list = []
    for name in name_list:
        for i in range(300):
            task_list.append(os.path.join(name,"colmap_"+str(i)))

    p = mp.Pool(GPU_NUM)

    manager = mp.Manager()
    queue = manager.Queue()
    for i in range(GPU_NUM):
        queue.put(i)

    for i in tqdm(range(len(task_list))):
        p.apply_async(build_3dgs, args=(queue, task_list[i]))

    p.close()
    p.join()