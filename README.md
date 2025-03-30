# Instant Gaussian Stream: Fast and Generalizable Streaming of Dynamic Scene Reconstruction via Gaussian Splatting [CVPR25] 

>Jinbo Yan, Rui Peng, Zhiyan Wang, Luyang Tang, Jiayu Yang, Jie Liang, Jiahao Wu, Ronggang Wang<br>
>[Arxiv](https://arxiv.org/pdf/2503.16979)|[Datasets](https://huggingface.co/datasets/yjb6/IGS_data)|[Weights](https://drive.google.com/file/d/1xh1DJ6oKUvNu-N2tWIkwdfOJv4LPAyMa/view?usp=drive_link)<br>
> *CVPR 25* 

This repository contains the official authors implementation associated with the paper: __Instant Gaussian Stream: Fast and Generalizable Streaming of Dynamic Scene Reconstruction via Gaussian Splatting__

## Bibtex
```
@misc{yan2025instantgaussianstreamfast,
      title={Instant Gaussian Stream: Fast and Generalizable Streaming of Dynamic Scene Reconstruction via Gaussian Splatting}, 
      author={Jinbo Yan and Rui Peng and Zhiyan Wang and Luyang Tang and Jiayu Yang and Jie Liang and Jiahao Wu and Ronggang Wang},
      year={2025},
      eprint={2503.16979},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.16979}, 
}
```

## Installation
- clone
    ```
    git clone https://github.com/yjb6/IGS.git --revursive
    ```
- Python >= 3.9
- Install `PyTorch >= 2.0.0`. We have tested on `torch2.0.0+cu118`, but other versions should also work fine.
    ```sh
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

    ```
- Install **torch_cluster** by following the instructions provided in the official repository: [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster).
- Install gaussian_rasterization. We use a variant of the Rade-GS renderer.
    ```sh
    pip install submodules/RaDe-GS/submodules/diff-gaussian-rasterization-clamp/

    pip install submodules/RaDe-GS/submodules/diff-gaussian-rasterization/
    ```
- Follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install simple-knn

- Install other pkgs:
    ```
    pip install -r requirements.txt
    ```





## Demo
#### Step 1: Donwnload the prepared [Data](https://drive.google.com/file/d/1uPhqkUE4vJhialpG1DC8uK-CkmiEjQe8/view?usp=drive_link) and [CKPT](https://drive.google.com/file/d/1xh1DJ6oKUvNu-N2tWIkwdfOJv4LPAyMa/view?usp=drive_link)
Extract the Prepared Data
```
.
├── bbox.json
├── sear_steak
│   ├── colmap_0
│   │   ├── images_512
│   │   ├── images_r2
│   │   └── start_gs
│   ├── colmap_...
│   │   ├── images_512
│   │   └── images_r2
└── sear_steak_total_50_interval_5.json
```
Extract the ckpt under the code directory

```
.
├── infer_batch.py
├── ckpt
│   ├── gmflow
│   └── igs
├── ...
```

#### Step 2: Run  
```bash
python infer_batch.py --config configs/demo.yaml
```  
• Remember to replace `data.data.root_dir` in [demo.yaml](configs/demo.yaml) with your own directory path.  
• Feel free to adjust the hyperparameters, such as `refine_iterations` and `max_num`.
#### Step 3: Generate Videos  
Use this [script](script/video.ipynb) to convert the rendered images into a video.

## Streaming Reconstruction

### Data Preparation
#### Step 1: Prepare Inputs
<a name="step1-prepare-inputs"></a>
We can prepare our streaming dataset following 3DGStream and SpacetimeGaussian, and here we provide a simple script.
```sh
cd script
./pre_test_data.sh /path/to/your/scene
cd ..
```
Then you will get:
```
<root>
│   |---<your_scene>
│   |   |--- colmap_0
│   |   |   |--- image
│   |   |--- colmap_...
│   |   |--- colmap_299
```
#### Step 2: Train the Gaussian of Frame 0
In this step, we recommend first optimizing the Gaussian model for several thousand iterations using ​RaDe-GS, followed by compressing the Gaussian points with ​LightGaussian. This process ensures an efficient and high-quality reconstruction of the 3D scene.

To facilitate this, we provide a script that includes the ​RaDe-GS training process, ​LightGaussian compression, and ​rendering. Before running the script, make sure to adjust the `YOUR_PATH` parameter to match your specific directory structure.

Install the submodules:
```
pip install submodules/RaDe-GS/submodules/diff-gaussian-rasterization-clamp/
pip install submodules/RaDe-GS/submodules/compress-diff-gaussian-rasterization/
```
Train:
```sh
cd submodules/RaDe-GS
./train.sh
cd -
```
#### Step 3: Downsize the Image  
Resize the images to 512x512 for processing by AGM-Net.  
```sh
cd script
python subsample.py -r /path/to/your/scene
cd ..
```
*(Optional)*  
If you chose to downsample the image during the training of the first frame of Gaussians, you need to resize the images to the corresponding dimensions here.

```sh
cd script
python subsample_pil.py -r /path/to/your/scene --target_size corrsponding_width corrsponding_height
#example: python subsample_pil.py -r /path/to/your/scene --target_size 1024 1024
cd ..
```

#### Step 4: Generate Key-Candidate Pair and Bounding Box Configuration  
Use the provided [script](script/generate_test_pair.ipynb) to control the interval between key frames and generate the corresponding key frame sequence.  

Additionally, you need to specify the bounding box (bbox) for the dynamic area of the scene. We have provided the BBOX configurations for the **N3DV** and **MeetingRoom** scenes. Make sure to copy them to your dataset root directory.

#### The final testing data structure:
```
<root>
│   |---bbox.json
│   |---<scene_name>_total_<>_interval<w>.json
│   |---<scene_name>
│   |   |--- colmap_0
│   |   |   |--- start_gs
│   |   |   |--- image
│   |   |   |--- image_512
│   |   |   |--- image_r2
│   |   |--- colmap_...
│   |   |   |--- image
│   |   |   |--- image_512
│   |   |   |--- image_r2
│   |   |--- colmap_299
│   |   |   |--- image
│   |   |   |--- image_512
│   |   |   |--- image_r2
```


### Run
To test the model, use the following command:
```
python infer_batch.py  --config <path to config> 
```
- We have provided the relevant configurations for **N3DV** and **Meeting Room**, and the details can be found in [configs](configs).


## Train AGM-Net
### Datasets Preparation
#### Our Training Dataset
Download our [processed data](https://huggingface.co/datasets/yjb6/IGS_data) from 4 sequences of N3DV, which can be directly used for training. It contains 1,200 optimized Gaussian points and requires 150GB of storage space.

After extraction, the directory structure is as follows:
```
.
└── IGS_data
    ├── bbox.json
    ├── coffee_martini_colmap
    ├── cook_spinach_colmap
    ├── flame_salmon_1
    ├── flame_steak_colmap
    └── N3D_train_gap_10.json
```
#### Prepare More Dataset
You can prepare additional data following:
##### Step 1: 
Follow the [tutorial](step1-prepare-inputs) to prepare the extracted images from multi-view video.
##### Step 2: 
For each frame, reconstruct the corresponding Gaussians and perform rendering. We provide a script that supports multi-GPU parallel processing for Gaussian point reconstruction, compression, and rendering. 
**Remember to change the parameters in the code.**
```
python build_3dgs_dataset.py
```
##### Step 3:  
Generate training pairs, camera groups, and bounding boxes by executing the provided [script](script/generate_train_pair.ipynb).

### Run
```
accelerate launch --config_file acc_cfg/default_config.yaml main.py --config configs/train.yaml 
```


## Acknowledgments  
Our work builds upon the following open-source projects and their contributions:  
• [3DGStream](https://github.com/SJoJoK/3DGStream)  
• [LGM](https://github.com/3DTopia/LGM?tab=readme-ov-file)  
• [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)  
• [Rade-GS](https://github.com/BaowenZ/RaDe-GS)  
• [LightGaussian](https://github.com/VITA-Group/LightGaussian)  

We are deeply grateful to the authors and communities behind these projects for their valuable work and inspiration.
