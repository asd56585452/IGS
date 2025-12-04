# Step 1: Train the Gaussian model using RaDe-GS
python train.py -s /home/tsaichenghan/IGS/dataset/sear_steak/colmap_0/ --loader N3D --eval -r 2 --iterations 6000 -m /home/tsaichenghan/IGS/dataset/sear_steak/colmap_0/test

# Step 2: Compress the Gaussian points using LightGaussian
python compress.py -m /home/tsaichenghan/IGS/dataset/sear_steak/colmap_0/test/ --iteration 5000 --prune_percent 0.45 --iterations 6000

# Step 3: Render the compressed Gaussian model
python render.py -m /home/tsaichenghan/IGS/dataset/sear_steak/colmap_0/test --iteration 6000_compress