# Step 1: Train the Gaussian model using RaDe-GS
python train.py -s /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/ --loader N3D --eval -r 2 --iterations 6000 -m /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test

# Step 2: Compress the Gaussian points using LightGaussian
python compress.py -m /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test/ --iteration 5000 --prune_percent 0.45 --iterations 6000

# Step 3: Render the compressed Gaussian model
python render.py -m /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test --iteration 6000_compress

# (IGS) cgvmis418@cgvmis418-System-Product-Name:/media/cgvmis418/2TBSSD/IGS/submodules/RaDe-GS$ bash monitor_vram_pro.sh ./train.sh
# VRAM 監控已啟動... (每 10 秒記錄一次)
# 正在執行您的指令: ./train.sh
# 程式輸出:
# ------------------------------------------
# Optimizing /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test
# this is rade-gs
# Output folder: /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test [27/12 15:24:15]
# N3D [27/12 15:24:15]
# Reading camera 21/21 [27/12 15:24:15]
# cameras extent: 6.467737865447998 [27/12 15:24:15]
# Converting point3d.bin to .ply, will happen only the first time you open the scene. [27/12 15:24:15]
# Loading Training Cameras: 20 . [27/12 15:24:18]
# Loading Test Cameras: 1 . [27/12 15:24:18]
# Number of points at initialisation :  7200 [27/12 15:24:18]
# Training progress:  67%|▋| 4000/6000 [01:17<00:34, 58.42it/s, Loss=0.0176, loss_pointnum torch.Size([194303, 3]) [27/12 15:25:35]

# [ITER 4000] Evaluating test: L1 0.01453300192952156 PSNR 34.35335922241211 [27/12 15:25:36]

# [ITER 4000] Evaluating train: L1 0.012471337243914605 PSNR 34.66603584289551 [27/12 15:25:38]

# [ITER 4000] Saving Gaussians [27/12 15:25:38]
# Training progress:  75%|▊| 4500/6000 [01:30<00:26, 57.20it/s, Loss=0.0173, loss_pointnum torch.Size([203066, 3]) [27/12 15:25:48]

# [ITER 4500] Evaluating test: L1 0.014511804096400738 PSNR 34.34843444824219 [27/12 15:25:48]

# [ITER 4500] Evaluating train: L1 0.011324077472090723 PSNR 35.325547027587895 [27/12 15:25:49]

# [ITER 4500] Saving Gaussians [27/12 15:25:49]
# Training progress:  77%|▊| 4600/6000 [01:33<00:27, 50.13it/s, Loss=0.0141, loss_pointnum torch.Size([204296, 3]) [27/12 15:25:52]

# [ITER 4600] Evaluating test: L1 0.015337353572249413 PSNR 33.990882873535156 [27/12 15:25:52]

# [ITER 4600] Evaluating train: L1 0.01139219030737877 PSNR 35.381491088867186 [27/12 15:25:53]

# [ITER 4600] Saving Gaussians [27/12 15:25:53]
# Training progress:  78%|▊| 4700/6000 [01:37<00:26, 49.56it/s, Loss=0.0184, loss_pointnum torch.Size([205527, 3]) [27/12 15:25:56]

# [ITER 4700] Evaluating test: L1 0.014346057549118996 PSNR 34.45615768432617 [27/12 15:25:56]

# [ITER 4700] Evaluating train: L1 0.011042156815528871 PSNR 35.52086563110352 [27/12 15:25:57]

# [ITER 4700] Saving Gaussians [27/12 15:25:57]
# Training progress:  80%|▊| 4800/6000 [01:41<00:24, 49.67it/s, Loss=0.0124, loss_pointnum torch.Size([206920, 3]) [27/12 15:26:00]

# [ITER 4800] Evaluating test: L1 0.014157064259052277 PSNR 34.51310729980469 [27/12 15:26:00]

# [ITER 4800] Evaluating train: L1 0.010956558585166932 PSNR 35.57766342163086 [27/12 15:26:01]

# [ITER 4800] Saving Gaussians [27/12 15:26:01]
# Training progress:  82%|▊| 4900/6000 [01:45<00:22, 49.33it/s, Loss=0.0159, loss_pointnum torch.Size([208043, 3]) [27/12 15:26:03]

# [ITER 4900] Evaluating test: L1 0.01485732663422823 PSNR 34.204505920410156 [27/12 15:26:04]

# [ITER 4900] Evaluating train: L1 0.010961326211690903 PSNR 35.632537841796875 [27/12 15:26:05]

# [ITER 4900] Saving Gaussians [27/12 15:26:05]
# Training progress:  83%|▊| 5000/6000 [01:49<00:20, 49.39it/s, Loss=0.0145, loss_pointnum torch.Size([209158, 3]) [27/12 15:26:07]

# [ITER 5000] Evaluating test: L1 0.014632071368396282 PSNR 34.31340408325195 [27/12 15:26:08]

# [ITER 5000] Evaluating train: L1 0.011002602614462377 PSNR 35.579359436035155 [27/12 15:26:09]

# [ITER 5000] Saving Gaussians [27/12 15:26:09]
# Training progress:  87%|▊| 5200/6000 [01:55<00:14, 56.54it/s, Loss=0.0161, loss_pointnum torch.Size([211582, 3]) [27/12 15:26:13]

# [ITER 5200] Evaluating test: L1 0.014529681764543056 PSNR 34.358943939208984 [27/12 15:26:13]

# [ITER 5200] Evaluating train: L1 0.010916325263679028 PSNR 35.709545898437504 [27/12 15:26:15]

# [ITER 5200] Saving Gaussians [27/12 15:26:15]
# Training progress:  90%|▉| 5400/6000 [02:01<00:10, 56.37it/s, Loss=0.0146, loss_pointnum torch.Size([213548, 3]) [27/12 15:26:19]

# [ITER 5400] Evaluating test: L1 0.014297617599368095 PSNR 34.44612121582031 [27/12 15:26:19]

# [ITER 5400] Evaluating train: L1 0.011329840496182443 PSNR 35.32296028137207 [27/12 15:26:20]

# [ITER 5400] Saving Gaussians [27/12 15:26:20]
# Training progress:  92%|▉| 5500/6000 [02:04<00:10, 48.97it/s, Loss=0.0165, loss_pointnum torch.Size([214640, 3]) [27/12 15:26:23]

# [ITER 5500] Evaluating test: L1 0.01400015689432621 PSNR 34.62771224975586 [27/12 15:26:23]

# [ITER 5500] Evaluating train: L1 0.010699324123561383 PSNR 35.807928466796874 [27/12 15:26:24]

# [ITER 5500] Saving Gaussians [27/12 15:26:24]
# Training progress:  93%|▉| 5600/6000 [02:08<00:08, 48.86it/s, Loss=0.0127, loss_pointnum torch.Size([215521, 3]) [27/12 15:26:27]

# [ITER 5600] Evaluating test: L1 0.014575829729437828 PSNR 34.30587387084961 [27/12 15:26:27]

# [ITER 5600] Evaluating train: L1 0.010404003597795964 PSNR 36.089735412597655 [27/12 15:26:28]

# [ITER 5600] Saving Gaussians [27/12 15:26:28]
# Training progress:  95%|▉| 5700/6000 [02:12<00:06, 48.78it/s, Loss=0.0146, loss_pointnum torch.Size([216621, 3]) [27/12 15:26:31]

# [ITER 5700] Evaluating test: L1 0.014085117727518082 PSNR 34.58765411376953 [27/12 15:26:31]

# [ITER 5700] Evaluating train: L1 0.011073575913906099 PSNR 35.63958930969238 [27/12 15:26:32]

# [ITER 5700] Saving Gaussians [27/12 15:26:32]
# Training progress:  97%|▉| 5800/6000 [02:16<00:04, 48.61it/s, Loss=0.0130, loss_pointnum torch.Size([217567, 3]) [27/12 15:26:35]

# [ITER 5800] Evaluating test: L1 0.01410159282386303 PSNR 34.52891540527344 [27/12 15:26:35]

# [ITER 5800] Evaluating train: L1 0.010551974549889565 PSNR 36.069070434570314 [27/12 15:26:36]

# [ITER 5800] Saving Gaussians [27/12 15:26:36]
# Training progress:  98%|▉| 5900/6000 [02:20<00:02, 48.76it/s, Loss=0.0206, loss_pointnum torch.Size([218654, 3]) [27/12 15:26:39]

# [ITER 5900] Evaluating test: L1 0.014334029518067837 PSNR 34.46018981933594 [27/12 15:26:39]

# [ITER 5900] Evaluating train: L1 0.010781554691493512 PSNR 35.899009704589844 [27/12 15:26:40]

# [ITER 5900] Saving Gaussians [27/12 15:26:40]
# Training progress: 100%|█| 6000/6000 [02:24<00:00, 41.40it/s, Loss=0.0145, loss_
# pointnum torch.Size([219357, 3]) [27/12 15:26:43]

# [ITER 6000] Evaluating test: L1 0.014289380051195621 PSNR 34.48678970336914 [27/12 15:26:43]

# [ITER 6000] Evaluating train: L1 0.010737073048949242 PSNR 35.859931564331056 [27/12 15:26:44]

# [ITER 6000] Saving Gaussians [27/12 15:26:44]

# Training complete. [27/12 15:26:45]
# Looking for config file in /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test/cfg_args
# Config file found: /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test/cfg_args
# rade Rendering /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test/
# Loading trained model at iteration 5000 [27/12 15:26:47]
# N3D [27/12 15:26:47]
# Reading camera 21/21 [27/12 15:26:47]
# cameras extent: 6.467737865447998 [27/12 15:26:47]
# Loading Training Cameras: 20 . [27/12 15:26:50]
# Loading Test Cameras: 1 . [27/12 15:26:50]

# [ITER 5000] Evaluating test: L1 0.014632071368396282 PSNR 34.31340408325195 [27/12 15:26:50]

# [ITER 5000] Evaluating train: L1 0.011313278251327575 PSNR 36.120182323455815 [27/12 15:26:50]
# After prune iteration, number of gaussians: 115037 [27/12 15:26:51]

# [ITER 5000] Evaluating test: L1 0.015327272936701775 PSNR 33.30660629272461 [27/12 15:26:51]

# [ITER 5000] Evaluating train: L1 0.014016351662576199 PSNR 32.5481258392334 [27/12 15:26:51]
# Training progress:  10%| | 100/1000 [00:01<00:14, 62.81it/s, Loss=0.0138, loss_n
# [ITER 5100] Evaluating test: L1 0.01461217924952507 PSNR 34.284332275390625 [27/12 15:26:53]

# [ITER 5100] Evaluating train: L1 0.011996449274010957 PSNR 35.081270122528075 [27/12 15:26:53]

# [ITER 5100] Saving Gaussians [27/12 15:26:53]
# Training progress:  20%|▏| 200/1000 [00:03<00:13, 60.82it/s, Loss=0.0173, loss_n
# [ITER 5200] Evaluating test: L1 0.014330845326185226 PSNR 34.44061279296875 [27/12 15:26:55]

# [ITER 5200] Evaluating train: L1 0.011565102403983474 PSNR 35.597604751586914 [27/12 15:26:55]

# [ITER 5200] Saving Gaussians [27/12 15:26:55]
# Training progress:  30%|▎| 300/1000 [00:05<00:11, 60.76it/s, Loss=0.0149, loss_n
# [ITER 5300] Evaluating test: L1 0.014403317123651505 PSNR 34.40465545654297 [27/12 15:26:57]

# [ITER 5300] Evaluating train: L1 0.011375292111188174 PSNR 35.820525360107425 [27/12 15:26:57]

# [ITER 5300] Saving Gaussians [27/12 15:26:57]
# Training progress:  40%|▍| 400/1000 [00:07<00:09, 60.56it/s, Loss=0.0147, loss_n
# [ITER 5400] Evaluating test: L1 0.014913583174347878 PSNR 34.15825653076172 [27/12 15:26:59]

# [ITER 5400] Evaluating train: L1 0.011188932345248759 PSNR 35.989403343200685 [27/12 15:26:59]

# [ITER 5400] Saving Gaussians [27/12 15:26:59]
# Training progress:  50%|▌| 500/1000 [00:09<00:08, 60.20it/s, Loss=0.0179, loss_n
# [ITER 5500] Evaluating test: L1 0.014091473072767258 PSNR 34.56049346923828 [27/12 15:27:01]

# [ITER 5500] Evaluating train: L1 0.011091273557394744 PSNR 36.11428623199463 [27/12 15:27:01]

# [ITER 5500] Saving Gaussians [27/12 15:27:01]
# Training progress:  60%|▌| 600/1000 [00:11<00:06, 59.97it/s, Loss=0.0172, loss_n
# [ITER 5600] Evaluating test: L1 0.014433244243264198 PSNR 34.400272369384766 [27/12 15:27:03]

# [ITER 5600] Evaluating train: L1 0.010983247356489301 PSNR 36.20483875274658 [27/12 15:27:03]

# [ITER 5600] Saving Gaussians [27/12 15:27:03]
# Training progress:  70%|▋| 700/1000 [00:13<00:04, 60.11it/s, Loss=0.0146, loss_n
# [ITER 5700] Evaluating test: L1 0.014722230844199657 PSNR 34.25836181640625 [27/12 15:27:05]

# [ITER 5700] Evaluating train: L1 0.01087366263382137 PSNR 36.26221523284912 [27/12 15:27:05]

# [ITER 5700] Saving Gaussians [27/12 15:27:05]
# Training progress:  80%|▊| 800/1000 [00:15<00:03, 59.74it/s, Loss=0.0205, loss_n
# [ITER 5800] Evaluating test: L1 0.014244272373616695 PSNR 34.490020751953125 [27/12 15:27:07]

# [ITER 5800] Evaluating train: L1 0.010805692849680783 PSNR 36.359512424468996 [27/12 15:27:07]

# [ITER 5800] Saving Gaussians [27/12 15:27:07]
# Training progress:  90%|▉| 900/1000 [00:17<00:01, 59.70it/s, Loss=0.0131, loss_n
# [ITER 5900] Evaluating test: L1 0.014652745798230171 PSNR 34.25370788574219 [27/12 15:27:09]

# [ITER 5900] Evaluating train: L1 0.010696570412255824 PSNR 36.38997220993042 [27/12 15:27:09]

# [ITER 5900] Saving Gaussians [27/12 15:27:09]
# Training progress: 100%|█| 1000/1000 [00:19<00:00, 50.19it/s, Loss=0.0197, loss_

# [ITER 6000] Evaluating test: L1 0.014401869848370552 PSNR 34.436485290527344 [27/12 15:27:11]

# [ITER 6000] Evaluating train: L1 0.010685850610025228 PSNR 36.494611549377446 [27/12 15:27:11]

# [ITER 6000] Saving Gaussians [27/12 15:27:11]
# Looking for config file in /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test/cfg_args
# Config file found: /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test/cfg_args
# rade Rendering /media/cgvmis418/2TBSSD/IGS/dataset/sear_steak/colmap_0/test
# Loading trained model at iteration 6000_compress [27/12 15:27:14]
# N3D [27/12 15:27:14]
# Reading camera 21/21 [27/12 15:27:14]
# cameras extent: 6.467737865447998 [27/12 15:27:14]
# Loading Training Cameras: 20 . [27/12 15:27:16]
# Loading Test Cameras: 1 . [27/12 15:27:16]
# Rendering progress:   0%|                                | 0/20 [00:00<?, ?it/s]0.0 [27/12 15:27:16]
# Rendering progress:   5%|█▏                      | 1/20 [00:00<00:09,  2.05it/s]0.0 [27/12 15:27:17]
# Rendering progress:  10%|██▍                     | 2/20 [00:00<00:08,  2.16it/s]0.0 [27/12 15:27:17]
# Rendering progress:  15%|███▌                    | 3/20 [00:01<00:07,  2.21it/s]0.0 [27/12 15:27:18]
# Rendering progress:  20%|████▊                   | 4/20 [00:01<00:07,  2.26it/s]0.0 [27/12 15:27:18]
# Rendering progress:  25%|██████                  | 5/20 [00:02<00:06,  2.31it/s]0.0 [27/12 15:27:18]
# Rendering progress:  30%|███████▏                | 6/20 [00:02<00:06,  2.33it/s]0.0 [27/12 15:27:19]
# Rendering progress:  35%|████████▍               | 7/20 [00:03<00:05,  2.31it/s]0.0 [27/12 15:27:19]
# Rendering progress:  40%|█████████▌              | 8/20 [00:03<00:05,  2.29it/s]0.0 [27/12 15:27:20]
# Rendering progress:  45%|██████████▊             | 9/20 [00:03<00:04,  2.27it/s]0.0 [27/12 15:27:20]
# Rendering progress:  50%|███████████▌           | 10/20 [00:04<00:04,  2.24it/s]0.0 [27/12 15:27:21]
# Rendering progress:  55%|████████████▋          | 11/20 [00:04<00:04,  2.23it/s]0.0 [27/12 15:27:21]
# Rendering progress:  60%|█████████████▊         | 12/20 [00:05<00:03,  2.21it/s]0.0 [27/12 15:27:22]
# Rendering progress:  65%|██████████████▉        | 13/20 [00:05<00:03,  2.24it/s]0.0 [27/12 15:27:22]
# Rendering progress:  70%|████████████████       | 14/20 [00:06<00:02,  2.25it/s]0.0 [27/12 15:27:22]
# Rendering progress:  75%|█████████████████▎     | 15/20 [00:06<00:02,  2.26it/s]0.0 [27/12 15:27:23]
# Rendering progress:  80%|██████████████████▍    | 16/20 [00:07<00:01,  2.26it/s]0.0 [27/12 15:27:23]
# Rendering progress:  85%|███████████████████▌   | 17/20 [00:07<00:01,  2.25it/s]0.0 [27/12 15:27:24]
# Rendering progress:  90%|████████████████████▋  | 18/20 [00:08<00:00,  2.23it/s]0.0 [27/12 15:27:24]
# Rendering progress:  95%|█████████████████████▊ | 19/20 [00:08<00:00,  2.26it/s]0.0 [27/12 15:27:25]
# Rendering progress: 100%|███████████████████████| 20/20 [00:08<00:00,  2.26it/s]
# Rendering progress:   0%|                                 | 0/1 [00:00<?, ?it/s]0.0 [27/12 15:27:25]
# Rendering progress: 100%|█████████████████████████| 1/1 [00:00<00:00,  2.33it/s]
# ------------------------------------------
# 您的指令已執行完畢。
# VRAM 監控已停止。
# 分析 VRAM 使用量...
# VRAM 使用量分析結果 (針對整個執行過程):
#  - 高峰值 (Peak): 6491.00 MiB
#  - 平均值 (Average): 2529.55 MiB
#  - 總共紀錄 20 筆有效資料
# 程式執行時間分析:
#  - 總執行時間: 3 分 15 秒 (共 195 秒)
# (IGS) cgvmis418@cgvmis418-System-Product-Name:/media/cgvmis418/2TBSSD/IGS/submodules/RaDe-GS$ 
