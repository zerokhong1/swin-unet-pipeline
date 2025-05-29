Access dataset ISIC2018 at the link:
https://www.kaggle.com/datasets/kookmindoo/isic2018
# pipeline_model_update
# Create output_folder
# 1. Run Script:
## !python train.py --mode evaluate --data "" --checkpoint "" --saveas "" 
# Training:
## !python train.py --mode train --epoch 3 --lr0 0.1 --batchsize 16 --data /kaggle/input/isic2018/ISIC2018 --saveas "/kaggle/working/outputSwinUNet1" 
# Pretrain:
## !python train.py --mode pretrain --epoch 50 --lr0 0.1 --batchsize 16 --data /kaggle/input/isic2018/ISIC2018 --checkpoint /kaggle/working/last_model_down.pth --saveas /kaggle/working/outputSwinUNet1
# Evaluate:
## !python train.py --mode evaluate --data /kaggle/input/isic2018/ISIC2018 --checkpoint /kaggle/working/last_model_down.pth --saveas /kaggle/working/outputSwinUNet1
# 2. Augmentation:
# Training:
## !python train.py --augment --mode train --epoch 3 --lr0 0.1 --batchsize 16 --data /kaggle/input/isic2018/ISIC2018 --saveas "/kaggle/working/outputSwinUNet1" 
# Pretrain:
## !python train.py --augment --mode pretrain --epoch 50 --lr0 0.1 --batchsize 16 --data /kaggle/input/isic2018/ISIC2018 --checkpoint /kaggle/working/last_model_down.pth --saveas /kaggle/working/outputSwinUNet1
