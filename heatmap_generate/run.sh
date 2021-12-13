# Imagenet
python3 erase_experiment_imagenet.py -a resnet50  -j 32 -e --pretrained --heatmap /data/imagenet18_all/data/imagenet/

# CIFAR 10
## TRAIN
python3 erase_experiment_cifar10.py -a resnet50  -j 32 --pretrained --heatmap /data/imagenet18_all/data/imagenet
## EVAL
python3 erase_experiment_cifar10.py -a resnet50  -j 32 --pretrained -e --resume ./model_best.pth.tar --heatmap /data/imagenet18_all/data/imagenet



#For multiple Heatmap
python3 erase_experiment_imagenet.py -a resnet50  -j 32 -b 2048 -e --pretrained ~/imagenet18/data/imagenet/