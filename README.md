# Introduction
This code repository stores program implementation for the accepted MobiSys 2023 paper "ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection". According to our paper, the code is intended to be run on embedded devices (e.g., Raspberry Pi and Nvidia Jetson TX2), but also applicable to workstations.

If you are looking for **the core of our implementation**, we suggest you take a look at the following:
* Tensor Timing Profiler -- `profiler.py`
* Tensor Importance Evaluator -- `elastic_training` in `train.py`
* Tensor Selector by Dynamic Programming -- `selection_solver_DP.py`.

**We are still finalizing our code and docs. Please stay tuned for camera-ready release of our paper**.

# Requirements
* Python 3.7+
* tensorflow 2
* tensorflow-datasets
* tensorflow-addons
* [tensorboard_plugin_profile](https://www.tensorflow.org/guide/profiler)
* [vit-keras](https://github.com/faustomorales/vit-keras)
* tqdm

# General Usage
Select NN models and datasets to run. Use `python main.py --help` and `python profiler.py --help` to see configurable parameters.

Supported NN architectures:
* ResNet50 -- `resnet50`
* VGG16 -- `vgg16`
* MobileNetV2 -- `mobilenetv2`
* Vision Transformer (16x16 patch) -- `vit`

Supported datasets:
* [CUB-200 (200 classes)](https://www.vision.caltech.edu/datasets/cub_200_2011/) -- `caltech_birds2011`
* [Oxford-IIIT Pet (37 classes)](https://www.robots.ox.ac.uk/~vgg/data/pets/) -- `oxford_iiit_pet`
* [Stanford Dogs (120 classes)](http://vision.stanford.edu/aditya86/ImageNetDogs/) -- `stanford_dogs`

**Note**: The NN architectures and datasets should be downloaded automatically. We use `tensorflow-datasets` APIs to download datasets from their collection list. If you encounter errors (e.g., checksum error) when downloading datasets, please refer to [instructions](https://www.tensorflow.org/datasets/overview#manual_download_if_download_fails) for manually downloading (not too difficult).

Below shows an example of training ResNet50 on CUB-200 dataset with our ElasticTrainer. First, profile the tensor timing on your dedicated device:
```
python profiler.py --model_name resnet50 \
                   --num_classes 200
```
Then start training your model on the device with speedup ratio of 0.5 (i.e., 2x faster):
```
python main.py --model_name resnet50 \
               --dataset_name caltech_birds2011 \
               --train_type elastic_training \
               --rho 0.5
```
# Artifact Evaluation
We provide experimental workflows that allow people to reproduce our main results (including baselines' results) in the paper. However, running all the experiments could take extremely long time (~800 hours), and thus we mark each experiment with its estimated execution time for users to choose based on their time budget. After you finish running each script, the figure will be automatically generated under `figures/`. For Nvidia Jetson TX2, we run experiments with its text-only interface, and to view the figures, you will need to switch back to the graphic interface.

We first describe how you can prepare the environment that allows you to run our experiments, and then we list command lines to reproduce every figure in our main results.

## Preparing Nvidia Jetson TX2
1. According to our artifact appendix, flash the Jetson using our provided OS image. Insert SSD.
2. Login the system where both username and password are `nvidia`. 
3. Run the following commands to finishs preparation:
```
sudo su -
cd ~/src/ElasticTrainer
chmod +x *.sh
```

## Preparing Raspberry Pi 4B
1. Flash the Raspberry Pi using our provided OS image.
2. Open a terminal and run the following commands to finish preparation:
```
cd ~/src/ElasticTrainer
. ../kai_stl_code/venv/bin/activate
chmod +x *.sh
```

## Figure 15(a)(d) - A minimal reproduction of main results (~10 hours)
On Nvidia Jetson TX2:
```
./run_figure15ad.sh
```
## Figure 15 from (a) to (f) (~33 hours)
On Nvidia Jetson TX2:
```
./run_figure15.sh
```
Alternatively, if you want to exclude baseline schemes, run the following (~6.5 hours):
```
./run_figure15_ego.sh
```
## Figure 16 from (a) to (d) (~221 hours)
On Raspberry Pi 4B:
```
./run_figure16.sh
```
Alternatively, if you want to exclude baseline schemes, run the following (~52 hours):
```
./run_figure16_ego.sh
```
## Figure 17 (a)(c) (~15+190 hours)
Run the following command on both Nvidia Jetson TX2 (~15 hours) and Raspberry Pi 4B (~190 hours):
```
./run_figure17ac.sh
```
Alternatively, if you want to exclude baseline schemes, run the following command on both Nvidia Jetson TX2 (~9 hours) and Raspberry Pi 4B (~85 hours):
```
./run_figure17ac_ego.sh
```
## Figure 19 from (a) to (d) (~20+310 hours)
Run the following command on both Nvidia Jetson TX2 (~20 hours) and Raspberry Pi 4B (~310 hours):
```
./run_figure19.sh
```
Alternatively, if you want to exclude baseline schemes, run the following command on both Nvidia Jetson TX2 (~3.5 hours) and Raspberry Pi 4B (~50 hours):
```
./run_figure19_ego.sh
```

## Checking Results
All the experiment results should be generated under `figures/`. On Pi, directly click them to view. On Jetson, to check experiments results, you will need to switch to graphic mode:

```
sudo systemctl start graphical.target
``` 
In graphic mode, open a terminal, gain root privilege, and navigate to our code directory:
```
sudo su -
cd ~/src/ElasticTrainer
```
 All the figures are stored under `figures/`. Use `ls` command to check their file names. Use `evince` command to view the figures, for example, `evince xxx.pdf`. To go back to text-only mode, simply reboot the system.
