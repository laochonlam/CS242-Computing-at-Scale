<h1 align="center">
  Improving Inference Performance through Object-Based Explanation (OBE)
</h1>

## Description
This code repository is used to demonstrate how we reduce the computation cost by zeroing out layers' input in ResNet50 with ImageNet dataset with the help of heatmaps generated based on OBE. It is primarily written in Python. 
We have two main folders for two different workstreams: (1) Heatmap Generation and (2) Zeroing out Layers' Input of Model.

## Step 1 --- Heatmap Generation
Folder `heatmap_generate` works by generating 8x8 heatmaps for each layer of ResNet50 through Object-Based Explanation (OBE), allowing us to identify which regions in each layer are least important and ones that we can zero out. Layer-wise heatmap results are stored in `heatmap_results`. 
`heatmap_generate_imagenet.py` and `run.sh` are used to generate the heatmaps as discribed in the following section. 

## Step 2 --- Zeroing Out Model
Folder `model_zero_out`. In `erase_experiment_imagenet.py`, we add a hook through every layer to zero out the input of that layer by the heatmaps generated (Step 1).
Then run inference on the model to evaluate the new model's accuracy with the hooks. The hook class has a function called `skip_computation_pre_layer` which is our implementation of which areas of a layer to zero out based on that layer's heatmap. Accuracy results are stored in `results`.


## Usage
For Step 1 (Heatmap Generation):
```
$ ./heatmap_results/run.sh
```


For Step 2 (Zeroing Out Model):
```
$ ./heatmap_results/run.sh
```