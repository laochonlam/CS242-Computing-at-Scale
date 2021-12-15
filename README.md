<h1 align="center">
  Improving Inference Performance through Object-Based Explanation (OBE).
</h1>

## Description
This code repository is used to simulate zeroing out parts of the ImageNet dataset and the ResNet50 model based on OBE and evaluate its performance. It is primarily written in Python. 
We have two main folders for two different workstreams: (1) Heatmap Generaiton and (2) Zeroing Out Model.

## Heatmap Generation
Folder heatmap_generate works by generating 8x8 heatmaps for each layer of ResNet50 through Object-Based Explanation (OBE), allowing us to identify which regions in each layer are least important and ones that we can zero out. Layer-heatmap results are stored in heatmap_results.

## Zeroing Out Model
Folder model_zero_out. In erase_experiment_imagenet.py, we add a hook through every layer and then run inference on the model to evaluate the new model's accuracy with the hook. The hook class has a function called skip_computation_pre_layer which is our implementation of each areas of a layer to zero out based on that layer's heatmap. Accuracy results are stored in results.

## Usage

TODO: Chonlam