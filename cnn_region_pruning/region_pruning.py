# CS242 Assignment 2 Project
import torch

torch.manual_seed(43) # to give stable randomness

def get_sparse_conv2d_layers(net):
    '''
    Helper function which returns all SparseConv2d layers in the net.
    Use this below to implement layerwise pruning.
    '''
    sparse_conv_layers = []
    for layer in net.children():
        if isinstance(layer, SparseConv2d):
            sparse_conv_layers.append(layer)
        else:
            child_layers = get_sparse_conv2d_layers(layer)
            sparse_conv_layers.extend(child_layers)
    
    return sparse_conv_layers

def filter_l1_pruning(net, prune_percent):
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        num_nonzero = layer._mask.sum().item()
        num_total = len(layer._mask)
        num_prune = round(num_total * prune_percent)
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print(num_prune, num_total, prune_percent)
        
        # PART 3.3: Implement pruning by settings elements in layer._mask
        #           to zero corresponding to the smallest l1-norm filters
        #           in layer._weight. Note: to update variable such as
        #           layer._mask and layer._weight, do the following:
        #           layer._mask.data[idx] = 0

        # #number of filters to prune : prune percent * total number of filters in layer
        # #indices = sorted(range(num_total), key = lambda sub: layer._weight[sub])[:num_prune]
        # #idx = np.argpartition(layer._weight, num_prune)
        # #print(indices)
        # print(-torch.topk(-layer._weight, k = num_prune))

        # pass

        # ---
        # Sort by smallest l1-norm filters in layer._weight and their index (sort by weight)
        indices = torch.argsort(layer._weight)[:num_prune]

        # Prune step for this layer - Iterate through list of indices and set to 0
        for idx in indices:
          layer._mask.data[idx] = 0



# Final Project Assignment of pruning based on region
def calculate_heatmap_value_threshold(prune_percent, heatmap):
    # TODO: Figure out how to allocate heatmap values to the weights in the layer
    n = len(heatmap)
    threshold_index = n - int(n * prune_percent)
    return heatmap.sort()[threshold_index]

def filter_region_pruning(net, prune_percent, heatmap):
    '''
    Input
        net = network model
        prune_percent = % you want to prune
        [Assumption] heatmap = [ v1, v2, v3, v4, ...]

    Process:
        Induce regular sparsity across layers

    Output: None because we're just editing layer._mask
    '''

    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        num_nonzero = layer._mask.sum().item()
        num_total = len(layer._mask)
        num_prune = round(num_total * prune_percent)
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print(num_prune, num_total, prune_percent)

        # # [Assumption] If the heatmap is the same size as the layers
        # heatmap_value_threshold = calculate_heatmap_value_threshold(prune_percent, heatmap)

        # for idx in range(0, len(layer._weight)):
        #     # If the heatmap value is low, then prune this weight
        #     weight_heatmap_value = heatmap[idx] # TODO: Change for if heatmap and layers are different sizes
        #     if weight_heatmap_value < heatmap_value_threshold:
        #         layer._mask.data[idx] = 0

        # [Assumption] If the heatmap is smaller than the layers
        heatmap_value_threshold = calculate_heatmap_value_threshold(prune_percent, heatmap)

        for idx in range(0, len(layer._weight)):
            # If the heatmap value is low, then prune this weight
            weight_heatmap_value = heatmap[idx] # TODO: Change for if heatmap and layers are different sizes
            if weight_heatmap_value < heatmap_value_threshold:
                layer._mask.data[idx] = 0

def calculate_tuples_to_prune(prune_percent, heatmap):
    '''
    Input: 2x2 heatmap
    Output: tuples that are indices of which heatmap values to prune
    '''

    n = len(heatmap) * len(heatmap[0])
    threshold_index = n - int(n * prune_percent)
    threshold_value = heatmap.flatten().sort()[threshold_index]

    tuples = []
    for i in range(len(heatmap)):
        for j in range(len(heatmap[0])):
            if heatmap[i][j] < threshold_value:
                tuples.append((i, j))

    return tuples

# Assuming heatmap is 2x2 and weights are at least 2x2
def filter_region_pruning(net, prune_percent, heatmap):
    '''
    Input
        net = network model
        prune_percent = % you want to prune
        [Assumption] heatmap = reshape into a 2x2 [ v1, v2, v3, v4, ...]

    Process:
        Induce regular sparsity across layers

    Output: None because we're just editing layer._mask
    '''

    
    transformed_heatmap = heatmap.reshape()

    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        num_nonzero = layer._mask.sum().item()
        num_total = len(layer._mask)
        num_prune = round(num_total * prune_percent)
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print(num_prune, num_total, prune_percent)

        # [Assumption] If the heatmap is smaller than the layers
        # heatmap_value_threshold = calculate_heatmap_value_threshold(prune_percent, heatmap)

        prune_tuples = calculate_tuples_to_prune(prune_percent, transformed_heatmap)

        for idx in range(0, len(layer._weight)):
            # If the heatmap value is low, then prune this weight
            weight_heatmap_value = heatmap[idx] # TODO: Change for if heatmap and layers are different sizes
            if weight_heatmap_value < heatmap_value_threshold:
                layer._mask.data[idx] = 0

# How to calculate the output dimensions of a Conv2d
# Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1.
# Output width = (Output width + padding width right + padding width left - kernel width) / (stride width) + 1.

