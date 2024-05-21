
from .ist_utils import (partition_FC_layer_by_output_dim_0, 
                       partition_FC_layer_by_dim_01, 
                       partition_FC_layer_by_input_dim_1, 
                       partition_BN_layer, 
                       update_tensor_by_update_lists_dim_0, 
                       update_tensor_by_update_lists_dim_01, 
                       update_tensor_by_update_lists_dim_1
                       )
from torch import nn
import torch
import random
import logging

class IndependentSubNetworks(nn.Module):
    def __init__(self, layer_dims, partition_num, input_size, label_num, track_bn_stats=False):
        super(IndependentSubNetworks, self).__init__()
        logging.debug("Making network with layer_dims: {} partition: {}".format(layer_dims, partition_num))
        assert all(dim % partition_num == 0 for dim in layer_dims[1:]), "All elements from the 2nd onwards must be divisible by the partition number"        
        self.layer_dims = layer_dims
        self.partition_num = partition_num
        self.input_size = input_size
        self.label_num = label_num
        self.partition_dims =[dim // partition_num for dim in layer_dims]

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.layer_weight_partitions = []
        self.bn_weight_partitions = []
        self.bn_bias_partitions = []
        self.hidden_indices = []

        self.partition_lists = []

        for in_dim, out_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            linear_layer = nn.Linear(in_dim, out_dim, bias=False)
            batch_norm = nn.BatchNorm1d(out_dim, momentum=1.0, affine=True, track_running_stats=track_bn_stats)
            self.layers.append(linear_layer)
            self.batch_norms.append(batch_norm)
        
        self.output_layer = nn.Linear(self.layer_dims[-1], self.label_num, bias=False)
        self.bn_output = nn.BatchNorm1d(self.label_num, momentum=1.0, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.flatten(x)
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = layer(x)
            x = batch_norm(x)
            x = nn.functional.relu(x, inplace=True)
        x = self.output_layer(x)
        x = self.bn_output(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

    def repartition_using_existing_list(self):
        self.layer_weight_partitions.clear()
        self.bn_weight_partitions.clear()
        self.bn_bias_partitions.clear()

        for i, layer in enumerate(self.layers):
            current_indices = self.partition_lists[i]
            if i==0: # this s the first layer
                current_layer_weight_partitions = partition_FC_layer_by_output_dim_0(
                    layer.weight, current_indices)
                current_bn_weight_partitions, current_bn_bias_partitions = partition_BN_layer(
                    self.batch_norms[i].weight, self.batch_norms[i].bias, current_indices)
                self.layer_weight_partitions.append(current_layer_weight_partitions)
                self.bn_weight_partitions.append(current_bn_weight_partitions)
                self.bn_bias_partitions.append(current_bn_bias_partitions)
            else:
                past_indices = self.partition_lists[i-1]
                current_layer_weight_partitions = partition_FC_layer_by_dim_01(
                    layer.weight, current_indices, past_indices)
                current_bn_weight_partitions, current_bn_bias_partitions = partition_BN_layer(
                    self.batch_norms[i].weight, self.batch_norms[i].bias, current_indices)
                self.layer_weight_partitions.append(current_layer_weight_partitions)
                self.bn_weight_partitions.append(current_bn_weight_partitions)
                self.bn_bias_partitions.append(current_bn_bias_partitions)
            
            # if this is the last layer
            if i == len(self.layers) - 1:
                final_layer_weight_partitions = partition_FC_layer_by_input_dim_1(
                    self.output_layer.weight, current_indices)
                self.layer_weight_partitions.append(final_layer_weight_partitions)
                        
    def partition_to_list(self):
        # first geenrate the partition indices
        self.partition_lists.clear()
        self.hidden_indices.clear()
        for i, layer in enumerate(self.layers):
            hidden_layer_index = [i for i in range(layer.out_features)]
            random.shuffle(hidden_layer_index)
            self.hidden_indices.append(hidden_layer_index)
            current_indices = []
            for j in range(self.partition_num): # why does this loop over all the partitions when the first layer is input?
                current_index = (torch.tensor(
                    hidden_layer_index[j * self.partition_dims[i+1]: (j+1) * self.partition_dims[i+1]]
                ))
                current_indices.append(current_index) # this should be sent to device
            self.partition_lists.append(current_indices)
    
        # we have a list indexed by layer that each contains a list of tensors that are the indices for each partition
        self.layer_weight_partitions.clear()
        self.bn_weight_partitions.clear()
        self.bn_bias_partitions.clear()

        for i, layer in enumerate(self.layers):
            current_indices = self.partition_lists[i]
            if i==0: # this s the first layer
                current_layer_weight_partitions = partition_FC_layer_by_output_dim_0(
                    layer.weight, current_indices)
                current_bn_weight_partitions, current_bn_bias_partitions = partition_BN_layer(
                    self.batch_norms[i].weight, self.batch_norms[i].bias, current_indices)
                self.layer_weight_partitions.append(current_layer_weight_partitions)
                self.bn_weight_partitions.append(current_bn_weight_partitions)
                self.bn_bias_partitions.append(current_bn_bias_partitions)
            else:
                past_indices = self.partition_lists[i-1]
                current_layer_weight_partitions = partition_FC_layer_by_dim_01(
                    layer.weight, current_indices, past_indices)
                current_bn_weight_partitions, current_bn_bias_partitions = partition_BN_layer(
                    self.batch_norms[i].weight, self.batch_norms[i].bias, current_indices)
                self.layer_weight_partitions.append(current_layer_weight_partitions)
                self.bn_weight_partitions.append(current_bn_weight_partitions)
                self.bn_bias_partitions.append(current_bn_bias_partitions)
            
            # if this is the last layer
            if i == len(self.layers) - 1:
                final_layer_weight_partitions = partition_FC_layer_by_input_dim_1(
                    self.output_layer.weight, current_indices)
                self.layer_weight_partitions.append(final_layer_weight_partitions)

    def get_indices_for_partition(self, partition_num):
        # Check if the partition number is valid
        assert 0 <= partition_num < self.partition_num, "Invalid partition number"

        # Get the indices for all layers for the given partition
        indices_for_partition = [layer_indices[partition_num] for layer_indices in self.partition_lists]

        return indices_for_partition
        
    def print_layer_partition_lists(self):
        for i, partitions in enumerate(self.partition_lists):
            print("Layer {} partitions: {}".format(i, partitions))

    def print_subnetwork_weight_matrices(self):
        print("Layer weight partitions: {}".format(self.layer_weight_partitions))
        for i, partitions in enumerate(self.layer_weight_partitions):
            if i != len(self.layers):
                print("Layer {} weight partitions: {}".format(i, self.layerse[i].weight.data))
            else:
                print("Layer {} weight partitions: {}".format(i, self.output_layer.weight.data))    
            for j, layter_subnetwork in enumerate(partitions):
                print("Layer {} subnetwork {} weight: {}".format(i, j, layter_subnetwork))

    def update_sub_component(self, subnetwork_state_dict, subnetwork_indices):
    #   logging.info("Updating resident model for client {} {}".format(self.client_idx, len(self.model_indices)))
    #   exit()

      #for updating the first few layers
      for i in range(len(subnetwork_indices)): # do this for each of the layers and the output layer
        target_layer_weight_dict_key = f"layers.{i}.weight"
        target_bn_weight_dict_key = f"batch_norms.{i}.weight"
        target_bn_bias_dict_key = f"batch_norms.{i}.bias"

        # logging.info(f"{target_layer_weight_dict_key} {target_bn_weight_dict_key} {target_bn_bias_dict_key}")

        current_indices = subnetwork_indices[i] # implement code for actually populating this

        if i == 0: # this it the first layer
          update_tensor_by_update_lists_dim_0(self.state_dict()[target_layer_weight_dict_key],
                                              [subnetwork_state_dict[target_layer_weight_dict_key].clone()],
                                              [current_indices])
          update_tensor_by_update_lists_dim_0(self.state_dict()[target_bn_weight_dict_key],
                                              [subnetwork_state_dict[target_bn_weight_dict_key].clone()],
                                              [current_indices])
          update_tensor_by_update_lists_dim_0(self.state_dict()[target_bn_bias_dict_key],
                                              [subnetwork_state_dict[target_bn_bias_dict_key].clone()],
                                              [current_indices])
        else: # otherwise if this is an intermediate
          previous_indices = subnetwork_indices[i-1] # implemente method for getting the previous list of indices
          update_tensor_by_update_lists_dim_01(self.state_dict()[target_layer_weight_dict_key],
                                               [subnetwork_state_dict[target_layer_weight_dict_key].clone()],
                                               [current_indices],
                                               [previous_indices])
          update_tensor_by_update_lists_dim_0(self.state_dict()[target_bn_weight_dict_key],
                                              [subnetwork_state_dict[target_bn_weight_dict_key].clone()],
                                              [current_indices])
          update_tensor_by_update_lists_dim_0(self.state_dict()[target_bn_bias_dict_key],
                                              [subnetwork_state_dict[target_bn_bias_dict_key].clone()],
                                              [current_indices])
        if(i == len(subnetwork_indices)-1): # if this is the last hidden layer
            output_layer_weight_dict_key = "output_layer.weight"
            update_tensor_by_update_lists_dim_1(self.state_dict()[output_layer_weight_dict_key],
                                                [subnetwork_state_dict[output_layer_weight_dict_key].clone()],
                                                [current_indices])
      return 