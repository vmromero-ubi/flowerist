from collections import OrderedDict
import flwr as fl
import random
from flwr.common import Parameters
from flwr.server.client_manager import ClientManager
from typing import (
    List,
    Optional,
    Tuple,
    Dict,
    Union,
)
from models.ist import IndependentSubNetworks
from fl_utils import get_parameters, set_parameters
from flwr.common import(
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from logging import INFO, DEBUG
from flwr.common.logger import log
import logging
import copy
import numpy as np
import torch
from models.ist_utils import (
    update_tensor_by_update_lists_dim_0,
    update_tensor_by_update_lists_dim_01,
    update_tensor_by_update_lists_dim_1,

)

# from ..trainer import train, test
# from flowerist.trainer import test

class FedIST(fl.server.strategy.Strategy):
    """ Federated training strategy based on independent subnetwork training. """
    def __init__(
            self, 
            num_participants: int = 100,
            model_partitions: int = 100, 
            conflict_resolution: str = "aggregate",
            ist_type: str = "legacy",
            layer_dims: list[int] = None, # used for creating the full network that will be trained
            label_num: int = 10,
            global_test_data_loader = None,
    ) -> None:
        super().__init__()
        self.num_participants = num_participants
        self.model_partitions = model_partitions
        self.conflict_resolution = conflict_resolution
        self.ist_type = ist_type
        if layer_dims is None:
            layer_dims = [784, 5000] # set default value when not provided
        self.layer_dims = layer_dims
        self.label_num = label_num
        self.global_test_data_loader = global_test_data_loader

        self.track_bn_stats = False
        self.oracle_model = None
        self.model_assignment = None # the last known subnetwork assignments
        self.historical_model_assignments = [] # keep track of subnetwork assignments through training rounds
        
        self.subnetwork_template = IndependentSubNetworks(
            layer_dims = [self.layer_dims[0] if i == 0 else dim // self.model_partitions for i, dim in enumerate(self.layer_dims)],
            partition_num=1,
            input_size=self.layer_dims[0],
            label_num=self.label_num,
            track_bn_stats=self.track_bn_stats, 
        )

        self.oracle_model_state_dict = None

        # end of __init__

    def __repr__(self) -> str:
        return "FedIST"
    
    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Optional[Parameters]:
        """
        Initialize the full neural network parameters that will be trained.

        Args:
            client_manager (ClientManager): The client manager object.

        Returns:
            Optional[Parameters]: The initialized parameters for the neural network.
        """
        # Create the oracle network that will be trained collaboratively
        self.oracle_model = IndependentSubNetworks(
            layer_dims=self.layer_dims,
            partition_num=self.model_partitions,
            input_size=self.layer_dims[0],
            label_num=self.label_num,
            track_bn_stats=self.track_bn_stats, 
        )

        self.oracle_model_state_dict = self.oracle_model.state_dict()

        self.oracle_model.partition_to_list() # partition the network for assignment later on

        ndarrays = get_parameters(self.oracle_model)
        return ndarrays_to_parameters(ndarrays)
    
    def configure_fit(
            self, 
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager, 
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        # this is the function called before each training round
        # this is reponsible for:
        # selecting the nodes that will participate in updating model parameters
        # configuring the training parameters for each node
        # classically, our simulation assumes that all nodes participate in each round
        # we can issue a check to determine if the parameters in this class matches with the servers
        model_partition_ids = [i for i in range(self.model_partitions)]
        self.model_assignment = [model_partition_ids[i % self.model_partitions] for i in range(self.num_participants)]
        random.shuffle(self.model_assignment)
        self.historical_model_assignments.append(copy.deepcopy(self.model_assignment))
        # print(f"model assignment for round {server_round}: {self.model_assignment}")
        log(INFO, f"model assignment for round {server_round}: {self.model_assignment}")
        clients = client_manager.sample(
            num_clients=self.num_participants,
            min_num_clients=self.num_participants,
        ) # just get them all

        fit_configurations = []
        standard_configuration = {"lr": 0.01}
        for idx, client in enumerate(clients):
            this_client_model = IndependentSubNetworks(
                # layer_dims= [dim // self.num_participants if i!=0 else dim[0] for i, dim in enumerate(self.layer_dims)] self.layer_dims,
                layer_dims = [self.layer_dims[0] if i == 0 else dim // self.model_partitions for i, dim in enumerate(self.layer_dims)],
                partition_num=1,
                input_size=self.layer_dims[0],
                label_num=self.label_num,
                track_bn_stats=self.track_bn_stats,
            )
            client_partition_idx = self.model_assignment[int(client.cid)]
            # print(f"client {client.cid} assigned to partition {client_partition_idx} based on idx {idx}")
            
            # populate the model subnetwork with parameters from the oracle model
            for j, _ in enumerate(this_client_model.layers):  # for each layer of the child network
                this_client_model.layers[j].weight.data = self.oracle_model.layer_weight_partitions[j][client_partition_idx].clone()
                this_client_model.batch_norms[j].weight.data = self.oracle_model.bn_weight_partitions[j][client_partition_idx].clone()
                this_client_model.batch_norms[j].bias.data = self.oracle_model.bn_bias_partitions[j][client_partition_idx].clone()
            this_client_model.output_layer.weight.data = self.oracle_model.layer_weight_partitions[len(this_client_model.layers)][client_partition_idx].clone()
            
            this_client_model_ndarray = get_parameters(this_client_model)
            client_parameters = fl.common.ndarrays_to_parameters(this_client_model_ndarray)
            fit_configurations.append((client, FitIns(client_parameters, standard_configuration)))
        return fit_configurations
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Reconstitute the oracle model with model aggregation support."""
        grouped_models = {}
        aggregated_models = {}
        unique_indices = {}

        for client, fit_result in results: # for every fit resut we received
            # current_model_state_dict = client.get_model()
            current_model_parameters = fit_result.parameters
            current_model_partition_id = self.model_assignment[int(client.cid)]
            current_model_indices = copy.deepcopy(self.oracle_model.get_indices_for_partition(current_model_partition_id))
            

            if current_model_partition_id not in grouped_models: 
                assert current_model_partition_id not in unique_indices, "Expected both model and indices to be unaccounted for"
                grouped_models[current_model_partition_id] = [current_model_parameters] 
                unique_indices[current_model_partition_id] = current_model_indices
            else:
                grouped_models[current_model_partition_id].append(current_model_parameters)

        # logging.debug("partition ids : {}".format(grouped_models.keys()))
        log(INFO, f"partition ids : {grouped_models.keys()}")
        for group_part_id, group_model_list in grouped_models.items():
            aggregated_models[group_part_id] = ndarrays_to_parameters(
                aggregate(
                    [
                        (parameters_to_ndarrays(model), 1) for model in group_model_list
                    ]
                )
            )

        for current_group_part_id, current_group_model_parameters in aggregated_models.items():
            current_group_indices = unique_indices[current_group_part_id]
            current_group_model = self.get_state_dict_representation(
                self.subnetwork_template,
                parameters_to_ndarrays(current_group_model_parameters)
            )

            for i, current_indices in enumerate(current_group_indices):
                target_layer_weight_dict_key = f"layers.{i}.weight"
                target_bn_weight_dict_key = f"batch_norms.{i}.weight"
                target_bn_bias_dict_key = f"batch_norms.{i}.bias"

                current_indices = current_group_indices[i] 
                if i == 0: 
                    update_tensor_by_update_lists_dim_0(self.oracle_model_state_dict[target_layer_weight_dict_key],
                                                        [current_group_model[target_layer_weight_dict_key].clone().detach()],
                                                        [current_indices])
                    update_tensor_by_update_lists_dim_0(self.oracle_model_state_dict[target_bn_weight_dict_key],
                                                        [current_group_model[target_bn_weight_dict_key].clone().detach()],
                                                        [current_indices])
                    update_tensor_by_update_lists_dim_0(self.oracle_model_state_dict[target_bn_bias_dict_key],
                                                        [current_group_model[target_bn_bias_dict_key].clone().detach()],
                                                        [current_indices])
                else: 
                    previous_indices = current_group_indices[i-1] 
                    update_tensor_by_update_lists_dim_01(self.oracle_model_state_dict[target_layer_weight_dict_key],
                                                        [current_group_model[target_layer_weight_dict_key].clone().detach()],
                                                        [current_indices],
                                                        [previous_indices])
                    update_tensor_by_update_lists_dim_0(self.oracle_model_state_dict[target_bn_weight_dict_key],
                                                        [current_group_model[target_bn_weight_dict_key].clone().detach()],
                                                        [current_indices])
                    update_tensor_by_update_lists_dim_0(self.oracle_model_state_dict[target_bn_bias_dict_key],
                                                        [current_group_model[target_bn_bias_dict_key].clone().detach()],
                                                        [current_indices])
                if(i == len(current_model_indices)-1): 
                    output_layer_weight_dict_key = "output_layer.weight"
                    update_tensor_by_update_lists_dim_1(self.oracle_model_state_dict[output_layer_weight_dict_key],
                                                        [current_group_model[output_layer_weight_dict_key].clone().detach()],
                                                        [current_indices])
        self.oracle_model.load_state_dict(self.oracle_model_state_dict)
        metrics_aggregated = {}
        return ndarrays_to_parameters(get_parameters(self.oracle_model)), metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        config = {}
        evaluate_ins = EvaluateIns(parameters, config)        

        clients = client_manager.sample(
            num_clients=self.num_participants,
            min_num_clients=self.num_participants,
        ) # just get them all

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        # print(f"calling evaluate function for round {server_round}")
        if self.global_test_data_loader is not None:
            loss, accuracy = test(self.oracle_model, self.global_test_data_loader)
            # print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
            log(INFO, "Server-side evaluation loss %s / accuracy %s", loss, accuracy)

        # Let's assume we won't perform the global model evaluation on the server side.
        return None
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    
    def get_state_dict_representation(self, net, parameters:List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        return copy.deepcopy(state_dict)

# end of fedist.py

def test(net, testloader, device: str = "cpu"):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


