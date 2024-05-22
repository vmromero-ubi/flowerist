from flwr.common import (
    Code, 
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

from typing import List, Tuple
import numpy as np
from fl_utils import get_parameters, set_parameters
from trainer import train, test
import flwr as fl
from models.ist import IndependentSubNetworks

class FlowerISTClient(fl.client.Client):
    def __init__(self, cid, net, full_net, trainloader, valloader, device):
        self.cid = cid
        self.net = net #this node's subnetwork
        self.full_net = full_net #the full network
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device



    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # if i understand correctly this is called by the server 
        # when determining the initial parameters that must be trained
        # in our case, we return the full network's parameters
        print(f"Client {self.cid}: get_parameters")

        ndarrays: List[np.ndarray] = get_parameters(self.full_net)

        parameters = ndarrays_to_parameters(ndarrays)

        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters
        )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit, config: {ins.config}")

        # Deserialize parameters to NumPy ndarrays
        parameters_original = ins.parameters
        ndarray_original = parameters_to_ndarrays(parameters_original)

        # Update local model, train, get updated parameters
        set_parameters(self.net, ndarray_original)
        train(self.net, self.trainloader, epochs=1, device = self.device)
        ndarrays_updated = get_parameters(self.net)

        # Serialize ndarray's into a Parameters object
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={}
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarrays
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_parameters(self.full_net, ndarrays_original)
        loss, accuracy = test(self.full_net, self.valloader, device = self.device)
        print (f"Client {self.cid}: eval loss {loss}, accuracy {accuracy}")
        
        # return float loss , len of valloader, and accuracy

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)}
        )