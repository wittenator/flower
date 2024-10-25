"""scaffold: A Flower Baseline."""

from functools import reduce
from logging import WARNING, log
from typing import Optional, Union
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitIns, FitRes, Parameters
from flwr.proto.transport_pb2 import Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np

from scaffold.utils import marshal_numpy, unmarshal_numpy

class SCAFFOLD(FedAvg):

    def __init__(self, initial_parameters: Parameters, global_lr: float = 0.1, **kwargs):
        super().__init__(initial_parameters=initial_parameters, **kwargs)
        self.global_lr = global_lr
        self.last_parameters = initial_parameters
        self.global_control = np.zeros(len(parameters_to_ndarrays(initial_parameters)))

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        log(WARNING, f"Client raised exception: {failures}")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        c_delta_results, y_delta_results = zip(*[(fit_res.c_delta, fit_res.y_delta) for _, fit_res in results])
        c_delta_results = [ unmarshal_numpy(c_delta) for c_delta in c_delta_results ]
        y_delta_results = [ unmarshal_numpy(y_delta) for y_delta in y_delta_results ]

        # Aggregate weights with control
        num_clients = len(results)

        # Compute average weights of each layer
        aggregated_ndarrays = [
            global_weight + (reduce(np.add, layer_updates) / num_clients) * self.global_lr
            for *layer_updates, global_weight in zip(*y_delta_results, parameters_to_ndarrays(self.last_parameters))
        ]

        aggregated_control = [reduce(np.add, c_delta) / num_clients for c_delta in zip(*c_delta_results)]

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.last_parameters = parameters_aggregated
        self.global_control = aggregated_control

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Sends the proximal factor mu to the clients
        """
        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "global_control": marshal_numpy(np.array(self.global_control))},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]