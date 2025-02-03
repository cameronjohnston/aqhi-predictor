""" Training using global neural networks (GNN) """

from dataclasses import dataclass
from datetime import datetime, time, timedelta
import math
import torch
from torch_geometric.data import Data
from typing import List

from application.ml.trainers import ModelTrainer
from domain.models import Wildfire, WindVelocity, AQHI


@dataclass
class PyGModelTrainer(ModelTrainer):
    """ PyTorch Geometric model trainer """
    wildfires: List[Wildfire]
    wind_velocities: List[WindVelocity]
    aqhi_data: List[AQHI]
    time_intervals: List[int]  # List of time intervals in hours (e.g., [6, 12, 24, 48])

    def train(self) -> None:
        """ Train the model """
        pass  # TODO: implement

    def prepare_training_data(self):
        """ Prepare training data """
        print(f"{self.cn} building nodes...")
        self._build_nodes()
        print(f"{self.cn} building edges...")
        self._build_edges()
        print(f"{self.cn} creating PyG data object...")
        self._create_pyg_data()
        print(f"{self.cn} done creating PyG data object.")

    def get_training_data(self):
        """ Return processed training data in a format agnostic to ML frameworks """
        return self.data  # TODO_EH: Error if training data isn't yet prepared?

    def _build_nodes(self):
        """ Convert domain entities into PyTorch tensors, duplicating them for each time step. """
        self.node_list = []
        self.node_map = {}  # Maps (latitude, longitude, time) to node index

        index = 0  # This shall assign a unique value to each node in the node_map
        for wf in self.wildfires:
            for t in self.time_intervals:
                # Assume the observation is from noon... TODO: better place to have this assumption, ideally?
                ts = datetime.combine(wf.asofdate, datetime.min.time()) + timedelta(hours=12+t)
                self.node_list.append([wf.latitude, wf.longitude, ts.timestamp(), wf.frp, 0.0])
                self.node_map[(wf.latitude, wf.longitude, ts.timestamp())] = index
                index += 1

        for wv in self.wind_velocities:
            for t in self.time_intervals:
                ts = wv.observed_datetime + timedelta(hours=t)
                self.node_list.append([wv.latitude, wv.longitude, ts.timestamp(), wv.x_component, wv.y_component])
                self.node_map[(wv.latitude, wv.longitude, ts.timestamp())] = index
                index += 1

        for aq in self.aqhi_data:
            for t in self.time_intervals:
                ts = aq.observed_datetime + timedelta(hours=t)
                self.node_list.append([aq.latitude, aq.longitude, ts.timestamp(), aq.value, 0.0])
                self.node_map[(aq.latitude, aq.longitude, ts.timestamp())] = index
                index += 1

        # Combine into a single tensor
        self.combined_tensor = torch.tensor(self.node_list, dtype=torch.float)

    def _build_edges(self):
        """ Create edges between wildfire, wind, and AQHI nodes, including Wind → Wind for multi-step transport. """
        edge_list = []
        edge_attributes = []

        # Wildfire → Wind (smoke injected into atmosphere)
        for wf in self.wildfires:
            wf_time = datetime.combine(wf.asofdate, datetime.min.time()).timestamp()
            wf_key = (wf.latitude, wf.longitude, wf_time)
            for wv in self.wind_velocities:
                for t in self.time_intervals:
                    wv_time = (wv.observed_datetime + timedelta(hours=t)).timestamp()
                    wv_key = (wv.latitude, wv.longitude, wv_time)

                    if wf_key in self.node_map and wv_key in self.node_map:
                        edge_list.append([self.node_map[wf_key], self.node_map[wv_key]])
                        influence = self._calculate_wind_influence(wf, wv)
                        edge_attributes.append([wv.x_component, wv.y_component, influence])

        # Wind → Wind (multi-step smoke transport)
        for wv1 in self.wind_velocities:
            for t1 in self.time_intervals:
                wv1_time = (wv1.observed_datetime + timedelta(hours=t1)).timestamp()
                wv1_key = (wv1.latitude, wv1.longitude, wv1_time)

                for wv2 in self.wind_velocities:
                    for t2 in self.time_intervals:
                        wv2_time = (wv2.observed_datetime + timedelta(hours=t2)).timestamp()
                        wv2_key = (wv2.latitude, wv2.longitude, wv2_time)

                        if wv1_key in self.node_map and wv2_key in self.node_map:
                            # Check if wind at wv1 can realistically transport smoke to wv2
                            if self._wind_can_transport(wv1, wv2):
                                edge_list.append([self.node_map[wv1_key], self.node_map[wv2_key]])
                                influence = self._calculate_wind_influence(wv1, wv2)
                                edge_attributes.append([wv1.x_component, wv1.y_component, influence])

        # Wind → AQHI (final transport stage)
        for wv in self.wind_velocities:
            for t in self.time_intervals:
                wv_time = (wv.observed_datetime + timedelta(hours=t)).timestamp()
                wv_key = (wv.latitude, wv.longitude, wv_time)
                for aq in self.aqhi_data:
                    aq_time = (aq.observed_datetime + timedelta(hours=t)).timestamp()
                    aq_key = (aq.latitude, aq.longitude, aq_time)

                    if wv_key in self.node_map and aq_key in self.node_map:
                        edge_list.append([self.node_map[wv_key], self.node_map[aq_key]])
                        influence = self._calculate_wind_influence(wv, aq)
                        edge_attributes.append([wv.x_component, wv.y_component, influence])

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    def _create_pyg_data(self):
        """ Create the PyG Data object """
        self.data = Data(x=self.combined_tensor, edge_index=self.edge_index, edge_attr=self.edge_attr)

    @staticmethod
    def _wind_can_transport(wind1, wind2) -> bool:
        # TODO: migrate to wind velocity service?
        """
        Determines if wind1 at (latitude1, longitude1, time1) can transport smoke to wind2 (latitude2, longitude2, time2).
        """
        dx = wind2.latitude - wind1.latitude
        dy = wind2.longitude - wind1.longitude
        distance = math.sqrt(dx ** 2 + dy ** 2)

        time_diff = (wind2.observed_datetime - wind1.observed_datetime).total_seconds() / 3600  # In hours
        if time_diff <= 0:
            return False  # Wind cannot go backward in time

        expected_travel_distance = wind1.speed * time_diff  # Distance wind would move in given time
        return distance <= expected_travel_distance  # True if wind can reach next point in given time

    @staticmethod
    def _calculate_wind_influence(source, target):
        # TODO: migrate to wind velocity service?
        """
        Compute directional influence of wind based on vector alignment, similar to before.
        """
        dx = target.latitude - source.latitude
        dy = target.longitude - source.longitude
        distance = math.sqrt(dx ** 2 + dy ** 2) + 1e-6  # Avoid division by zero

        wind_vector_x = getattr(source, "x_component", 0.0)
        wind_vector_y = getattr(source, "y_component", 0.0)
        wind_magnitude = math.sqrt(wind_vector_x ** 2 + wind_vector_y ** 2) + 1e-6

        # Compute alignment
        direction_x = dx / distance
        direction_y = dy / distance
        wind_x_norm = wind_vector_x / wind_magnitude
        wind_y_norm = wind_vector_y / wind_magnitude
        dot_product = (direction_x * wind_x_norm) + (direction_y * wind_y_norm)

        # Influence factor
        influence = max(0, dot_product) / distance
        return influence

