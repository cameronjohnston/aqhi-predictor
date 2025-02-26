
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

from domain.entities import Wildfire, WindVelocity, AQHI
from domain.util.distance import haversine_distance
from geopy import distance
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, DynamicGraphTemporalSignal
from scipy.spatial import KDTree


class AQHIDatasetLoader:
    def __init__(self, wildfires: List[Wildfire], wind_velocities: List[WindVelocity], aqhi_readings: List[AQHI]):
        self.wildfires = wildfires
        self.wind_velocities = wind_velocities
        self.aqhi_readings = aqhi_readings

    def compute_edge_weight(self, source, target):
        """Computes the edge weight between source and target based on domain-specific logic."""
        distance = haversine_distance(source.latitude, source.longitude, target.latitude, target.longitude)

        if isinstance(source, Wildfire):
            if distance > 1000:  # Ignore if too far
                return 0
            return source.frp / (distance + 1e-5)  # Higher FRP & closer distance = stronger impact

        if isinstance(source, WindVelocity):
            time_diff = (target.observed_datetime - source.observed_datetime).total_seconds() / 3600  # in hours

            if time_diff <= 0 or time_diff > 24:
                return 0  # Must be within the next 24h

            # Compute wind transport feasibility
            x_disp = source.x_component * time_diff
            y_disp = source.y_component * time_diff
            est_lat, est_lon = source.latitude + y_disp / 111, source.longitude + x_disp / (
                        111 * np.cos(np.radians(source.latitude)))
            est_distance = haversine_distance(est_lat, est_lon, target.latitude, target.longitude)

            if est_distance > 50:  # If wind couldn't realistically transport smoke here
                return 0

            return 1 / (est_distance + 1e-5)  # Closer transport gets higher weight

        return 0

    def _node_key(self, node):
        return (type(node).__name__, node.latitude, node.longitude, node.observed_datetime)

    def _feature_key(self, node):
        return (type(node).__name__, node.latitude, node.longitude)

    def get_dataset(self) -> DynamicGraphTemporalSignal:
        """Builds and returns the dataset as a DynamicGraphTemporalSignal object."""
        edge_indices = []
        edge_weights = []
        features = []
        targets = []

        nodes = self.wildfires + self.wind_velocities + self.aqhi_readings
        node_index_map = {self._node_key(node): i for i, node in enumerate(nodes)}

        timestamps = sorted(set(n.observed_datetime for n in nodes))
        # unique_features = sorted(set(self._feature_key(n) for n in nodes))
        unique_features = sorted(set(self._feature_key(n) for n in nodes))
        unique_features_index_map = {f: i for i, f in enumerate(unique_features)}

        for t in timestamps:
            snapshot_edges = []  # np.zeros((len(unique_features_index_map), len(unique_features_index_map), 2))  # []
            snapshot_weights = []  # np.zeros((len(unique_features_index_map), len(unique_features_index_map)))  # []
            snapshot_features = np.zeros((len(unique_features_index_map), 3))  # []
            snapshot_targets = np.zeros((len(unique_features_index_map), 1))  # []

            wildfires_t = [wf for wf in self.wildfires if wf.observed_datetime == t]
            winds_t = [wv for wv in self.wind_velocities if wv.observed_datetime == t]
            aqhi_t = [aqhi for aqhi in self.aqhi_readings if aqhi.observed_datetime == t]

            nodes_t = wildfires_t + winds_t + aqhi_t

            # Create edges: Wildfire -> Wind
            for wf in wildfires_t:
                for wv in winds_t:
                    weight = self.compute_edge_weight(wf, wv)
                    if weight > 0:
                        snapshot_edges.append([node_index_map[self._node_key(wf)], node_index_map[self._node_key(wv)]])
                        snapshot_weights.append(weight)

            # Create edges: Wind -> Wind (transport over time)
            for wv1 in self.wind_velocities:
                for wv2 in self.wind_velocities:
                    if wv1.observed_datetime < wv2.observed_datetime <= wv1.observed_datetime + timedelta(hours=24):
                        weight = self.compute_edge_weight(wv1, wv2)
                        if weight > 0:
                            snapshot_edges.append([
                                node_index_map[self._node_key(wv1)], node_index_map[self._node_key(wv2)]
                            ])
                            snapshot_weights.append(weight)

            # Create edges: Wind -> AQHI (future impact)
            for wv in winds_t:
                for aqhi in self.aqhi_readings:
                    if 0 < (aqhi.observed_datetime - wv.observed_datetime).total_seconds() / 3600 <= 24:
                        weight = self.compute_edge_weight(wv, aqhi)
                        if weight > 0:
                            snapshot_edges.append([node_index_map[self._node_key(wv)], node_index_map[self._node_key(aqhi)]])
                            snapshot_weights.append(weight)

            # Node features and targets
            for node in nodes_t:
                # Features shall have 3 elements: FRP, wind x component, wind y component
                # Targets shall have just one element: AQHI
                # Both features and targets will have already been initialized with zeros (start of timestamp loop);
                # Therefore below we only need to explicitly populate the non-zero values
                if isinstance(node, Wildfire):
                    snapshot_features[unique_features_index_map[self._feature_key(node)]] = [node.frp, 0, 0]
                elif isinstance(node, WindVelocity):
                    snapshot_features[unique_features_index_map[self._feature_key(node)]] = [0, node.x_component, node.y_component]
                elif isinstance(node, AQHI):
                    snapshot_targets[unique_features_index_map[self._feature_key(node)]] = [node.value]

            # edge_indices.extend(np.array(snapshot_edges).T if snapshot_edges else np.empty((2, 0)))
            edge_indices.append(np.array(snapshot_edges) if snapshot_edges else np.empty(0))
            edge_weights.append(np.array(snapshot_weights) if snapshot_weights else np.empty(0))

            # Ensure snapshot_features have the same length
            # max_len = max(len(f) if isinstance(f, list) else 1 for f in snapshot_features)
            # snapshot_features = [f if isinstance(f, list) and len(f) == max_len else f + [0] * (max_len - len(f)) for f
            #                      in snapshot_features]

            features.append(np.array(snapshot_features))
            targets.append(np.array(snapshot_targets))

        # Pad indices and weights - must have the same shape
        max_indices = max([len(i) for i in edge_indices])
        max_weights = max([len(i) for i in edge_weights])
        for i, e in enumerate(edge_indices):
            edge_indices[i] = np.pad(e, ((0, max_indices - len(e)), (0, 0)), mode='constant', constant_values=0)
        for i, e in enumerate(edge_weights):
            edge_weights[i] = np.pad(e, ((0, max_weights - len(e))), mode='constant', constant_values=0)
        return DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)


class AQHIDatasetLoader_v1:
    def __init__(self, wildfires, wind_velocities, aqhi_readings):
        """ Params are the lists of domain entities to be used as training data """
        self.wildfires = wildfires
        self.wind_velocities = wind_velocities
        self.aqhi_readings = aqhi_readings

        self.node_mapping = self._generate_node_mapping()
        self.kd_tree = KDTree(list(self.node_mapping.keys()))
        self.aqhi_timestamps = sorted(set(aqhi.observed_datetime for aqhi in self.aqhi_readings))

    def _generate_node_mapping(self) -> Dict[Tuple[float, float], int]:
        """ Map unique (lat, lon) pairs to node indices """
        unique_locations = set(
            (e.latitude, e.longitude) for e in self.wildfires + self.wind_velocities + self.aqhi_readings)
        return {loc: idx for idx, loc in enumerate(sorted(unique_locations))}

    def _generate_adjacency_matrix(self):
        """ Create adjacency matrix modeling wind-driven smoke transport """
        num_nodes = len(self.node_mapping)
        num_timesteps = len(self.aqhi_timestamps)
        adjacency_matrix = np.zeros((num_nodes * num_timesteps, num_nodes * num_timesteps))

        for wind in self.wind_velocities:
            wind_node = self.node_mapping.get((wind.latitude, wind.longitude))
            if wind_node is None:
                continue
            wind_vector = np.array([wind.x_component, wind.y_component])

            for target_loc, target_idx in self.node_mapping.items():
                if target_idx == wind_node:
                    continue
                target_vector = np.array([target_loc[0] - wind.latitude, target_loc[1] - wind.longitude])
                if np.dot(wind_vector, target_vector) > 0:  # Ensure wind is blowing toward target
                    distance = np.linalg.norm(target_vector)
                    transport_time = int(distance / (wind.speed + 1e-6))  # Time delay estimate
                    for t in range(num_timesteps - transport_time):
                        src_idx = t * num_nodes + wind_node
                        tgt_idx = (t + transport_time) * num_nodes + target_idx
                        adjacency_matrix[src_idx, tgt_idx] = wind.speed / (distance + 1e-6)

        edge_index, edge_weight = dense_to_sparse(torch.tensor(adjacency_matrix, dtype=torch.float))
        return edge_index, edge_weight

    def _edge_weight(self, source, target):
        """ Estimate the ability for wind to carry smoke from source at t=0 to target at t=1 """
        # ASSUMPTION: Target is 24h after source

        if isinstance(source, Wildfire):
            dist = distance.distance((source.latitude, source.longitude), (target.latitude, target.longitude))
            return 0 if dist.km >= 1000 else (1000 - dist.km) / 1000.0

        # if isinstance(source, WindVelocity):


    def _generate_feature_matrix(self):
        """ Generate time-aware feature matrix tracking wildfire impact """
        num_nodes = len(self.node_mapping)
        num_timesteps = len(self.aqhi_timestamps)
        num_features = 5  # [FRP, Wind Speed, Wind X, Wind Y, AQHI]
        feature_matrix = np.zeros((num_timesteps, num_nodes, num_features))
        smoke_concentration = np.zeros((num_nodes, num_timesteps))

        for fire in self.wildfires:
            fire_time_idx = self.aqhi_timestamps.index(fire.observed_datetime)
            fire_node = self.node_mapping.get((fire.latitude, fire.longitude))
            for t in range(fire_time_idx, num_timesteps):
                smoke_concentration[fire_node, t] += fire.frp  # Accumulate smoke impact

        for wind in self.wind_velocities:
            wind_time_idx = self.aqhi_timestamps.index(wind.observed_datetime)
            wind_node = self.node_mapping.get((wind.latitude, wind.longitude))
            if wind_node is not None:
                feature_matrix[wind_time_idx, wind_node, 1] = wind.speed
                feature_matrix[wind_time_idx, wind_node, 2] = wind.x_component
                feature_matrix[wind_time_idx, wind_node, 3] = wind.y_component

        for aqhi in self.aqhi_readings:
            aqhi_time_idx = self.aqhi_timestamps.index(aqhi.observed_datetime)
            aqhi_node = self.node_mapping.get((aqhi.latitude, aqhi.longitude))
            if aqhi_node is not None:
                feature_matrix[aqhi_time_idx, aqhi_node, 4] = aqhi.value
                feature_matrix[aqhi_time_idx, aqhi_node, 0] = smoke_concentration[aqhi_node, aqhi_time_idx]

        return torch.tensor(feature_matrix, dtype=torch.float).permute(1, 2, 0)  # (nodes, features, timesteps)

    def get_dataset(self):
        logging.info(f'{self.cn} generating adjacency matrix...')
        edge_index, edge_weight = self._generate_adjacency_matrix()
        logging.info(f'{self.cn} generating feature matrix...')
        features = self._generate_feature_matrix()
        targets = [None for _ in range(features.shape[2])]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure edge_index and edge_weight are tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long) if isinstance(edge_index, list) else edge_index
        edge_weight = torch.tensor(edge_weight, dtype=torch.float) if isinstance(edge_weight, list) else edge_weight
        return StaticGraphTemporalSignal(
            edge_index=edge_index,
            edge_weight=edge_weight,
            features=features.permute(2, 0, 1),  # Ensure correct shape (time, nodes, features)
            targets=targets
        )

    @property
    def cn(self):
        return type(self).__name__


