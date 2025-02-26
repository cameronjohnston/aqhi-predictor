
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import math
import torch
from torch_geometric.data import Data
from typing import List, Tuple

from application.ml.predictors import AQHIPredictor
from domain.entities import Wildfire, WindVelocity, AQHI, ModelTrainingResult
from infrastructure.ml.training.gnn import GNNModel


@dataclass
class PyGModelPredictor(AQHIPredictor):
    """PyTorch Geometric model predictor for AQHI"""
    wildfires: List[Wildfire]
    wind_velocities: List[WindVelocity]
    aqhi_data: List[AQHI]
    time_intervals: List[int]
    model_training_result: ModelTrainingResult

    def predict(self) -> List[AQHI]:
        """Run inference on new data"""

        # Step 1: Prepare data (same as training)
        self.prepare_training_data()

        # Step 2: Load trained model
        model = GNNModel(input_dim=self.data.x.shape[1], hidden_dim=64, output_dim=1)
        checkpoint = self.model_training_result.model_data
        logging.info(f"model_data has the following keys: {checkpoint.keys()}")
        model.load_state_dict(checkpoint)
        # Above should be equivalent to model.load_state_dict(torch.load(self.model_path))
        model.eval()  # Set to evaluation mode

        # Step 3: Run inference
        with torch.no_grad():
            predictions = model(self.data.x, self.data.edge_index)

        # Step 3.1: Extract AQHI node indices
        self.aqhi_indices = torch.tensor([self.node_map[(aq.latitude, aq.longitude, (aq.observed_datetime + timedelta(hours=t)).timestamp())]
                                          for aq in self.aqhi_data for t in self.time_intervals], dtype=torch.long)

        # Step 4: Extract AQHI node predictions
        aqhi_predictions = predictions[self.aqhi_indices].flatten().tolist()

        # Step 5: Map predictions back to AQHI node locations/timestamps
        predicted_aqhi = []
        for (aq, t), pred_value in zip([(aq, t) for aq in self.aqhi_data for t in self.time_intervals], aqhi_predictions):
            ts = aq.observed_datetime + timedelta(hours=t)
            predicted_aqhi.append((aq.latitude, aq.longitude, ts, pred_value))

        logging.info(f"Produced {len(predicted_aqhi)} AQHI predictions.")
        aqhi = [
            AQHI(
                latitude=p[0],
                longitude=p[1],
                observed_datetime=p[2],
                value=p[3],
                source=self.model_training_result.model_id,
                forecast_datetime=datetime.now(),
            ) for p in predicted_aqhi
        ]
        return aqhi

    def prepare_training_data(self):
        """ Prepare training data """
        logging.info(f"{self.cn} building nodes...")
        self._build_nodes()
        logging.info(f"{self.cn} building edges...")
        self._build_edges()
        logging.info(f"{self.cn} creating PyG data object...")
        self._create_pyg_data()
        logging.info(f"{self.cn} done creating PyG data object.")

    def get_training_data(self):
        """ Return processed training data in a format agnostic to ML frameworks """
        return self.data  # TODO_EH: Error if training data isn't yet prepared?

    def _build_nodes(self):
        """ Convert domain entities into PyTorch tensors, duplicating them for each time step. """
        self.node_list = []
        self.aqhi_node_list = []
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
                self.aqhi_node_list.append([aq.value])
                self.node_map[(aq.latitude, aq.longitude, ts.timestamp())] = index
                index += 1

        # Combine into a single tensor
        self.combined_tensor = torch.tensor(self.node_list, dtype=torch.float)

        # Also add AQHI tensor
        self.aqhi_tensor = torch.tensor(self.aqhi_node_list, dtype=torch.float)

        # Check sample of nodes
        logging.info(f"Node map sample: {list(self.node_map.items())[:5]}")

    def _build_edges(self):
        """ Create edges between wildfire, wind, and AQHI nodes, including Wind -> Wind for multi-step transport. """
        edge_list = []
        edge_attributes = []

        # Wildfire -> Wind (smoke injected into atmosphere)
        logging.info("Processing Wildfire -> Wind edges...")
        for i, wf in enumerate(self.wildfires):
            if i % 100 == 0:
                logging.info(f"  Processed {i}/{len(self.wildfires)} wildfires")
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

        logging.info("Finished processing Wildfire -> Wind edges.")
        logging.info(f"Total edges: {len(edge_list)}")
        logging.info(f"Estimated memory usage: {len(edge_list) * 16 / (1024 ** 3):.2f} GB")  # Est 16 bytes per edge

        # Wind -> Wind (multi-step smoke transport)
        logging.info("Processing Wind -> Wind edges...")
        for i, wv1 in enumerate(self.wind_velocities):
            if i % 100 == 0:
                logging.info(f"  Processed {i}/{len(self.wind_velocities)} wind velocities")
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

        logging.info("Finished processing Wind -> Wind edges.")
        logging.info(f"Total edges: {len(edge_list)}")
        logging.info(f"Estimated memory usage: {len(edge_list) * 16 / (1024 ** 3):.2f} GB")  # Est 16 bytes per edge

        # Wind -> AQHI (final transport stage)
        logging.info("Processing Wind -> AQHI edges...")
        for i, wv in enumerate(self.wind_velocities):
            if i % 100 == 0:
                logging.info(f"  Processed {i}/{len(self.wind_velocities)} wind velocities")
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

        logging.info("Finished processing all edges.")
        logging.info(f"Total edges: {len(edge_list)}")
        logging.info(f"Estimated memory usage: {len(edge_list) * 16 / (1024 ** 3):.2f} GB")  # Est 16 bytes per edge

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    def _create_pyg_data(self):
        """ Create the PyG Data object """
        self.data = Data(x=self.combined_tensor, edge_index=self.edge_index, edge_attr=self.edge_attr,
                         y=self.aqhi_tensor.view(-1, 1),  # Ensure correct shape
                         )

    @staticmethod
    def _wind_can_transport(wind1, wind2) -> bool:
        # TODO: migrate to wind velocity service?
        """
        Determines if wind1 at (latitude1, longitude1, time1) can transport smoke to wind2 (latitude2, longitude2, time2).
        """

        time_diff = (wind2.observed_datetime - wind1.observed_datetime).total_seconds() / 3600  # In hours
        if time_diff <= 0:
            return False  # Wind cannot go backward in time
        if time_diff > 24:
            return False  # Disregard observations > 24h apart, since there will be other observations closer together

        dx = wind2.latitude - wind1.latitude
        dy = wind2.longitude - wind1.longitude
        distance_sq = dx ** 2 + dy ** 2
        distance = math.sqrt(distance_sq) + 1e-6

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


class PyGModelPredictor_v2_possibly(AQHIPredictor):
    """PyTorch Geometric model predictor for AQHI"""

    def __init__(self, model_training_result: ModelTrainingResult):
        self.model_training_result = model_training_result
        self.model = self.load_model(model_training_result.model_data)

    def predict(self, input_data: Data) -> torch.Tensor:
        """Run inference on new data"""
        self.model.eval()
        with torch.no_grad():
            return self.model(input_data.x, input_data.edge_index)

    def load_model(self, model_data: bytes):
        """Load model from serialized bytes"""
        model = torch.load(torch.io.BytesIO(model_data))
        return model

    def save_model(self) -> bytes:
        """Serialize model to bytes"""
        buffer = torch.io.BytesIO()
        torch.save(self.model, buffer)
        return buffer.getvalue()


