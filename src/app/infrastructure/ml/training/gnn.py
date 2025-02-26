""" Training using global neural networks (GNN) """

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import logging
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric_temporal.nn.recurrent import GConvGRU, A3TGCN
from torch_geometric_temporal.signal import temporal_signal_split
from typing import List, Tuple

from application.ml.trainers import ModelTrainer
from domain.entities import Wildfire, WindVelocity, AQHI
from infrastructure.ml.training.dataset_loaders import AQHIDatasetLoader


class GNNModel(torch.nn.Module):
    """Simple Graph Neural Network using PyTorch Geometric"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

@dataclass
class PyGModelTrainer(ModelTrainer):
    """ PyTorch Geometric model trainer """
    wildfires: List[Wildfire]
    wind_velocities: List[WindVelocity]
    aqhi_data: List[AQHI]
    time_intervals: List[int]  # List of time intervals in hours (e.g., [6, 12, 24, 48])

    def train(self) -> Tuple[torch.nn.Module, dict]:
        """Train a PyTorch Geometric model"""

        # Step 1: Prepare training data
        self.prepare_training_data()
        data: Data = self.get_training_data()

        # Step 1.1: Extract AQHI node indices
        self.aqhi_indices = torch.tensor([self.node_map[(aq.latitude, aq.longitude, (aq.observed_datetime + timedelta(hours=t)).timestamp())]
                                          for aq in self.aqhi_data for t in self.time_intervals], dtype=torch.long)

        # Step 2: Define model, optimizer, and loss function
        input_dim = data.x.shape[1]  # Number of input features
        hidden_dim = 64
        output_dim = 1  # AQHI prediction target

        model = GNNModel(input_dim, hidden_dim, output_dim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer = Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)  # Reduce LR every step_size epochs
        loss_fn = torch.nn.MSELoss()  # Mean Squared Error for regression

        # Step 3: Training loop
        model.train()
        num_epochs = 100
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)  # Forward pass

            # Ensure `out` only includes AQHI nodes
            out_aqhi = out[self.aqhi_indices]  # Filter predictions to AQHI nodes
            loss = loss_fn(out_aqhi, data.y)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            if epoch % 5 == 0:
                logging.info(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.4f}")

        # Step 4: Return trained model and training metrics
        return model, {"final_loss": loss.item()}

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


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features,
                           out_channels=3,
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(3, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        logging.info(f"Before: x shape {x.shape}, edge_index shape {edge_index.shape}")

        # if x.dim() == 2:  # Add time dimension if missing
        #     x = x.unsqueeze(0)

        # # Ensure edge_index is valid
        # valid_mask = edge_index[0] < x.shape[1]
        # edge_index = edge_index[:, valid_mask]
        # valid_mask = edge_index[1] < x.shape[1]
        # edge_index = edge_index[:, valid_mask]

        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        logging.info(f"After: h shape {h.shape}")

        return h

class SimpleGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_timesteps_out):
        super(SimpleGNN, self).__init__()
        self.recurrent = GConvGRU(num_features, hidden_dim, K=2)
        self.linear = nn.Linear(hidden_dim, num_timesteps_out)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, torch.cat(edge_index, dim=-1), torch.cat(edge_weight, dim=-1))
        return self.linear(h)


@dataclass
class PyGTModelTrainer(ModelTrainer):
    """ PyTorch Geometric Temporal model trainer """
    wildfires: List[Wildfire]
    wind_velocities: List[WindVelocity]
    aqhi_data: List[AQHI]
    time_intervals: List[int] = field(default_factory=list)  # List of time intervals in hours (e.g., [6, 12, 24, 48])
    # TODO: assess how appropriate time_intervals list is

    def __post_init__(self):
        aqhi_times = [aq.observed_datetime for aq in self.aqhi_data]
        self.wildfires = [wf for wf in self.wildfires if wf.observed_datetime in aqhi_times]
        self.wind_velocities = [wv for wv in self.wind_velocities if wv.observed_datetime in aqhi_times]

    def train(self) -> None:
        """ Train the model """
        loader = AQHIDatasetLoader(self.wildfires, self.wind_velocities, self.aqhi_data)
        dataset = loader.get_dataset()
        logging.info(f'{self.cn} got dataset.')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        subset = 2000 # TODO: does this make sense here?
        # model = SimpleGNN(num_features=5, hidden_dim=16, num_timesteps_out=12).to(device)
        model = TemporalGNN(node_features=len(dataset.features[0]), periods=len(dataset.features)).to(device)
        optimizer = Adam(model.parameters(), lr=0.01)

        # Train-Test Split
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        train_dataset = dataset

        logging.info(f'{self.cn} starting 10 epochs...')
        for epoch in range(10):
            loss = 0
            step = 0
            for snapshot in train_dataset:
                snapshot = snapshot.to(device)
                # Get model predictions
                y_hat = model(snapshot.x, snapshot.edge_index)
                # Mean squared error
                loss = loss + torch.mean((y_hat-snapshot.y)**2)
                step += 1
                if step > subset:
                  break

            loss = loss / (step + 1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logging.info("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))

    def train_v1(self) -> None:
        """ Train the model """
        loader = AQHIDatasetLoader(self.wildfires, self.wind_velocities, self.aqhi_data)
        dataset = loader.get_dataset()
        logging.info(f'{self.cn} got dataset.')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = SimpleGNN(num_features=5, hidden_dim=16, num_timesteps_out=12).to(device)
        model = TemporalGNN(node_features=3, periods=1).to(device)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Convert lists to tensors
        longest_feature_len = max([len(f) for f in dataset.features])
        features = torch.stack([torch.tensor(f, dtype=torch.float32) for f in dataset.features], dim=0).to(device)
        edge_index = [torch.tensor(i, dtype=torch.long).to(device) for i in dataset.edge_indices]
        edge_weight = [torch.tensor(w, dtype=torch.float32).to(device) for w in dataset.edge_weights]

        # Check dimensions
        num_timesteps = features.shape[-1]  # Ensure last dimension is time
        train_size = int(num_timesteps * 0.8)
        test_size = num_timesteps - train_size

        # Train-Test Split
        train_features = features[..., :train_size]  # Keep all nodes and feature dims, but limit time
        test_features = features[..., train_size:]

        train_targets = train_features[:, 4, -12:]  # Last 12 timesteps AQHI
        test_targets = test_features[:, 4, -12:]

        train_losses = []
        test_losses = []

        logging.info(f'{self.cn} starting 100 epochs...')
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            train_output = model(train_features, edge_index, edge_weight)
            train_loss = criterion(train_output[:, 4, :], train_targets)
            train_loss.backward()
            optimizer.step()

            # Evaluate on test data
            model.eval()
            with torch.no_grad():
                test_output = model(test_features, edge_index, edge_weight)
                test_loss = criterion(test_output[:, 4, :], test_targets)

            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())

            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

        # Final Evaluation & Plotting
        model.eval()
        train_predicted_aqhi = model(train_features, edge_index, edge_weight).detach().cpu().numpy()
        test_predicted_aqhi = model(test_features, edge_index, edge_weight).detach().cpu().numpy()

        train_actual_aqhi = train_targets.cpu().numpy()
        test_actual_aqhi = test_targets.cpu().numpy()

        # Plot predictions vs actual
        plt.figure(figsize=(12, 5))
        for i, (actual, predicted, title) in enumerate([
            (train_actual_aqhi[0], train_predicted_aqhi[0], "Train Set"),
            (test_actual_aqhi[0], test_predicted_aqhi[0], "Test Set")
        ]):
            plt.subplot(1, 2, i + 1)
            plt.plot(range(12), actual, label="Actual AQHI", marker="o")
            plt.plot(range(12), predicted, label="Predicted AQHI", marker="x")
            plt.xlabel("Time Step")
            plt.ylabel("AQHI")
            plt.legend()
            plt.title(f"{title}: AQHI Prediction vs Actual")

        plt.show()

        # Plot loss curves
        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training vs Test Loss")
        plt.show()

    def prepare_training_data(self):
        """ Prepare training data """
        pass

    def get_training_data(self):
        """ Return processed training data in a format agnostic to ML frameworks """
        pass


