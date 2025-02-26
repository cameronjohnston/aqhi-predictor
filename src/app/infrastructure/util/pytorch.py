""" pytorch-related utilities specific to this implementation """

import torch


def to_tensor(wildfires, wind_velocities, aqhi_readings):
    """
    Convert domain entities into tensors while incorporating spatial and temporal features.
    """

    # Convert wildfires to tensor
    wildfire_tensors = torch.tensor(
        [[wf.latitude, wf.longitude, wf.asofdate.timestamp(), wf.frp]
            for wf in wildfires], dtype=torch.float32
    ) if wildfires else torch.empty(0, 4)

    # Convert wind velocities to tensor
    wind_tensors = torch.tensor(
        [[wv.latitude, wv.longitude, wv.observed_datetime.timestamp(), wv.x_component, wv.y_component]
            for wv in wind_velocities], dtype=torch.float32
    ) if wind_velocities else torch.empty(0, 5)

    # Convert AQHI readings to tensor
    aqhi_tensors = torch.tensor(
        [[aq.latitude, aq.longitude, aq.observed_datetime.timestamp(), aq.value]
            for aq in aqhi_readings], dtype=torch.float32
    ) if aqhi_readings else torch.empty(0, 4)

    return wildfire_tensors, wind_tensors, aqhi_tensors

