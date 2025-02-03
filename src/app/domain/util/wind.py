""" Wind-related conversions / calculations """

from typing import Tuple
import math


def meteorological2math(degrees_cw_from_northerly: float) -> float:
    """
    Convert from meteorological convention (degrees clockwise from North)
    to mathematical convention (radians counterclockwise from West).
    """
    return math.radians((270 - degrees_cw_from_northerly) % 360)


def math2meteorological(radians_ccw_from_westerly: float) -> float:
    """
    Convert from mathematical convention (radians counterclockwise from West)
    to meteorological convention (degrees clockwise from North).
    """
    return (270 - math.degrees(radians_ccw_from_westerly)) % 360


def x_y_components(velocity) -> Tuple[float, float]:
    """ Provide X and Y components of velocity as a tuple (x, y) """
    radians_ccw_from_westerly = meteorological2math(velocity.source_direction)
    x_component = velocity.speed * math.cos(radians_ccw_from_westerly)  # East-West component
    y_component = velocity.speed * math.sin(radians_ccw_from_westerly)  # North-South component
    return x_component, y_component


