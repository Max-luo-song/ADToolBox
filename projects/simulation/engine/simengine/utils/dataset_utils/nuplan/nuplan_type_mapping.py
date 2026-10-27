"""
Category mapping from nuplan track objects to world engine.
"""


import logging

from simengine.utils.type import SimEngineObjectType

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
NuPlanEgoType = TrackedObjectType.EGO


def get_traffic_obj_type(nuplan_type):
    if nuplan_type == TrackedObjectType.VEHICLE:
        return SimEngineObjectType.VEHICLE
    elif nuplan_type == TrackedObjectType.TRAFFIC_CONE:
        return SimEngineObjectType.TRAFFIC_CONE
    elif nuplan_type == TrackedObjectType.PEDESTRIAN:
        return SimEngineObjectType.PEDESTRIAN
    elif nuplan_type == TrackedObjectType.BICYCLE:
        return SimEngineObjectType.CYCLIST
    elif nuplan_type == TrackedObjectType.BARRIER:
        return SimEngineObjectType.TRAFFIC_BARRIER
    elif nuplan_type == TrackedObjectType.GENERIC_OBJECT:
        return SimEngineObjectType.TRAFFIC_OBJECT
    elif nuplan_type == TrackedObjectType.EGO:
        raise ValueError("Ego should not be in detected results")
    else:
        return None


def set_light_status(status):
    if status == TrafficLightStatusType.GREEN:
        return SimEngineObjectType.LIGHT_GREEN
    elif status == TrafficLightStatusType.RED:
        return SimEngineObjectType.LIGHT_RED
    elif status == TrafficLightStatusType.YELLOW:
        return SimEngineObjectType.LIGHT_YELLOW
    elif status == TrafficLightStatusType.UNKNOWN:
        return SimEngineObjectType.LIGHT_UNKNOWN
