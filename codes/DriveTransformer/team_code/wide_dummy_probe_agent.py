"""Ordinary-route sensor-suite probe.

This diagnostic agent deliberately keeps the frozen 0629 camera geometry and
sensor sizes, but removes the model, auxiliary perception, and controller
code.  It is used only to decide whether CARLA crashes while creating or
delivering the six-camera sensor suite.
"""
from __future__ import print_function

import os

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


def get_entry_point():
    return "WideDummyProbeAgent"


class WideDummyProbeAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS

    def sensors(self):
        low_resolution = os.environ.get("CVCI_WIDE_DUMMY_LOWRES", "0") == "1"
        include_speedometer = os.environ.get("CVCI_WIDE_DUMMY_SPEEDOMETER", "1") == "1"
        camera_count = int(os.environ.get("CVCI_WIDE_DUMMY_CAMERA_COUNT", "6"))
        if camera_count < 1 or camera_count > 6:
            raise ValueError("CVCI_WIDE_DUMMY_CAMERA_COUNT must be in [1, 6]")
        selected_names = os.environ.get("CVCI_WIDE_DUMMY_CAMERA_NAMES", "").strip()
        selected_names = set(filter(None, selected_names.split(","))) if selected_names else None
        width, height = (800, 450) if low_resolution else (1600, 900)
        sensors = [
            {
                "type": "sensor.camera.rgb",
                "x": 0.80,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": width,
                "height": height,
                "fov": 70,
                "id": "CAM_FRONT",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 0.27,
                "y": -0.55,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -55.0,
                "width": width,
                "height": height,
                "fov": 70,
                "id": "CAM_FRONT_LEFT",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 0.27,
                "y": 0.55,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 55.0,
                "width": width,
                "height": height,
                "fov": 70,
                "id": "CAM_FRONT_RIGHT",
            },
            {
                "type": "sensor.camera.rgb",
                "x": -2.0,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 180.0,
                "width": width,
                "height": height,
                "fov": 110,
                "id": "CAM_BACK",
            },
            {
                "type": "sensor.camera.rgb",
                "x": -0.32,
                "y": -0.55,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -110.0,
                "width": width,
                "height": height,
                "fov": 70,
                "id": "CAM_BACK_LEFT",
            },
            {
                "type": "sensor.camera.rgb",
                "x": -0.32,
                "y": 0.55,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 110.0,
                "width": width,
                "height": height,
                "fov": 70,
                "id": "CAM_BACK_RIGHT",
            },
            {
                "type": "sensor.other.imu",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "IMU",
            },
            {
                "type": "sensor.other.gnss",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "GPS",
            },
        ]
        camera_sensors = sensors[:6]
        if selected_names is not None:
            camera_sensors = [sensor for sensor in camera_sensors if sensor["id"] in selected_names]
        else:
            camera_sensors = camera_sensors[:camera_count]
        sensors = camera_sensors + sensors[6:]
        if include_speedometer:
            sensors.append({"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"})
        return sensors

    def run_step(self, input_data, timestamp):
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        control.hand_brake = False
        return control
