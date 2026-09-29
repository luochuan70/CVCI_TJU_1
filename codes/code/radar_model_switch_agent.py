"""DriveTransformer agent with first-radar-frame model and rule dispatch.

The shared model, perception, preprocessing, and base-control paths are built
during setup without a checkpoint.  The first live RADAR packet selects one
route family; that decision then selects both the checkpoint and the
route-family-specific auxiliary/rule implementation before model inference.
"""

from __future__ import absolute_import

import json
import os
import pathlib
import time
from collections import OrderedDict

import carla
import torch

from DriveTransformer.team_code.drivetransformer_b2d_agent import (
    DriveTransformerAgent as _0629DriveTransformerAgent,
)

try:
    from radar_scene_gate import RadarSceneGate
except ImportError:
    from .radar_scene_gate import RadarSceneGate

try:
    from scene_runtime_dispatch import activate as activate_runtime_code
except ImportError:
    from ..scene_runtime_dispatch import activate as activate_runtime_code


class RadarModelSwitchAgent(_0629DriveTransformerAgent):
    """Use the frozen 144 checkpoint or the official 220 checkpoint per gate."""

    def _build_scene_gate(self):
        return RadarSceneGate(
            route_xml=os.environ.get("CVCI_SCENE_GATE_ROUTE_XML", ""),
            output_path=os.environ.get("CVCI_SCENE_GATE_LOG_PATH", ""),
            reference_dir=os.environ.get("CVCI_SCENE_REFERENCE_DIR", ""),
        )

    def setup(self, path_to_conf_file):
        parts = path_to_conf_file.split("+")
        if len(parts) < 2:
            raise ValueError(
                "agent config must be CONFIG+PLACEHOLDER+route_save_name"
            )
        config_path = parts[0]
        save_name = parts[-1]

        # The 0629 parent builds the exact model/preprocessing stack.  Passing
        # the literal None placeholder leaves the model uninitialized until
        # the first radar packet has selected its checkpoint.
        super().setup(config_path + "+None+" + save_name)
        self.ckpt_path = None
        self._scene_gate = self._build_scene_gate()
        self._scene_gate_result = None
        self._model_loaded = False
        self._runtime_code_variant = None
        self._model_load_started_at = None
        self._model_load_finished_at = None
        self._diagnostic_radar_extractor = None
        self._diagnostic_radar_handle = None
        self._extreme_checkpoint = os.environ.get(
            "CVCI_EXTREME_CHECKPOINT", ""
        ).strip()
        self._ordinary_checkpoint = os.environ.get(
            "CVCI_ORDINARY_CHECKPOINT", ""
        ).strip()
        self._radar_sensor = {
            "type": "sensor.other.radar",
            "x": 1.50,
            "y": 0.0,
            "z": 1.30,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "horizontal_fov": 80.0,
            "vertical_fov": 20.0,
            "sensor_tick": 0.05,
            "id": "RADAR_FRONT",
        }
        self.all_sensors["RADAR_FRONT"] = dict(self._radar_sensor)
        print(
            "CVCI_RADAR_MODEL_SWITCH setup=deferred config={} "
            "extreme_checkpoint={} ordinary_checkpoint={}".format(
                config_path, self._extreme_checkpoint, self._ordinary_checkpoint
            ),
            flush=True,
        )

    def _activate_runtime_code(self):
        if self._runtime_code_variant is not None:
            return
        manifest = activate_runtime_code(
            self, self._scene_gate_result["decision"]
        )
        self._scene_gate_result["runtime_code_variant"] = manifest["name"]
        self._scene_gate_result["runtime_code_source_root"] = manifest[
            "source_root"
        ]
        self._scene_gate_result["runtime_code_files"] = manifest["files"]
        if "environment_profile" in manifest:
            self._scene_gate_result["runtime_environment_profile"] = dict(
                manifest["environment_profile"]
            )
        self._scene_gate.write(self._scene_gate_result)
        print(
            "CVCI_RUNTIME_CODE scene_set={} variant={} source_root={} files={}".format(
                self._scene_gate_result["decision"],
                manifest["name"],
                manifest["source_root"],
                ",".join(item["path"] for item in manifest["files"]),
            ),
            flush=True,
        )

    def sensors(self):
        sensors = list(super().sensors())
        if not any(sensor.get("id") == "RADAR_FRONT" for sensor in sensors):
            sensors.append(dict(self._radar_sensor))
        return sensors

    def tick(self, input_data):
        """Expose radar to the ordinary auxiliary adapter only.

        The frozen 144 branch receives the exact parent tick dictionary.  The
        extra key is added only after the first-layer gate has selected
        ordinary_220, so the new classifier cannot observe or alter 144.
        """
        tick_data = super().tick(input_data)
        if (
            self._scene_gate_result is not None
            and self._scene_gate_result.get("decision") == "ordinary_220"
        ):
            tick_data["radar_front"] = input_data.get("RADAR_FRONT")
            # The evaluator already supplies a 1 m interpolated global plan to
            # every leaderboard agent.  Expose a bounded live horizon to the
            # ordinary controller so it can follow the legal curve instead of
            # joining the sparse near/far endpoints with a chord.  No XML,
            # route identity, map lookup, or simulator actor is consulted.
            try:
                horizon = []
                cumulative = 0.0
                previous = None
                for item in list(self._route_planner.route)[:80]:
                    point = item[0]
                    xy = [float(point[0]), float(point[1])]
                    if previous is not None:
                        cumulative += float(
                            ((xy[0] - previous[0]) ** 2 + (xy[1] - previous[1]) ** 2)
                            ** 0.5
                        )
                    horizon.append(xy)
                    previous = xy
                    if cumulative >= 55.0:
                        break
                tick_data["route_horizon_xy"] = horizon
            except Exception:
                tick_data["route_horizon_xy"] = []
        return tick_data

    @staticmethod
    def _checkpoint_metadata(path, family):
        stat = os.stat(path)
        known_sha = os.environ.get(
            "CVCI_{}_CHECKPOINT_SHA256".format(
                "EXTREME" if family == "extreme_144" else "ORDINARY"
            ),
            "",
        ).strip()
        return {
            "family": family,
            "path": os.path.abspath(path),
            "size_bytes": int(stat.st_size),
            "mtime_epoch": float(stat.st_mtime),
            "sha256": known_sha,
        }

    def _load_selected_checkpoint(self):
        decision = self._scene_gate_result["decision"]
        if decision == "extreme_144":
            checkpoint = self._extreme_checkpoint
            expected_name = "iter_25000.pth"
        elif decision == "ordinary_220":
            checkpoint = self._ordinary_checkpoint
            expected_name = "drivetransformer_large.pth"
        else:
            raise RuntimeError("unknown scene-set decision: {}".format(decision))
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                "checkpoint for {} is missing: {}".format(decision, checkpoint)
            )
        if os.path.basename(os.path.normpath(checkpoint)) != expected_name:
            raise RuntimeError(
                "checkpoint for {} must be {}, got {}".format(
                    decision, expected_name, checkpoint
                )
            )

        self._model_load_started_at = time.time()
        print(
            "CVCI_MODEL_SWITCH loading scene_set={} checkpoint={}".format(
                decision, checkpoint
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        checkpoint_obj = torch.load(checkpoint, map_location="cpu")
        state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("model.", "").replace("._orig_mod", "")
            new_state_dict[new_key] = value
        load_result = self.model.load_state_dict(new_state_dict, strict=False)
        del new_state_dict
        del state_dict
        del checkpoint_obj
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.ckpt_path = checkpoint
        self._model_loaded = True
        self._model_load_finished_at = time.time()
        self._scene_gate_result["selected_model"] = self._checkpoint_metadata(
            checkpoint, decision
        )
        self._scene_gate_result["selected_model"][
            "required_filename"
        ] = expected_name
        self._scene_gate_result["model_load_seconds"] = round(
            self._model_load_finished_at - self._model_load_started_at, 3
        )
        self._scene_gate_result["model_trainable_parameter_count"] = int(
            sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
        )
        self._scene_gate_result["state_dict_load_missing_keys"] = list(
            getattr(load_result, "missing_keys", [])
        )
        self._scene_gate_result["state_dict_load_unexpected_keys"] = list(
            getattr(load_result, "unexpected_keys", [])
        )
        self._scene_gate.write(self._scene_gate_result)
        print(
            "CVCI_MODEL_SWITCH loaded scene_set={} checkpoint={} "
            "seconds={:.3f} trainable_params=0 missing_keys={} unexpected_keys={}".format(
                decision,
                checkpoint,
                self._scene_gate_result["model_load_seconds"],
                len(self._scene_gate_result["state_dict_load_missing_keys"]),
                len(self._scene_gate_result["state_dict_load_unexpected_keys"]),
            ),
            flush=True,
        )

    def run_step(self, input_data, timestamp):
        if not self._model_loaded:
            self._scene_gate_result = self._scene_gate.decide(
                input_data.get("RADAR_FRONT")
            )
            self._scene_gate_result["route_id"] = os.environ.get(
                "CVCI_ROUTE_ID", ""
            )
            self._scene_gate_result["agent_first_run_timestamp"] = float(timestamp)
            self._scene_gate.write(self._scene_gate_result)
            print(
                "CVCI_RADAR_GATE route={} sensor_frame={} decision={} "
                "source={} radar_candidate={} radar_score={}".format(
                    self._scene_gate_result["route_id"],
                    self._scene_gate_result["first_radar"].get("sensor_frame"),
                    self._scene_gate_result["decision"],
                    self._scene_gate_result["decision_source"],
                    self._scene_gate_result["radar_candidate"],
                    self._scene_gate_result["radar_extreme_score"],
                ),
                flush=True,
            )
            self._activate_runtime_code()
            self._load_selected_checkpoint()
            if (
                self._scene_gate_result.get("decision") == "ordinary_220"
                and os.environ.get("CVCI_RADAR_RECORD_INDEPENDENT", "0") == "1"
            ):
                # This recorder is opt-in and is opened only after the first
                # radar gate has selected ordinary_220.  It permits a formal
                # ordinary-disabled baseline to provide radar training data
                # without importing or executing the ordinary rule router.
                try:
                    from harness.radar_features import RadarFeatureExtractor, vector_feature_names

                    self._diagnostic_radar_extractor = RadarFeatureExtractor()
                    record_path = os.environ.get("CVCI_RADAR_RECORD_PATH", "").strip()
                    if record_path:
                        parent = os.path.dirname(os.path.abspath(record_path))
                        if parent:
                            os.makedirs(parent, exist_ok=True)
                        self._diagnostic_radar_handle = open(record_path, "a", encoding="utf-8")
                    self._diagnostic_feature_names = vector_feature_names()
                except Exception as exc:
                    self._diagnostic_radar_extractor = None
                    self._diagnostic_radar_handle = None
                    print("CVCI_RADAR_RECORD independent_open_error={}".format(repr(exc)), flush=True)
        control = super().run_step(input_data, timestamp)
        self._record_independent_radar(input_data, timestamp, control)
        return control

    @staticmethod
    def _independent_speed(input_data):
        try:
            speed_packet = input_data.get("SPEED")
            if isinstance(speed_packet, (tuple, list)) and len(speed_packet) == 2:
                speed_packet = speed_packet[1]
            if isinstance(speed_packet, dict):
                return float(speed_packet.get("speed", 0.0))
        except (TypeError, ValueError):
            pass
        return 0.0

    @staticmethod
    def _serializable_radar(packet):
        if packet is None:
            return None
        if isinstance(packet, (tuple, list)) and len(packet) == 2:
            frame, data = packet
        else:
            frame, data = None, packet
        try:
            values = data.tolist()
        except Exception:
            values = []
        return [frame, values]

    def _record_independent_radar(self, input_data, timestamp, control):
        if self._diagnostic_radar_extractor is None or self._diagnostic_radar_handle is None:
            return
        if self._scene_gate_result is None or self._scene_gate_result.get("decision") != "ordinary_220":
            return
        try:
            speed = self._independent_speed(input_data)
            packet = input_data.get("RADAR_FRONT")
            features = self._diagnostic_radar_extractor.extract(
                packet,
                ego_speed=speed,
                steer=getattr(control, "steer", 0.0),
                brake=getattr(control, "brake", 0.0),
                timestamp=timestamp,
            )
            row = {
                "timestamp": timestamp,
                "radar": self._serializable_radar(packet),
                "ego_speed": speed,
                "throttle": getattr(control, "throttle", 0.0),
                "steer": getattr(control, "steer", 0.0),
                "brake": getattr(control, "brake", 0.0),
                "feature_vector": features.get("vector", []),
                "feature_names": getattr(self, "_diagnostic_feature_names", []),
                "predicted_class_id": 0,
                "locked_class_id": 0,
                "confidence": 0.0,
                "candidate_confidence": 0.0,
                "classifier_reason": "diagnostic_raw_radar",
                "transition": None,
                "action_debug": {
                    "source": "ordinary_disabled_baseline",
                    "state": "RAW_RADAR_RECORD_ONLY",
                },
            }
            self._diagnostic_radar_handle.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            self._diagnostic_radar_handle.flush()
        except Exception:
            # Recording must never change the formal control path.
            return

    def destroy(self):
        if (
            getattr(self, "_scene_gate_result", None) is not None
            and getattr(self, "_model_loaded", False)
        ):
            self._scene_gate_result["destroy_time_epoch"] = time.time()
            self._scene_gate.write(self._scene_gate_result)
        router = getattr(self, "_ordinary_44_router", None)
        if router is not None:
            router.close()
        handle = getattr(self, "_diagnostic_radar_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
            self._diagnostic_radar_handle = None
        return super().destroy()


def get_entry_point():
    return "RadarModelSwitchAgent"
