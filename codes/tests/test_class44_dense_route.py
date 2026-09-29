from __future__ import absolute_import

import os
import sys
import unittest
from unittest import mock

CODES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODES_ROOT not in sys.path:
    sys.path.insert(0, CODES_ROOT)

from ordinary_rules.class_44 import Rule44
from ordinary_rules.configs import get_config
from ordinary_rules.local_planner import (
    OrdinaryLocalPathController,
    route_horizon_observation,
    route_waypoint_observation,
)
from ordinary_rules.router import OrdinaryRuleRouter


class DenseRouteClass44Tests(unittest.TestCase):
    def test_class44_uses_online_turn_side_to_reject_cross_street_red(self):
        rule = Rule44(get_config(44))

        class Control(object):
            throttle = 0.0
            steer = 0.0
            brake = 0.0
            reverse = False

        class Features(object):
            ego_speed = 0.0
            red_light_active = True
            traffic_control_kind = ""
            traffic_light_region_votes = {
                "left": {"color": "red", "confidence": 1.0},
                "right": {"color": "green", "confidence": 0.86},
            }

        radar = {
            "_timestamp": 1.0,
            "_classifier_flow_pass_seen": True,
            "_route_waypoint_observation": {
                "far": {"heading_error_rad": -0.72},
            },
        }
        _, first = rule.apply(Control(), radar, Features())
        self.assertTrue(first["signal_hold"])
        self.assertEqual(first["signal_owner"]["owned_side"], "right")
        radar["_timestamp"] = 1.1
        _, second = rule.apply(Control(), radar, Features())
        self.assertFalse(second["signal_hold"])
        self.assertTrue(second["signal_owner"]["owned_green_latched"])

    def test_class44_does_not_release_when_owned_turn_head_is_red(self):
        rule = Rule44(get_config(44))

        class Control(object):
            throttle = 0.0
            steer = 0.0
            brake = 0.0
            reverse = False

        class Features(object):
            ego_speed = 0.0
            red_light_active = True
            traffic_control_kind = ""
            traffic_light_region_votes = {
                "left": {"color": "green", "confidence": 1.0},
                "right": {"color": "red", "confidence": 1.0},
            }

        radar = {
            "_timestamp": 1.0,
            "_classifier_flow_pass_seen": True,
            "_route_waypoint_observation": {
                "far": {"heading_error_rad": -0.72},
            },
        }
        for index in range(4):
            radar["_timestamp"] = 1.0 + 0.1 * index
            _, debug = rule.apply(Control(), radar, Features())
        self.assertTrue(debug["signal_hold"])
        self.assertFalse(debug["signal_owner"]["owned_green_latched"])

    def test_dense_route_pid_uses_online_horizon_lookahead(self):
        calls = []

        def pid(**kwargs):
            calls.append(kwargs)
            return 0.36

        controller = OrdinaryLocalPathController(route_pid_steer=pid)
        tick = {
            "pos": [0.0, 0.0],
            "compass": 0.0,
            "speed": 4.0,
            "command_near_xy": [0.0, -4.0],
            "command_far_xy": [12.0, -18.0],
            "route_horizon_xy": [
                [0.0, 0.0], [0.0, -2.0], [0.0, -4.0], [0.5, -6.0],
                [1.5, -8.0], [3.0, -10.0], [5.0, -11.5], [7.0, -12.5],
            ],
        }
        observation = route_horizon_observation(tick)
        self.assertIsNotNone(observation)
        self.assertGreater(observation["available_distance_m"], 10.0)
        feedback = controller.dense_route_pid_step(
            {
                "phase": "class44_dense_route_follow",
                "lookahead_base_m": 5.0,
                "lookahead_speed_s": 0.45,
                "steer_limit": 0.48,
                "steer_rate_per_frame": 0.10,
            },
            tick,
            fallback_steer=0.0,
        )
        self.assertEqual(calls[0]["phase"], "class44_dense_route_follow")
        self.assertNotEqual(
            calls[0]["tick_data"]["command_near_xy"],
            tick["command_near_xy"],
        )
        self.assertEqual(feedback["source"], "carla_public_pid_dense_online_route")
        self.assertAlmostEqual(feedback["steer"], 0.10)

    def test_dense_route_pid_applies_opt_in_signed_left_offset(self):
        calls = []

        def pid(**kwargs):
            calls.append(kwargs)
            return 0.0

        controller = OrdinaryLocalPathController(route_pid_steer=pid)
        tick = {
            "pos": [0.0, 0.0],
            "compass": 3.141592653589793,
            "speed": 4.0,
            "command_near_xy": [0.0, 2.0],
            "command_far_xy": [0.0, 12.0],
            "route_horizon_xy": [
                [0.0, 0.0], [0.0, 2.0], [0.0, 4.0],
                [0.0, 6.0], [0.0, 8.0], [0.0, 10.0],
            ],
        }
        feedback = controller.dense_route_pid_step(
            {
                "phase": "class16_dense_merge_clearance",
                "target_route_offset_m": 2.5,
                "lookahead_base_m": 5.0,
                "lookahead_speed_s": 0.0,
            },
            tick,
            timestamp=1.0,
        )
        self.assertAlmostEqual(
            calls[0]["tick_data"]["command_near_xy"][0], -2.5
        )
        self.assertAlmostEqual(feedback["target_route_offset_m"], 2.5)

    def test_dense_route_pose_filter_rejects_gnss_jump_without_motion_lag(self):
        calls = []

        def pid(**kwargs):
            calls.append(kwargs)
            return 0.0

        controller = OrdinaryLocalPathController(route_pid_steer=pid)
        directive = {
            "phase": "class44_dense_route_follow",
            "pose_filter_frames": 3,
        }
        base = {
            "compass": 3.141592653589793,
            "speed": 1.0,
            "command_near_xy": [0.0, 2.0],
            "command_far_xy": [0.0, 12.0],
            "route_horizon_xy": [
                [0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0],
            ],
        }
        for timestamp, noisy_x, y in (
            (1.0, 0.0, 0.0),
            (1.1, 2.0, 0.1),
            (1.2, 0.0, 0.2),
        ):
            tick = dict(base, pos=[noisy_x, y])
            feedback = controller.dense_route_pid_step(
                directive, tick, timestamp=timestamp
            )
        self.assertEqual(feedback["pose_filter_sample_count"], 3)
        self.assertAlmostEqual(feedback["raw_position_xy"][0], 0.0)
        self.assertAlmostEqual(feedback["filtered_position_xy"][0], 0.0)
        self.assertAlmostEqual(feedback["filtered_position_xy"][1], 0.2)
        self.assertAlmostEqual(calls[-1]["tick_data"]["pos"][0], 0.0)
        self.assertAlmostEqual(calls[-1]["tick_data"]["pos"][1], 0.2)

    def test_class44_stages_dense_route_after_sparse_clearance(self):
        rule = Rule44(get_config(44))
        self.assertEqual(rule.config["flow_next_safe_m"], 12.0)
        self.assertEqual(rule.config["flow_next_min_ttc_s"], 2.8)
        rule.state = Rule44.COMMITTED_ENTRY

        class Control(object):
            throttle = 0.0
            steer = 0.0
            brake = 0.0
            reverse = False

        class Features(object):
            ego_speed = 0.0
            red_light_active = False
            traffic_control_kind = ""

        _, debug = rule.apply(Control(), {"_timestamp": 1.0}, Features())
        self.assertNotIn("ordinary_dense_route_pid", debug)
        self.assertIn("ordinary_route_waypoint", debug)

        rule.committed_distance_m = 15.0
        for index in range(3):
            _, debug = rule.apply(Control(), {
                "_timestamp": 1.1 + index * 0.1,
                "_route_lateral_m": 0.5,
            }, Features())
        self.assertTrue(debug["dense_handoff"]["latched"])
        self.assertTrue(debug["ordinary_dense_route_pid"]["active"])

    def test_class44_advances_to_second_stage_then_follows_actor_tail(self):
        rule = Rule44(get_config(44))
        rule.state = Rule44.COMMITTED_ENTRY

        class Control(object):
            throttle = 0.0
            steer = 0.0
            brake = 0.0
            reverse = False

        class Features(object):
            ego_speed = 7.2
            red_light_active = False
            traffic_control_kind = ""

        near = {"count": 1, "distance_m": 3.0, "ttc_s": 0.4,
                "closing_speed_mps": 7.5}
        clear = {"count": 1, "distance_m": 24.0, "ttc_s": 3.0,
                 "closing_speed_mps": 8.0}
        with mock.patch.object(rule, "_flow_observation", return_value=near):
            first, first_debug = rule.apply(
                Control(), {"_timestamp": 1.0}, Features()
            )
        self.assertTrue(first_debug["emergency_flow"])
        self.assertEqual(first_debug["state"], "CLASS44_STAGED_BRAKE")
        self.assertGreaterEqual(first.brake, 0.85)

        Features.ego_speed = 0.0
        with mock.patch.object(rule, "_flow_observation", return_value=clear):
            rule.apply(Control(), {"_timestamp": 1.1}, Features())
            advancing, advancing_debug = rule.apply(
                Control(), {"_timestamp": 1.2}, Features()
            )
        self.assertEqual(advancing_debug["state"], "CLASS44_STAGED_ADVANCE")
        self.assertAlmostEqual(advancing.throttle, 0.45)
        self.assertEqual(advancing.brake, 0.0)

        rule.staged_advance_distance_m = 2.8
        rule.staged_advance_braking_latched = True
        with mock.patch.object(rule, "_flow_observation", return_value=clear):
            final_wait, final_wait_debug = rule.apply(
                Control(), {"_timestamp": 1.3}, Features()
            )
        self.assertEqual(final_wait_debug["state"], "CLASS44_FINAL_WAIT")
        self.assertGreaterEqual(final_wait.brake, 0.85)

        with mock.patch.object(rule, "_flow_observation", side_effect=[
            near,
            {"count": 0, "distance_m": 0.0, "ttc_s": 0.0,
             "closing_speed_mps": 0.0},
            clear,
        ]):
            rule.apply(Control(), {"_timestamp": 1.4}, Features())
            rule.apply(Control(), {"_timestamp": 1.5}, Features())
            second_advance, second_advance_debug = rule.apply(
                Control(), {"_timestamp": 1.6}, Features()
            )
        self.assertEqual(
            second_advance_debug["state"], "CLASS44_STAGED_ADVANCE"
        )
        self.assertFalse(second_advance_debug["staged_release_latched"])
        self.assertAlmostEqual(second_advance.throttle, 0.45)

        rule.staged_advance_distance_m = 1.4
        rule.staged_advance_braking_latched = True
        with mock.patch.object(rule, "_flow_observation", return_value=clear):
            _, second_wait_debug = rule.apply(
                Control(), {"_timestamp": 1.7}, Features()
            )
        self.assertEqual(second_wait_debug["state"], "CLASS44_FINAL_WAIT")
        self.assertGreaterEqual(second_wait_debug["staged_advance_total_m"], 4.0)

        with mock.patch.object(rule, "_flow_observation", side_effect=[
            near,
            {"count": 0, "distance_m": 0.0, "ttc_s": 0.0,
             "closing_speed_mps": 0.0},
            clear,
        ]):
            rule.apply(Control(), {"_timestamp": 1.8}, Features())
            rule.apply(Control(), {"_timestamp": 1.9}, Features())
            merged, merged_debug = rule.apply(
                Control(), {"_timestamp": 2.0}, Features()
            )
        self.assertTrue(merged_debug["final_pass_seen"])
        self.assertEqual(merged_debug["state"], "CLASS44_MERGE_ENTRY")
        self.assertTrue(merged_debug["staged_release_latched"])
        self.assertEqual(merged.brake, 0.0)
        self.assertGreaterEqual(merged.throttle, 0.75)

    def test_class44_radar_silence_does_not_release_final_wait(self):
        rule = Rule44(get_config(44))
        rule.state = Rule44.FINAL_WAIT
        rule.final_pass_seen = True

        class Control(object):
            throttle = 0.2
            steer = 0.0
            brake = 0.0
            reverse = False

        class Features(object):
            ego_speed = 0.0
            red_light_active = False
            traffic_control_kind = ""

        for index in range(10):
            output, debug = rule.apply(
                Control(), {"_timestamp": 2.0 + index * 0.1}, Features()
            )
        self.assertEqual(debug["state"], "CLASS44_FINAL_WAIT")
        self.assertFalse(debug["staged_release_candidate"])
        self.assertEqual(output.throttle, 0.0)
        self.assertGreaterEqual(output.brake, 0.85)

    def test_class44_staged_advance_stops_within_bounded_distance(self):
        rule = Rule44(get_config(44))
        rule.state = Rule44.STAGED_ADVANCE

        class Control(object):
            throttle = 0.0
            steer = 0.0
            brake = 0.0
            reverse = False

        class Features(object):
            ego_speed = 0.0
            red_light_active = False
            traffic_control_kind = ""

        far = {"count": 2, "distance_m": 20.0, "ttc_s": 2.5,
               "closing_speed_mps": 8.0}
        speed = 0.0
        for index in range(80):
            Features.ego_speed = speed
            with mock.patch.object(rule, "_flow_observation", return_value=far):
                output, debug = rule.apply(
                    Control(), {"_timestamp": 3.0 + index * 0.1}, Features()
                )
            acceleration = 4.0 * output.throttle - 6.0 * output.brake
            speed = max(0.0, speed + acceleration * 0.1)
            if debug["state"] == "CLASS44_FINAL_WAIT":
                break
        self.assertEqual(debug["state"], "CLASS44_FINAL_WAIT")
        self.assertGreaterEqual(debug["staged_advance_distance_m"], 2.0)
        self.assertLessEqual(debug["staged_advance_distance_m"], 3.5)

        hazard_rule = Rule44(get_config(44))
        hazard_rule.state = Rule44.STAGED_ADVANCE
        Features.ego_speed = 1.5
        near = {"count": 2, "distance_m": 3.5, "ttc_s": 0.5,
                "closing_speed_mps": 7.0}
        with mock.patch.object(hazard_rule, "_flow_observation", return_value=near):
            output, debug = hazard_rule.apply(
                Control(), {"_timestamp": 4.0}, Features()
            )
        self.assertTrue(debug["staged_advance_braking_latched"])
        self.assertEqual(output.throttle, 0.0)
        self.assertGreaterEqual(output.brake, 0.85)

    def test_router_uses_only_dense_output_when_horizon_is_available(self):
        router = OrdinaryRuleRouter(route_pid_steer=lambda **kwargs: 0.0)

        class Control(object):
            throttle = 0.2
            steer = 0.1
            brake = 0.0
            reverse = False

        class Features(object):
            ego_speed = 4.0
            detector_objects = []
            tracked_objects = []

        router.classifier.observe = mock.Mock(return_value={
            "class_id": 44,
            "candidate_class_id": 44,
            "confidence": 1.0,
            "candidate_confidence": 1.0,
            "reason": "test",
            "transition": None,
            "features": {},
            "feature_vector": [],
            "feature_names": [],
            "evidence": {},
        })
        router.rules[44].apply = mock.Mock(return_value=(Control(), {
            "ordinary_route_waypoint": {"active": True},
            "ordinary_dense_route_pid": {"active": True},
        }))
        router.local_path_controller.route_waypoint_step = mock.Mock()
        router.local_path_controller.dense_route_pid_step = mock.Mock(
            return_value={"steer": 0.25, "source": "dense-test"}
        )
        tick = {
            "radar_front": None,
            "pos": [0.0, 0.0],
            "compass": 0.0,
            "speed": 4.0,
            "command_near_xy": [0.0, -4.0],
            "command_far_xy": [12.0, -18.0],
            "route_horizon_xy": [
                [0.0, 0.0], [0.0, -2.0], [0.0, -4.0], [1.0, -6.0]
            ],
        }
        output, debug = router.process(Control(), Features(), tick, 1.0)
        routed_radar = router.rules[44].apply.call_args[0][1]
        self.assertEqual(
            routed_radar["_route_waypoint_observation"],
            route_waypoint_observation(tick),
        )
        router.local_path_controller.route_waypoint_step.assert_not_called()
        router.local_path_controller.dense_route_pid_step.assert_called_once()
        self.assertAlmostEqual(output.steer, 0.25)
        self.assertEqual(debug["action"]["override_source"], "dense-test")
        router.close()


if __name__ == "__main__":
    unittest.main()
