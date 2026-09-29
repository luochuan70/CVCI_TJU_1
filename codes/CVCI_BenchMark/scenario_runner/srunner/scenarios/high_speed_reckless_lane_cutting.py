import math

import py_trees
import carla
import numpy as np

def get_speed(actor):
    vel = actor.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    WaypointFollower,
    LaneChange
)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToVehicle,
    InTriggerDistanceToLocation,
    InTriggerRegion,
    DriveDistance
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    Criterion, 
    WrongLaneTest, 
    CutInBrakeResponseCriterion, 
    CutInSafeBypassCriterion, 
    CutInResumeCriterion,
    MinTTCAutoCriterion
    )


def _read_param(config, name, default_value, cast_type=float):
    if name not in config.other_parameters:
        return default_value
    return cast_type(config.other_parameters[name].get('value', default_value))

def _get_trigger_location(config):
    if getattr(config, 'trigger_points', None) and config.trigger_points[0]:
        return config.trigger_points[0].location
    return None


class EgoSpeedControl(py_trees.behaviour.Behaviour):
    """
    主车速度保持节点（适合有坡度）
    - 未接管时：仅做纵向速度保持
    - 检测到人工输入后：永久放权并返回 SUCCESS
    - 不使用 set_target_velocity
    - 不修改 steer，避免覆盖人工转向/route 跟踪
    """
    def __init__(
        self,
        ego_vehicle,
        target_speed=10.0,
        throttle_gain=0.20,
        brake_gain=0.10,
        max_throttle=0.1,
        max_brake=0.50,
        takeover_steer_threshold=0.02,
        takeover_throttle_threshold=0.02,
        takeover_brake_threshold=0.02,
        name="EgoSpeedControl"
    ):
        super(EgoSpeedControl, self).__init__(name)
        self.ego_vehicle = ego_vehicle
        self.target_speed = target_speed

        self.throttle_gain = throttle_gain
        self.brake_gain = brake_gain
        self.max_throttle = max_throttle
        self.max_brake = max_brake

        self.takeover_steer_threshold = takeover_steer_threshold
        self.takeover_throttle_threshold = takeover_throttle_threshold
        self.takeover_brake_threshold = takeover_brake_threshold

        self._taken_over = False

    def _get_speed(self):
        v = self.ego_vehicle.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def update(self):
        if not self.ego_vehicle or not self.ego_vehicle.is_alive:
            return py_trees.common.Status.FAILURE

        if self._taken_over:
            return py_trees.common.Status.SUCCESS

        current_control = self.ego_vehicle.get_control()

        # 接管判定：方向/油门/刹车任一超过阈值，立即放权
        # 注意：这仍然不是最理想的“原始输入检测”，但比只看 throttle/brake 好很多
        if (
            abs(current_control.steer) > self.takeover_steer_threshold or
            current_control.throttle > self.takeover_throttle_threshold or
            current_control.brake > self.takeover_brake_threshold
        ):
            print("[EgoSpeedControl] Manual takeover detected, release control.")
            self._taken_over = True
            return py_trees.common.Status.SUCCESS

        current_speed = self._get_speed()
        speed_error = self.target_speed - current_speed

        new_control = carla.VehicleControl()

        # 不碰方向，避免覆盖人工/上层横向控制
        new_control.steer = current_control.steer
        new_control.hand_brake = False
        new_control.reverse = False
        new_control.manual_gear_shift = False

        if speed_error >= 0.0:
            # 速度偏低：补油
            throttle_cmd = np.clip(speed_error * self.throttle_gain, 0.0, self.max_throttle)
            new_control.throttle = float(throttle_cmd)
            new_control.brake = 0.0
        else:
            # 速度偏高：轻刹
            brake_cmd = np.clip((-speed_error) * self.brake_gain, 0.0, self.max_brake)
            new_control.throttle = 0.0
            new_control.brake = float(brake_cmd)

        self.ego_vehicle.apply_control(new_control)
        return py_trees.common.Status.RUNNING

class CutInCollision(BasicScenario):
    """
    场景：自车直行，右侧一辆车辆保持直行，另一辆车辆从右侧切入自车车道。
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=150):

        self.timeout = timeout
        self._trigger_location = None

        # 场景行为参数
        self._trigger_distance = _read_param(config, 'trigger_distance', 35.0)
        self._trigger_region_x = _read_param(config, 'trigger_region_x', 0.0)
        self._trigger_region_y = _read_param(config, 'trigger_region_y', 0.0)
        self._cutin_ready_distance = _read_param(config, 'cutin_ready_distance', 12.0)
        self._spawn_ahead_distance = _read_param(config, 'spawn_ahead_distance', 12.0)
        self._straight_speed = _read_param(config, 'straight_speed', 25.0)
        self._cutin_initial_speed = _read_param(config, 'cutin_initial_speed', 10.0)
        self._cutin_lane_change_speed = _read_param(config, 'cutin_lane_change_speed', 16.0)
        self._distance_same_lane = _read_param(config, 'distance_same_lane', 3.0)
        self._distance_other_lane = _read_param(config, 'distance_other_lane', 30.0)
        self._distance_lane_change = _read_param(config, 'distance_lane_change', 10.0)
        self._end_distance = _read_param(config, 'end_distance', 150.0)
        self._brake_trigger_distance = _read_param(config, 'brake_trigger_distance', 20.0)
        self._brake_threshold = _read_param(config, 'brake_threshold', 0.15)
        self._speed_drop_ratio = _read_param(config, 'speed_drop_ratio', 0.25)
        self._response_timeout = _read_param(config, 'response_timeout', 4.0)
        self._pass_distance = _read_param(config, 'pass_distance', 10.0)
        self._recovery_min_speed = _read_param(config, 'recovery_min_speed', 8.0)
        self._recovery_max_speed = _read_param(config, 'recovery_max_speed', 22.0)
        self._lane_center_tolerance = _read_param(config, 'lane_center_tolerance', 1.75)
        self._goal_distance_threshold = _read_param(config, 'goal_distance_threshold', 15.0)
        self._route_end_location = None

        if getattr(config, 'route', None):
            self._route_end_location = config.route[-1][0].location

        super(CutInCollision, self).__init__("CutInCollision",
                                                   ego_vehicles,
                                                   config,
                                                   world,
                                                   debug_mode=debug_mode,
                                                   criteria_enable=criteria_enable)
        if self.ego_vehicles:
            ego = self.ego_vehicles[0]
            yaw = math.radians(ego.get_transform().rotation.yaw)
            ego.set_target_velocity(carla.Vector3D(math.cos(yaw) * (self._straight_speed-5), math.sin(yaw) * (self._straight_speed-5)))

    def _create_space_trigger(self, ego, cutin_vehicle):
        return InTriggerDistanceToVehicle(
            cutin_vehicle,
            ego,
            distance=self._trigger_distance,
            name="Trigger_CutInStart"
        )

    def _initialize_actors(self, config):
        for actor_index, actor_conf in enumerate(config.other_actors):
            transform = actor_conf.transform
            if actor_index == 0:
                transform = self._spawn_transform_right_near_ego(transform)

            vehicle = CarlaDataProvider.request_new_actor(
                actor_conf.model,
                transform,
                rolename='scenario'
            )
            if vehicle is not None:
                self.other_actors.append(vehicle)
                vehicle.set_autopilot(False)
                if actor_index == 0:
                    self._set_initial_forward_speed(vehicle, transform, speed_mps=self._cutin_initial_speed)
            else:
                print(f"Spawn failed: {actor_conf.model} @ {transform.location}")



    def _spawn_transform_right_near_ego(self, transform):
        if not self.ego_vehicles:
            return transform

        ego_location = CarlaDataProvider.get_location(self.ego_vehicles[0])
        if ego_location is None:
            return transform

        world_map = CarlaDataProvider.get_map()
        reference_wp = world_map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if reference_wp is None:
            return transform

        right_wp = reference_wp.get_right_lane()
        while right_wp is not None and right_wp.lane_type != carla.LaneType.Driving:
            right_wp = right_wp.get_right_lane()

        if right_wp is None:
            return transform

        near_ahead = right_wp.next(self._spawn_ahead_distance)
        spawn_wp = near_ahead[0] if near_ahead else right_wp

        adjusted = carla.Transform(spawn_wp.transform.location, spawn_wp.transform.rotation)
        adjusted.location.z = max(transform.location.z, spawn_wp.transform.location.z + 0.2)
        return adjusted

    def _set_initial_forward_speed(self, vehicle, transform, speed_mps=6.0):
        forward = transform.get_forward_vector()
        vehicle.set_target_velocity(carla.Vector3D(
            x=forward.x * speed_mps,
            y=forward.y * speed_mps,
            z=0.0
        ))

    def _create_behavior(self):
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
            name="root"
        )
        root_1 = py_trees.composites.Sequence(name="CutInCollisionBehavior")

        if not self.ego_vehicles or len(self.other_actors) < 2:
            return root_1

        ego = self.ego_vehicles[0]
        cutin_vehicle = self.other_actors[0]
        straight_vehicle = self.other_actors[1]

        trigger = self._create_space_trigger(ego, cutin_vehicle)

        main_actions = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
            name="MainActions"
        )

        # 2. 直行车辆保持本车道行驶
        main_actions.add_child(WaypointFollower(
            straight_vehicle,
            target_speed=self._straight_speed,
            name="Straight_KeepLane"
        ))

        # 3. 切入车辆行为序列
        cutin_seq = py_trees.composites.Sequence(name="CutIn_Sequence")

        phase1 = py_trees.composites.Parallel(
            "CutIn_Phase1_Drive",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        phase1.add_child(WaypointFollower(
            cutin_vehicle,
            target_speed=self._cutin_initial_speed,
            name="CutIn_Phase1_Straight"
        ))
        phase1.add_child(InTriggerDistanceToVehicle(
            cutin_vehicle,
            ego,
            distance=self._cutin_ready_distance,
            name="CutIn_Ready_LaneChange"
        ))
        cutin_seq.add_child(phase1)

        phase2_cutin = LaneChange(
            cutin_vehicle,
            speed=self._cutin_lane_change_speed,
            direction='left',
            distance_same_lane=self._distance_same_lane,
            distance_other_lane=self._distance_other_lane,
            distance_lane_change=self._distance_lane_change,
            lane_changes=1,
            name="CutIn_Phase2_LaneChange"
        )
        cutin_seq.add_child(phase2_cutin)

        phase3_continue = WaypointFollower(
            cutin_vehicle,
            target_speed=self._cutin_lane_change_speed,
            name="CutIn_Phase3_Continue"
        )
        cutin_seq.add_child(phase3_continue)

        main_actions.add_child(cutin_seq)

        # 4. 自车完成预设行驶距离后结束场景
        end_cond = DriveDistance(
            ego,
            distance=self._end_distance,
            name="End_Distance"
        )
        main_actions.add_child(end_cond)

        root_1.add_child(trigger)
        root_1.add_child(main_actions)
        root.add_child(EgoSpeedControl(ego, target_speed=self._straight_speed))
        root.add_child(root_1)
        return root_1

    def _create_test_criteria(self):
        if not self.ego_vehicles or len(self.other_actors) < 2:
            return []

        ego = self.ego_vehicles[0]
        cutin_vehicle = self.other_actors[0]
        straight_vehicle = self.other_actors[1]

        return [
            CutInBrakeResponseCriterion(
                actor=ego,
                hazard_actor=cutin_vehicle,
                trigger_distance=self._brake_trigger_distance,
                brake_threshold=self._brake_threshold,
                speed_drop_ratio=self._speed_drop_ratio,
                max_response_time=self._response_timeout,
            ),
            CutInSafeBypassCriterion(
                actor=ego,
                hazard_actor=cutin_vehicle,
                pass_distance=self._pass_distance,
                min_speed=self._recovery_min_speed,
                max_speed=self._recovery_max_speed,
            )
        ]

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()