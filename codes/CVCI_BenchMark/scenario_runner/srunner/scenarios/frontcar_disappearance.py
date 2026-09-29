import py_trees
import carla

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToVehicle,
    DriveDistance,
)
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    WaypointFollower,
    LaneChange,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    StaticObstacleBrakeSlowDownCriterion,
    StaticObstacleSafePassCriterion,
    ReachEndPointCriterion,
    MinTTCAutoCriterion
)


class PassiveEgoSpeedHold(py_trees.behaviour.Behaviour):
    """保持自车初始速度，直到自动驾驶算法开始接管"""

    def __init__(self, actor, target_speed, name="PassiveEgoSpeedHold"):
        super().__init__(name)
        self.actor = actor
        self.target_speed = target_speed
        self._manual_override_detected = False

    def update(self):
        if not self.actor or not self.actor.is_alive:
            return py_trees.common.Status.RUNNING

        control = self.actor.get_control()
        passive_input = (
            abs(control.steer) < 0.05
            and control.brake < 0.05
            and control.throttle < 0.05
        )

        if not passive_input:
            self._manual_override_detected = True

        if not self._manual_override_detected and passive_input:
            transform = self.actor.get_transform()
            forward = transform.get_forward_vector()
            self.actor.set_target_velocity(
                carla.Vector3D(
                    forward.x * self.target_speed,
                    forward.y * self.target_speed,
                    forward.z * self.target_speed,
                )
            )

        return py_trees.common.Status.RUNNING


class CarDisappearDiagonalAccident(BasicScenario):
    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        randomize=False,
        debug_mode=False,
        criteria_enable=True,
        timeout=120,
    ):
        self.timeout = timeout
        self.ego_vehicle = ego_vehicles[0]
        self.lead_vehicle = None
        self.accident_vehicle = None
        self.background_vehicles = []

        # ========= 可调参数 =========
        self.global_y = 41.85
        self.ego_initial_speed = self._get_config_value(config, "init_speed", 19.4)
        self.lead_vehicle_speed = self.ego_initial_speed
        self.background_vehicle_speed = self.ego_initial_speed

        # phase1 触发：当前车距离障碍车足够近时，前车开始右变道
        self.phase1_trigger_distance = self._get_config_value(config, "trigger_distance", 50.0)

        # phase2 右变道参数
        self._distance_same_lane = 5.0   # 变道前还要前行多远
        self._distance_other_lane = 20.0 # 进入新车道行驶的距离
        self._distance_lane_change = 20.0 # 变道过程的纵向距离

        # phase3 变道后继续前进的距离
        self.phase3_drive_distance = 200.0

        super().__init__(
            "CarDisappearDiagonalAccident",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

    @staticmethod
    def _get_config_value(config, key, default_value):
        """兼容 ScenarioRunner 中常见的 config.other_parameters 读取方式"""
        if not hasattr(config, "other_parameters") or config.other_parameters is None:
            return default_value

        param = config.other_parameters.get(key)
        if param is None:
            return default_value

        if isinstance(param, dict):
            raw_value = param.get("value", default_value)
        else:
            raw_value = getattr(param, "value", default_value)

        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default_value

    @staticmethod
    def _set_actor_forward_speed(actor, speed):
        if actor is None or not actor.is_alive:
            return
        forward = actor.get_transform().get_forward_vector()
        actor.set_target_velocity(
            carla.Vector3D(
                forward.x * speed,
                forward.y * speed,
                forward.z * speed,
            )
        )

    def _initialize_actors(self, config):
        # =========================
        # 0: 障碍车（静止）
        # =========================
        accident_actor_conf = config.other_actors[0]
        self.accident_vehicle = CarlaDataProvider.request_new_actor(
            accident_actor_conf.model,
            accident_actor_conf.transform,
        )
        if self.accident_vehicle:
            light_state = (
                carla.VehicleLightState.Position
                | carla.VehicleLightState.LowBeam
                | carla.VehicleLightState.Special1
            )
            self.accident_vehicle.set_light_state(carla.VehicleLightState(light_state))
            self.accident_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.other_actors.append(self.accident_vehicle)

        # =========================
        # 1: 前车 lead_vehicle
        # =========================
        lead_actor_conf = config.other_actors[1]
        self.lead_vehicle = CarlaDataProvider.request_new_actor(
            lead_actor_conf.model,
            lead_actor_conf.transform
        )
        if self.lead_vehicle:
            light_state = (
                carla.VehicleLightState.Position
                | carla.VehicleLightState.LowBeam
                | carla.VehicleLightState.Special1
            )
            self.lead_vehicle.set_light_state(carla.VehicleLightState(light_state))
            self._set_actor_forward_speed(self.lead_vehicle, self.lead_vehicle_speed)
            self.other_actors.append(self.lead_vehicle)

        # =========================
        # 2~N: 新增背景车
        # =========================
        for bg_actor_conf in config.other_actors[2:]:
            bg_vehicle = CarlaDataProvider.request_new_actor(
                bg_actor_conf.model,
                bg_actor_conf.transform
            )
            if bg_vehicle:
                light_state = (
                    carla.VehicleLightState.Position
                    | carla.VehicleLightState.LowBeam
                )
                bg_vehicle.set_light_state(carla.VehicleLightState(light_state))
                self._set_actor_forward_speed(bg_vehicle, self.background_vehicle_speed)
                self.background_vehicles.append(bg_vehicle)
                self.other_actors.append(bg_vehicle)

        # =========================
        # 自车初速度
        # =========================
        ego = self.ego_vehicles[0]
        self._set_actor_forward_speed(ego, self.ego_initial_speed)

    def _create_behavior(self):
        root = py_trees.composites.Parallel(
            name="RootBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )

        # 持续保持自车初速度，直到外部控制接管
        ego_speed_hold = PassiveEgoSpeedHold(
            self.ego_vehicles[0],
            self.ego_initial_speed,
        )

        # =========================
        # lead_vehicle 三阶段行为
        # =========================
        lead_vehicle_sequence = py_trees.composites.Sequence(name="LeadVehicleThreePhases")

        # phase1: 当前车道行驶，直到触发右变道
        phase1 = py_trees.composites.Parallel(
            name="Lead_Phase1_FollowLaneUntilTrigger",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        phase1_follow = WaypointFollower(
            self.lead_vehicle,
            self.lead_vehicle_speed,
            name="Lead_Phase1_WaypointFollower",
        )
        phase1_trigger = InTriggerDistanceToVehicle(
            self.lead_vehicle,
            self.accident_vehicle,
            self.phase1_trigger_distance,
            name="Lead_Phase1_TriggerLaneChange",
        )
        phase1.add_children([phase1_follow, phase1_trigger])

        # phase2: 向右变道
        phase2 = LaneChange(
            self.lead_vehicle,
            speed=self.lead_vehicle_speed,
            direction='right',
            distance_same_lane=self._distance_same_lane,
            distance_other_lane=self._distance_other_lane,
            distance_lane_change=self._distance_lane_change,
            lane_changes=1,
            name="Lead_Phase2_LaneChangeRight",
        )

        # phase3: 变道后沿新车道继续前进
        phase3 = py_trees.composites.Parallel(
            name="Lead_Phase3_FollowNewLane",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        phase3_follow = WaypointFollower(
            self.lead_vehicle,
            self.lead_vehicle_speed,
            name="Lead_Phase3_WaypointFollower",
        )
        phase3_finish = DriveDistance(
            self.lead_vehicle,
            self.phase3_drive_distance,
            name="Lead_Phase3_DriveDistance",
        )
        phase3.add_children([phase3_follow, phase3_finish])

        lead_vehicle_sequence.add_children([phase1, phase2, phase3])

        # =========================
        # 新增背景车辆：一直保持车道行驶
        # =========================
        background_flow = py_trees.composites.Parallel(
            name="BackgroundTrafficFlow",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )

        for idx, bg_vehicle in enumerate(self.background_vehicles):
            bg_follow = WaypointFollower(
                bg_vehicle,
                self.background_vehicle_speed,
                name=f"BackgroundVehicle_{idx}_WaypointFollower",
            )
            background_flow.add_child(bg_follow)

        root.add_children([ego_speed_hold, lead_vehicle_sequence, background_flow])
        return root

    def _create_test_criteria(self):
        criteria = []
        if not self.ego_vehicle or not self.accident_vehicle:
            return criteria

        # 规则1：刹车减速
        criteria.append(
            StaticObstacleBrakeSlowDownCriterion(
                actor=self.ego_vehicle,
                hazard_actor=self.accident_vehicle,
                trigger_distance=60.0,
                decel_threshold=3.0,
                min_speed_after=5.0,
            )
        )

        # 规则2：绕行通过（开过事故车 + 偏移1.5米）
        criteria.append(
            StaticObstacleSafePassCriterion(
                actor=self.ego_vehicle,
                hazard_actor=self.accident_vehicle,
                lateral_safe_threshold=1.5,
                route_center_y=self.global_y,
            )
        )

        # 规则3：到达终点 (396.2, 42)
        criteria.append(
            ReachEndPointCriterion(
                actor=self.ego_vehicle,
                end_x=396.2,
                end_y=42.0,
                end_z=0.0,
                distance_threshold=20.0
            )
        )
        criteria.append(
            MinTTCAutoCriterion(
                actor=self.ego_vehicles[0],
                other_actors=self.other_actors,
                distance_threshold=40.0,
                forward_angle_deg=140.0,
                terminate_on_failure=False)
        )

        return criteria

    def __del__(self):
        self.remove_all_actors()