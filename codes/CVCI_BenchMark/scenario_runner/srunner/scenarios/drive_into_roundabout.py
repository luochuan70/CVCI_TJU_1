#!/usr/bin/env python

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    WaypointFollower,
)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    CollisionTest,
    RoundaboutDecelerateCriterion,
    RoundaboutYieldConvoyCriterion,
    RoundaboutSafePassCriterion,
    MinTTCAutoCriterion
)


class RoundaboutMergeConflict(BasicScenario):

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        randomize=False,
        debug_mode=False,
        criteria_enable=True,
        timeout=60,
    ):
        self.timeout = timeout
        self.world = world
        self._map = CarlaDataProvider.get_map()

        # 1. 触发点
        trigger_transform = config.trigger_points[0]

        self._trigger_location = carla.Location(x=4.9, y=40.7, z=0.0)

        self._init_speed = float(
            config.other_parameters.get("init_speed", {}).get("value", 11.0)
        )

        # 2. 基础参数
        self._start_distance = 15.0
        self._adversary_speed = float(
            config.other_parameters.get("adversary_speed", {}).get("value", 6.0)
        )
        self._adversary_intitial_speed = float(
            config.other_parameters.get("adversary_intitial_speed", {}).get("value", 2.0)
        )
        self._chedui_speed = 5.0
        self._num_convoy_vehicles = 5

        manual_adversary_waypoints = [
            carla.Location(x=-0.17, y=23.48, z=0.00),
            carla.Location(x=1.87, y=23.48, z=0.00),
            carla.Location(x=3.86, y=23.26, z=0.00),
            carla.Location(x=5.86, y=22.73, z=0.00),
            carla.Location(x=7.82, y=22.08, z=0.00),
            carla.Location(x=9.64, y=21.21, z=0.00),
            carla.Location(x=11.35, y=20.12, z=0.00),
            carla.Location(x=13.03, y=19.00, z=0.00),
            carla.Location(x=14.67, y=17.79, z=0.00),
            carla.Location(x=16.16, y=16.44, z=0.00),
            carla.Location(x=17.59, y=15.03, z=0.00),
            carla.Location(x=18.88, y=13.40, z=0.00),
            carla.Location(x=19.98, y=11.70, z=0.00),
            carla.Location(x=20.88, y=9.88, z=0.00),
            carla.Location(x=21.53, y=7.92, z=0.00),
            carla.Location(x=22.00, y=5.95, z=0.00),
            carla.Location(x=22.25, y=3.90, z=0.00),
            carla.Location(x=22.42, y=1.87, z=0.00),
            carla.Location(x=22.52, y=-0.20, z=0.00),
            carla.Location(x=22.47, y=-2.21, z=0.00),
            carla.Location(x=22.12, y=-4.23, z=0.00),
            carla.Location(x=21.64, y=-6.18, z=0.00),
            carla.Location(x=20.90, y=-8.08, z=0.00),
            carla.Location(x=20.04, y=-9.93, z=0.00),
            carla.Location(x=18.96, y=-11.64, z=0.00),
            carla.Location(x=17.84, y=-13.32, z=0.00),
            carla.Location(x=16.58, y=-14.89, z=0.00),
            carla.Location(x=15.16, y=-16.39, z=0.00),
            carla.Location(x=13.70, y=-17.80, z=0.00),
            carla.Location(x=12.55, y=-19.44, z=0.00),
            carla.Location(x=11.51, y=-21.16, z=0.00),
            carla.Location(x=10.66, y=-23.00, z=0.00),
            carla.Location(x=9.99, y=-24.94, z=0.00),
            carla.Location(x=9.52, y=-27.03, z=0.00),
            carla.Location(x=9.15, y=-29.01, z=0.00),
            carla.Location(x=8.85, y=-31.00, z=0.00),
            carla.Location(x=8.66, y=-33.08, z=0.00),
            carla.Location(x=8.46, y=-35.07, z=0.00),
            carla.Location(x=8.25, y=-37.11, z=0.00),
            carla.Location(x=8.20, y=-39.16, z=0.00),
            carla.Location(x=8.11, y=-41.17, z=0.00),
            carla.Location(x=8.01, y=-43.19, z=0.00),
            carla.Location(x=7.95, y=-45.24, z=0.00),
            carla.Location(x=7.95, y=-47.37, z=0.00),
            carla.Location(x=7.93, y=-49.47, z=0.00),
            carla.Location(x=7.92, y=-51.54, z=0.00),
            carla.Location(x=7.98, y=-53.59, z=0.00),
            carla.Location(x=8.02, y=-55.61, z=0.00)
        ]

        # 先把手写点吸附到 Driving lane
        self._adversary_waypoints = manual_adversary_waypoints


        self._convoy_seed_location = carla.Location(x=0.6, y=21.0, z=0.5)

        # 车距（沿车道弧长方向的间距）
        self._convoy_spacing = 8.0

        # 轨迹采样间隔与长度
        self._convoy_plan_step = 4.0
        self._convoy_plan_length = 40

        # 基于 seed waypoint 生成整支车队
        self._convoy_seed_wp = self._get_strict_driving_waypoint(self._convoy_seed_location)

        # 车队出生点：从 seed 沿同一条 lane 向后依次排布
        self._convoy_start_transforms = self._generate_convoy_start_transforms(
            seed_wp=self._convoy_seed_wp,
            num_vehicles=self._num_convoy_vehicles,
            spacing=self._convoy_spacing,
        )

        # 车队统一轨迹：从 seed 沿同一条 lane 向前采样
        self._convoy_waypoints = self._generate_lane_plan_from_wp(
            start_wp=self._convoy_seed_wp,
            step=self._convoy_plan_step,
            num_points=self._convoy_plan_length,
        )

        self.convoy_plans = [
            list(self._convoy_waypoints) for _ in range(self._num_convoy_vehicles)
        ]

        super(RoundaboutMergeConflict, self).__init__(
            "RoundaboutMergeConflict",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable,
        )

    def _get_strict_driving_waypoint(self, location):
        """
        获取一个明确的 Driving lane waypoint。
        如果最近点不可靠，直接报错，避免偷偷吸附到草坪旁错误车道。
        """
        wp = self._map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            raise RuntimeError(
                f"[RoundaboutMergeConflict] Cannot find Driving waypoint near location: {location}"
            )
        return wp



    def _step_along_lane(self, wp, distance, forward=True):
        """
        沿同一条 lane 前进/后退一小段。
        为了更稳，分多步走，避免在环岛分叉处跳 branch。
        """
        current_wp = wp
        remaining = distance
        step = 1.0

        while remaining > 1e-3:
            ds = min(step, remaining)
            candidates = current_wp.next(ds) if forward else current_wp.previous(ds)
            if not candidates:
                break

            # 优先选 lane_id / road_id 相同的 waypoint，避免跳到旁边车道
            same_lane = [
                c for c in candidates
                if c.lane_id == current_wp.lane_id and c.road_id == current_wp.road_id
            ]
            if len(same_lane) > 0:
                current_wp = same_lane[0]
            else:
                current_yaw = current_wp.transform.rotation.yaw
                current_yaw = (current_yaw + 360.0) % 360.0

                def yaw_diff(c):
                    cy = (c.transform.rotation.yaw + 360.0) % 360.0
                    d = abs(cy - current_yaw)
                    return min(d, 360.0 - d)

                current_wp = min(candidates, key=yaw_diff)

            remaining -= ds

        return current_wp

    def _generate_convoy_start_transforms(self, seed_wp, num_vehicles, spacing):
        """
        从 seed waypoint 开始，沿同一车道向后生成 N 辆车的出生位姿
        """
        transforms = []
        for i in range(num_vehicles):
            wp_i = self._step_along_lane(seed_wp, i * spacing, forward=False)
            tf = carla.Transform(
                carla.Location(
                    x=wp_i.transform.location.x,
                    y=wp_i.transform.location.y,
                    z=0.5
                ),
                wp_i.transform.rotation
            )
            transforms.append(tf)
        return transforms

    def _generate_lane_plan_from_wp(self, start_wp, step=4.0, num_points=30):
        """
        从指定 waypoint 开始，沿同一条 lane 向前生成轨迹
        """
        plan = []
        current_wp = start_wp

        for _ in range(num_points):
            loc = current_wp.transform.location
            plan.append(carla.Location(x=loc.x, y=loc.y, z=0.5))
            current_wp = self._step_along_lane(current_wp, step, forward=True)

        return plan


    def _initialize_actors(self, config):
        # 1. 障碍车初始化
        obstacle_transform = carla.Transform(
            carla.Location(x=3, y=32.0, z=0.5),
            carla.Rotation(pitch=0.0, yaw=270.0, roll=0.0)
        )
        obstacle_actor = CarlaDataProvider.request_new_actor(
            'vehicle.toyota.prius',
            obstacle_transform
        )
        if obstacle_actor:
            obstacle_actor.set_simulate_physics(True)
            self.other_actors.append(obstacle_actor)

        adversary_actor = CarlaDataProvider.request_new_actor(config.other_actors[0].model, config.other_actors[0].transform)

        if adversary_actor:
            adversary_actor.set_simulate_physics(True)

            # 让 original_adversary 一出生就具有目标速度
            adv_tf = adversary_actor.get_transform()
            adv_forward = adv_tf.get_forward_vector()
            adversary_actor.set_target_velocity(
                carla.Vector3D(
                    x=adv_forward.x * self._adversary_intitial_speed,
                    y=adv_forward.y * self._adversary_intitial_speed,
                    z=adv_forward.z * self._adversary_intitial_speed
                )
            )

        self.other_actors.append(adversary_actor)

        # 3. 车队初始化
        self.convoy_actors = []
        for i in range(self._num_convoy_vehicles):
            spawn_transform = self._convoy_start_transforms[i]
            actor = CarlaDataProvider.request_new_actor(
                'vehicle.lincoln.mkz_2017',
                spawn_transform
            )
            if actor:
                actor.set_simulate_physics(True)
                self.other_actors.append(actor)
                self.convoy_actors.append(actor)

        # 4. 自车初始速度
        if self.ego_vehicles:
            ego = self.ego_vehicles[0]
            forward_vector = ego.get_transform().get_forward_vector()
            velocity_ms = self._init_speed
            ego.set_target_velocity(
                carla.Vector3D(
                    x=forward_vector.x * velocity_ms,
                    y=forward_vector.y * velocity_ms,
                    z=forward_vector.z * velocity_ms
                )
            )


    def _create_behavior(self):
        original_adversary = self.other_actors[1]

        # 1. 原 adversary 行为
        original_sequence = py_trees.composites.Sequence("OriginalAdversaryBehavior")

        trigger_1 = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            self._trigger_location,
            self._start_distance,
            name="Wait for ego (Original)"
        )

        keep_driving_1 = WaypointFollower(
            original_adversary,
            target_speed=self._adversary_speed,
            plan=self._adversary_waypoints,
            avoid_collision=False,
            name="FollowWaypoints (Original)"
        )

        original_sequence.add_children([trigger_1, keep_driving_1])

        # 2. 车队行为
        convoy_sequence = py_trees.composites.Sequence("ConvoyBehavior")

        trigger_2 = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            self._trigger_location,
            self._start_distance,
            name="Wait for ego (Convoy)"
        )

        convoy_parallel = py_trees.composites.Parallel(
            "ConvoyParallelBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )

        for idx, actor in enumerate(self.convoy_actors):
            keep_driving_convoy = WaypointFollower(
                actor,
                target_speed=self._chedui_speed,
                plan=self.convoy_plans[idx],
                avoid_collision=False,
                name=f"FollowWaypoints_Vehicle_{idx}"
            )
            convoy_parallel.add_child(keep_driving_convoy)

        convoy_sequence.add_children([trigger_2, convoy_parallel])

        # 3. 总控制器
        root = py_trees.composites.Parallel(
            "AllAdversariesRoot",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )
        root.add_children([original_sequence, convoy_sequence])

        return root


    def _create_test_criteria(self):
        criteria = [CollisionTest(self.ego_vehicles[0])]

        ego = self.ego_vehicles[0]
        adversary = self.other_actors[1]

        criteria.append(
            RoundaboutDecelerateCriterion(
                actor=ego,
                adversary=adversary,
                trigger_distance=20.0,
                speed_drop_ratio=0.75,
                brake_threshold=0.15,
                min_brake_duration=0.2
            )
        )

        criteria.append(
            RoundaboutYieldConvoyCriterion(
                actor=ego,
                convoy_actors=self.convoy_actors,
                merge_conflict_center=carla.Location(x=5.0, y=23.0, z=0.0),
                activation_radius=20.0,
                convoy_conflict_radius=12.0,
                ego_low_speed_threshold=2.0,
                required_pass_count=2
            )
        )

        criteria.append(
            RoundaboutSafePassCriterion(
                actor=ego,
                goal_location=carla.Location(x=3.7, y=-45.3, z=0.0),
                goal_dist_threshold=6.0,
                min_resume_speed=2.0
            )
        )
        criteria.append(
            MinTTCAutoCriterion(actor=self.ego_vehicles[0],
                                other_actors=self.other_actors,
                                distance_threshold=40.0,
                                forward_angle_deg=140.0,
                                terminate_on_failure=False)
        )

        return criteria

    def __del__(self):
        self.remove_all_actors()