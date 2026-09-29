import random

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    CollisionTest,
    BarrierSlowDownCriterion,
    BarrierDetourCriterion,
    BarrierReachGoalCriterion,
    MinTTCAutoCriterion
)

class PassiveEgoSpeedHold(py_trees.behaviour.Behaviour):
    """Keep the ego at a target speed until the driver provides manual input."""

    def __init__(self, actor, target_speed, name="PassiveEgoSpeedHold"):
        super(PassiveEgoSpeedHold, self).__init__(name)
        self.actor = actor
        self.target_speed = target_speed
        self._manual_override_detected = False

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor:
            return new_status

        control = self.actor.get_control()
        passive_input = (
            abs(control.steer) < 0.05 and
            control.brake < 0.05 and
            control.throttle < 0.05
        )

        if not passive_input:
            self._manual_override_detected = True

        if not self._manual_override_detected and passive_input:
            transform = self.actor.get_transform()
            forward_vector = transform.get_forward_vector()
            self.actor.set_target_velocity(carla.Vector3D(
                forward_vector.x * self.target_speed,
                forward_vector.y * self.target_speed,
                forward_vector.z * self.target_speed
            ))

        return new_status


class StaticBarrier(BasicScenario):
    """Static cone barrier scenario with light deterministic background traffic."""

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=60):
        self.timeout = timeout

        self._ego_initial_speed =13.0
        self._barrier_obstacles = []
        self._traffic_seed = 21
        self._traffic_rng = random.Random(self._traffic_seed)
        self._background_vehicles = []
        self._tm_port = CarlaDataProvider.get_traffic_manager_port()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(self._tm_port)

        self._barrier_distance = 40.0
        self._barrier_lateral_min = -0.2
        self._barrier_lateral_max = 5.0
        self._num_barrier_cones = 6

        self._longitudinal_length = 50.0
        self._longitudinal_spacing = 2.0
        self._longitudinal_lateral_offset = -1.4

        super(StaticBarrier, self).__init__(
            "StaticBarrier",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

    @staticmethod
    def _is_driving_lane(waypoint):
        return waypoint is not None and waypoint.lane_type == carla.LaneType.Driving

    @staticmethod
    def _lane_alignment(reference_wp, candidate_wp):
        ref_forward = reference_wp.transform.get_forward_vector()
        cand_forward = candidate_wp.transform.get_forward_vector()
        return ref_forward.x * cand_forward.x + ref_forward.y * cand_forward.y + ref_forward.z * cand_forward.z

    def _find_adjacent_driving_lane(self, reference_wp, want_same_direction):
        for method_name in ("get_left_lane", "get_right_lane"):
            current_wp = reference_wp
            for _ in range(3):
                current_wp = getattr(current_wp, method_name)() if current_wp else None
                if current_wp is None:
                    break
                if not self._is_driving_lane(current_wp):
                    continue

                alignment = self._lane_alignment(reference_wp, current_wp)
                if want_same_direction and alignment > 0.2:
                    return current_wp
                if not want_same_direction and alignment < -0.2:
                    return current_wp
        return None

    def _get_left_same_direction_lanes(self, reference_wp, max_lanes=2):
        lanes = []
        current_wp = reference_wp
        while current_wp is not None and len(lanes) < max_lanes:
            current_wp = current_wp.get_left_lane()
            if current_wp is None:
                break
            if not self._is_driving_lane(current_wp):
                continue
            if self._lane_alignment(reference_wp, current_wp) <= 0.2:
                continue
            lanes.append(current_wp)
        return lanes

    def _find_left_flow_lanes(self, ego_wp, max_lanes=2):
        if ego_wp is None:
            return []

        search_offsets = [0.0, 15.0, 30.0, 45.0, 60.0]
        for offset in search_offsets:
            anchor_wp = self._advance_waypoint(ego_wp, offset)
            lanes = self._get_left_same_direction_lanes(anchor_wp, max_lanes=max_lanes)
            if lanes:
                return lanes
        return []

    @staticmethod
    def _advance_waypoint(waypoint, distance, step=5.0):
        if waypoint is None:
            return None

        current_wp = waypoint
        remaining = abs(distance)
        if remaining < 0.1:
            return current_wp

        while remaining > 0.1 and current_wp is not None:
            travel_step = min(step, remaining)
            candidates = current_wp.next(travel_step) if distance >= 0.0 else current_wp.previous(travel_step)
            if not candidates:
                break
            current_wp = candidates[0]
            remaining -= travel_step

        return current_wp

    def _get_base_waypoint(self):
        base_location = carla.Location(x=264.43, y=173.68, z=0.0)
        return CarlaDataProvider.get_map().get_waypoint(base_location)

    def _get_barrier_waypoint(self):
        base_wp = self._get_base_waypoint()
        if base_wp is None:
            return None

        barrier_wps = base_wp.next(self._barrier_distance)
        return barrier_wps[0] if barrier_wps else None

    def _get_route_anchor_locations(self):
        route_start_loc = self.config.trigger_points[0].location
        route_end_loc = carla.Location(x=314.43, y=150.58, z=0.0)

        if self.config.route:
            route_start_loc = self.config.route[0][0].location
            route_end_loc = self.config.route[-1][0].location

        return route_start_loc, route_end_loc

    def _get_target_left_lane_waypoint(self):
        barrier_wp = self._get_barrier_waypoint()
        if barrier_wp is None:
            return None

        left_lanes = self._get_left_same_direction_lanes(barrier_wp, max_lanes=1)
        return left_lanes[0] if left_lanes else None

    def _spawn_background_vehicle(self, waypoint, speed_diff):
        if waypoint is None:
            return None

        vehicle_models = [
            "vehicle.tesla.model3",
            "vehicle.audi.tt",
            "vehicle.lincoln.mkz_2020",
            "vehicle.mercedes.coupe_2020",
        ]
        model = self._traffic_rng.choice(vehicle_models)
        actor = CarlaDataProvider.request_new_actor(
            model,
            waypoint.transform,
            rolename="background",
            autopilot=True,
            actor_category="car"
        )
        if actor is None:
            return None

        self._tm.auto_lane_change(actor, False)
        self._tm.distance_to_leading_vehicle(actor, 6.0)
        self._tm.vehicle_percentage_speed_difference(actor, speed_diff)
        self._tm.update_vehicle_lights(actor, True)

        self._background_vehicles.append(actor)
        self.other_actors.append(actor)
        return actor

    def _spawn_background_traffic(self, ego_wp, barrier_wp):
        if ego_wp is None and barrier_wp is None:
            return

        left_lanes = self._find_left_flow_lanes(ego_wp, max_lanes=2)
        if not left_lanes:
            reference_wp = barrier_wp if barrier_wp is not None else ego_wp
            if reference_wp is not None:
                left_lanes = self._get_left_same_direction_lanes(reference_wp, max_lanes=2)
                if not left_lanes:
                    fallback_lane = self._find_adjacent_driving_lane(reference_wp, want_same_direction=True)
                    if fallback_lane is not None:
                        left_lanes = [fallback_lane]

        spawn_plan = []
        if len(left_lanes) > 0:
            spawn_plan.extend([
                (self._advance_waypoint(left_lanes[0], 15.0), -18.0),
                (self._advance_waypoint(left_lanes[0], 35.0), -16.0),
                (self._advance_waypoint(left_lanes[0], 55.0), -14.0),
                (self._advance_waypoint(left_lanes[0], 75.0), -12.0),
                (self._advance_waypoint(left_lanes[0], 95.0), -10.0),
            ])
        if len(left_lanes) > 1:
            spawn_plan.extend([
                (self._advance_waypoint(left_lanes[1], 25.0), -16.0),
                (self._advance_waypoint(left_lanes[1], 55.0), -14.0),
                (self._advance_waypoint(left_lanes[1], 85.0), -12.0),
            ])
        elif barrier_wp is not None:
            spawn_plan.extend([
                (self._advance_waypoint(barrier_wp, 25.0), -8.0),
                (self._advance_waypoint(barrier_wp, 55.0), -6.0),
            ])

        # ===================== 【新增：自车车道 + 自车后方 生成车辆】 =====================
        # 在自车所在车道、自车后方 15米 / 35米 / 55米 各生成一辆车
        if ego_wp is not None:
            # 向后（自车后方）15米
            ego_back_wp1 = self._advance_waypoint(ego_wp, -15.0)
            # 向后 35米
            ego_back_wp2 = self._advance_waypoint(ego_wp, -35.0)
            # 向后 55米
            ego_back_wp3 = self._advance_waypoint(ego_wp, -55.0)

            # 速度比限速慢 10%~20%，不会超车
            spawn_plan.append((ego_back_wp1, -10.0))
            spawn_plan.append((ego_back_wp2, -15.0))
            spawn_plan.append((ego_back_wp3, -20.0))
        # ============================================================================

        for waypoint, speed_diff in spawn_plan:
            self._spawn_background_vehicle(waypoint, speed_diff)


    def _initialize_actors(self, config):
        ego = self.ego_vehicles[0]
        ego_forward = ego.get_transform().get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(
            ego_forward.x * self._ego_initial_speed,
            ego_forward.y * self._ego_initial_speed,
            ego_forward.z * self._ego_initial_speed
        ))

        self._tm.set_random_device_seed(self._traffic_seed)
        self._tm.global_percentage_speed_difference(0.0)

        carla_map = CarlaDataProvider.get_map()
        ego_wp = carla_map.get_waypoint(ego.get_location())
        barrier_wp = self._get_barrier_waypoint()
        if barrier_wp is None:
            return

        barrier_transform = barrier_wp.transform
        barrier_right_vec = barrier_transform.get_right_vector()
        cone_model = "static.prop.constructioncone"

        for index in range(self._num_barrier_cones):
            fraction = index / float(self._num_barrier_cones - 1)
            lateral_offset = self._barrier_lateral_min + (
                self._barrier_lateral_max - self._barrier_lateral_min
            ) * fraction

            cone_location = barrier_transform.location + barrier_right_vec * lateral_offset
            cone_location.z += 0.05
            cone_transform = carla.Transform(cone_location, barrier_transform.rotation)

            cone = CarlaDataProvider.request_new_actor(cone_model, cone_transform)
            if cone:
                cone.set_simulate_physics(False)
                self.other_actors.append(cone)
                self._barrier_obstacles.append(cone)

        current_wp = barrier_wp
        accumulated_distance = 0.0
        while accumulated_distance < self._longitudinal_length and current_wp is not None:
            current_right_vec = current_wp.transform.get_right_vector()

            cone_location = current_wp.transform.location + current_right_vec * self._longitudinal_lateral_offset
            cone_location.z += 0.05
            cone_transform = carla.Transform(cone_location, current_wp.transform.rotation)

            cone = CarlaDataProvider.request_new_actor(cone_model, cone_transform)
            if cone:
                cone.set_simulate_physics(False)
                self.other_actors.append(cone)
                self._barrier_obstacles.append(cone)

            next_wps = current_wp.next(self._longitudinal_spacing)
            if not next_wps:
                break
            current_wp = next_wps[0]
            accumulated_distance += self._longitudinal_spacing

        if ego_wp is not None:
            self._spawn_background_traffic(ego_wp, barrier_wp)
        self._export_curve_points()

    # 
    def _export_curve_points(self, filename="curve_points.txt"):
        """
        导出弯道每隔2m的坐标点，格式：<position x="..." y="..." z="0.0" />
        """
        import carla

        base_location = carla.Location(x=264.43, y=173.68, z=0.0)
        carla_map = CarlaDataProvider.get_map()
        current_wp = carla_map.get_waypoint(base_location)

        step = 2.0
        total_length = 250.0  # 足够覆盖 170+50 米路障
        lines = []
        dist = 0.0

        while dist < total_length:
            loc = current_wp.transform.location
            line = f'<position x="{loc.x:.2f}" y="{loc.y:.2f}" z="0.0" />'
            lines.append(line)

            next_wps = current_wp.next(step)
            if not next_wps:
                break
            current_wp = next_wps[0]
            dist += step

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n✅ 弯道坐标已导出到：{filename}\n")

    #

    def _create_behavior(self):
        root = py_trees.composites.Parallel(
            "StaticBarrierBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        total_distance = self._barrier_distance + self._longitudinal_length + 50.0
        root.add_child(PassiveEgoSpeedHold(self.ego_vehicles[0], self._ego_initial_speed))
        root.add_child(DriveDistance(self.ego_vehicles[0], total_distance))
        return root

    def _create_test_criteria(self):
        criteria = []
        ego = self.ego_vehicles[0]

        barrier_wp = self._get_barrier_waypoint()
        if not barrier_wp or not self._barrier_obstacles:
            return criteria

        barrier_loc = barrier_wp.transform.location
        original_lane_y = barrier_loc.y
        goal_loc = carla.Location(x=353.59, y=124.05, z=0.0)

        # 1. 减速
        criteria.append(BarrierSlowDownCriterion(
            ego, barrier_location=barrier_loc, trigger_dist=70.0, min_decel=4.0
        ))

        # 2. 绕行
        criteria.append(BarrierDetourCriterion(
            ego, obstacle_actors=self._barrier_obstacles,
            original_lane_y=original_lane_y, lateral_threshold=1.5, collision_distance=1.6
        ))

        # 3. 到达终点
        criteria.append(BarrierReachGoalCriterion(
            ego, goal_location=goal_loc, distance_threshold=5.0
        ))
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
