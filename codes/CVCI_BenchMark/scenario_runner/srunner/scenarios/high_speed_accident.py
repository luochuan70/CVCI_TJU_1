import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario


def get_value_parameter(config, name, p_type, default):
    parameter = getattr(config, "other_parameters", {}).get(name, {})
    if "value" not in parameter:
        return default
    return p_type(parameter["value"])


class HighSpeedAccident(BasicScenario):
    _REFERENCE_ROUTE_START = carla.Location(x=166.32, y=206.90, z=0.5)
    _REFERENCE_ROUTE_END = carla.Location(x=296.34, y=155.95, z=0.5)

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=120):
        self._init_speed = get_value_parameter(config, "init_speed", float, 10.0)
        self._accident_lateral_distance = get_value_parameter(config, "lateral_distance", float, 4.63)
        self._accident_longitudinal_distance = 47.95
        self._accident_yaw = -135.0

        super(HighSpeedAccident, self).__init__(
            "HighSpeedAccident",
            ego_vehicles,
            config,
            world,
            debug_mode=debug_mode,
            criteria_enable=criteria_enable
        )

    def _get_route_anchor_locations(self):
        """Use the route XML endpoints when they are available."""
        route_start_loc = self.config.trigger_points[0].location
        route_end_loc = self._REFERENCE_ROUTE_END

        if self.config.route:
            route_start_loc = self.config.route[0][0].location
            route_end_loc = self.config.route[-1][0].location

        return route_start_loc, route_end_loc

    def _get_reference_route_locations(self):
        return self._REFERENCE_ROUTE_START, self._REFERENCE_ROUTE_END

    def _get_route_length(self):
        route_start_loc, route_end_loc = self._get_route_anchor_locations()
        return route_start_loc.distance(route_end_loc)

    def _get_scenario_end_distance(self):
        route_start_loc, route_end_loc = self._get_route_anchor_locations()
        forward_xy = self._normalize_vector_2d(
            route_end_loc.x - route_start_loc.x,
            route_end_loc.y - route_start_loc.y
        )
        accident_loc = self._get_accident_transform().location
        to_accident_x = accident_loc.x - route_start_loc.x
        to_accident_y = accident_loc.y - route_start_loc.y
        longitudinal_to_accident = to_accident_x * forward_xy[0] + to_accident_y * forward_xy[1]
        return max(10.0, longitudinal_to_accident + 50.0)

    @staticmethod
    def _normalize_vector_2d(x_value, y_value):
        magnitude = (x_value ** 2 + y_value ** 2) ** 0.5
        if magnitude < 1e-6:
            return 1.0, 0.0
        return x_value / magnitude, y_value / magnitude

    def _get_accident_transform(self):
        route_start_loc, route_end_loc = self._get_reference_route_locations()
        forward_xy = self._normalize_vector_2d(
            route_end_loc.x - route_start_loc.x,
            route_end_loc.y - route_start_loc.y
        )
        right_xy = (-forward_xy[1], forward_xy[0])

        accident_loc = carla.Location(
            x=route_start_loc.x + forward_xy[0] * self._accident_longitudinal_distance
            + right_xy[0] * self._accident_lateral_distance,
            y=route_start_loc.y + forward_xy[1] * self._accident_longitudinal_distance
            + right_xy[1] * self._accident_lateral_distance,
            z=0.5
        )
        accident_rot = carla.Rotation(yaw=self._accident_yaw)
        return carla.Transform(accident_loc, accident_rot)

    def _spawn_accident_actor(self):
        base_transform = self._get_accident_transform()
        route_start_loc, route_end_loc = self._get_reference_route_locations()
        forward_xy = self._normalize_vector_2d(
            route_end_loc.x - route_start_loc.x,
            route_end_loc.y - route_start_loc.y
        )
        right_xy = (-forward_xy[1], forward_xy[0])

        candidate_offsets = [
            (0.0, 0.0),
            (1.5, 0.0),
            (-1.5, 0.0),
            (0.0, 0.8),
            (0.0, -0.8),
            (2.5, 1.0),
            (2.5, -1.0),
            (-2.5, 1.0),
            (-2.5, -1.0),
            (4.0, 0.0),
            (6.0, 0.0),
            (8.0, 0.0),
            (-4.0, 0.0),
            (4.0, 1.2),
            (4.0, -1.2),
            (6.0, 1.2),
            (6.0, -1.2),
        ]
        candidate_blueprints = [
            'vehicle.tesla.model3',
            'vehicle.lincoln.mkz_2020',
            'vehicle.audi.tt',
            'vehicle.nissan.patrol_2021',
        ]

        for blueprint in candidate_blueprints:
            for longitudinal_offset, lateral_offset in candidate_offsets:
                spawn_loc = carla.Location(
                    x=base_transform.location.x + forward_xy[0] * longitudinal_offset + right_xy[0] * lateral_offset,
                    y=base_transform.location.y + forward_xy[1] * longitudinal_offset + right_xy[1] * lateral_offset,
                    z=0.8
                )
                spawn_tf = carla.Transform(spawn_loc, base_transform.rotation)
                try:
                    actor = CarlaDataProvider.request_new_actor(blueprint, spawn_tf)
                    if actor:
                        return actor
                except Exception:
                    continue

        return None

    def _initialize_actors(self, config):
        ego = self.ego_vehicles[0]
        forward_vector = ego.get_transform().get_forward_vector()

        ego.set_target_velocity(carla.Vector3D(
            forward_vector.x * self._init_speed,
            forward_vector.y * self._init_speed,
            forward_vector.z * self._init_speed
        ))

        accident_car = self._spawn_accident_actor()
        if accident_car:
            self.other_actors.append(accident_car)
            accident_car.set_light_state(carla.VehicleLightState(carla.VehicleLightState.All))
        else:
            print("Actor creation failed: cannot find a valid spawn transform for hazard vehicle")

    def _create_behavior(self):
        root = py_trees.composites.Parallel(
            "NightBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        target_distance = self._get_scenario_end_distance()
        root.add_child(DriveDistance(self.ego_vehicles[0], target_distance))
        return root

    def _create_test_criteria(self):
        criteria = []
        ego = self.ego_vehicles[0]
        route_start_loc, route_end_loc = self._get_route_anchor_locations()

        if len(self.other_actors) == 0:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
            return criteria

        hazard_vehicle = self.other_actors[0]

        from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
            HighSpeedBrakeCriterion,
            HighSpeedBypassCriterion,
            HighSpeedResumeCriterion,
            MinTTCAutoCriterion
        )

        criteria.append(
            HighSpeedBrakeCriterion(
                actor=ego,
                hazard_actor=hazard_vehicle,
                route_start_location=route_start_loc,
                route_end_location=route_end_loc,
                trigger_distance=50.0,
                brake_threshold=0.3,
            )
        )

        criteria.append(
            HighSpeedBypassCriterion(
                actor=ego,
                hazard_actor=hazard_vehicle,
                route_start_location=route_start_loc,
                route_end_location=route_end_loc,
                safe_lateral_margin=2.4,
                danger_lateral_margin=1.9,
                passing_longitudinal_zone=7.5,
                min_route_offset=0.55,
            )
        )

        criteria.append(
            HighSpeedResumeCriterion(
                actor=ego,
                hazard_actor=hazard_vehicle,
                route_start_location=route_start_loc,
                route_end_location=route_end_loc,
                escape_distance=15.0,
                min_resume_speed=5.0,
            )
        )

        criteria.append(CollisionTest(self.ego_vehicles[0]))
        criteria.append(MinTTCAutoCriterion(actor=self.ego_vehicles[0],
                            other_actors=self.other_actors,
                            distance_threshold=40.0,
                            forward_angle_deg=140.0,
                            terminate_on_failure=False))
        return criteria
