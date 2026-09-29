import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import KeepVelocity, Idle
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
    InTriggerDistanceToVehicle,
    DriveDistance,
)
from srunner.scenarios.basic_scenario import BasicScenario


def get_value_parameter(config, name, p_type, default):
    parameter = getattr(config, "other_parameters", {}).get(name, {})
    if "value" not in parameter:
        return default
    return p_type(parameter["value"])


class PassiveEgoSpeedHold(py_trees.behaviour.Behaviour):
    """Keep the ego at a target speed only while no active control input is applied."""

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


class GhostProbeScenario(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=120):
        self.timeout = timeout
        self._bikes = []
        self._peds = []
        self._ego_initial_speed = get_value_parameter(config, "init_speed", float, 9.0)
        self._ped_start_distance = get_value_parameter(config, "trigger_distance", float, 18.0)
        self._bike_start_distance = max(self._ped_start_distance + 26.0, 40.0)
        super(GhostProbeScenario, self).__init__(
            "GhostProbe",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

    def _get_route_anchor_locations(self):
        route_start_loc = self.config.trigger_points[0].location
        route_end_loc = carla.Location(x=18.0, y=2.7, z=1.5)

        if self.config.route:
            route_start_loc = self.config.route[0][0].location
            route_end_loc = self.config.route[-1][0].location

        return route_start_loc, route_end_loc

    def _get_route_length(self):
        route_start_loc, route_end_loc = self._get_route_anchor_locations()
        return route_start_loc.distance(route_end_loc)

    def _initialize_actors(self, config):
        ego = self.ego_vehicles[0]
        forward_vector = ego.get_transform().get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(
            forward_vector.x * self._ego_initial_speed,
            forward_vector.y * self._ego_initial_speed,
            forward_vector.z * self._ego_initial_speed
        ))

        bike_models = ['vehicle.vespa.zx125', 'vehicle.gazelle.omafiets']
        bike_transforms = [
            carla.Transform(carla.Location(x=-16.8, y=6.00, z=1.5), carla.Rotation(yaw=0.0)),
            carla.Transform(carla.Location(x=-19.4, y=6.28, z=1.5), carla.Rotation(yaw=0.0)),
        ]

        for index, transform in enumerate(bike_transforms):
            actor = CarlaDataProvider.request_new_actor(bike_models[index], transform)
            if actor:
                self.other_actors.append(actor)
                self._bikes.append(actor)

        ped_models = [
            'walker.pedestrian.0009',
            'walker.pedestrian.0010',
            'walker.pedestrian.0013',
            'walker.pedestrian.0012',
        ]
        ped_transforms = [
            carla.Transform(carla.Location(x=-12.55, y=6.14, z=1.5), carla.Rotation(yaw=270.0)),
            carla.Transform(carla.Location(x=-12.40, y=6.34, z=1.5), carla.Rotation(yaw=270.0)),
            carla.Transform(carla.Location(x=-12.25, y=6.54, z=1.5), carla.Rotation(yaw=270.0)),
            carla.Transform(carla.Location(x=-12.10, y=6.74, z=1.5), carla.Rotation(yaw=270.0)),
        ]

        for index, transform in enumerate(ped_transforms):
            actor = CarlaDataProvider.request_new_actor(ped_models[index], transform)
            if actor:
                self.other_actors.append(actor)
                self._peds.append(actor)

    def _create_behavior(self):
        root = py_trees.composites.Parallel(
            "GhostProbeBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        scenario_flow = py_trees.composites.Parallel(
            "ScenarioFlow",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )

        scenario_flow.add_child(
            PassiveEgoSpeedHold(self.ego_vehicles[0], self._ego_initial_speed)
        )

        bike_sequence = py_trees.composites.Sequence("BikeMovement")
        if self._bikes:
            bike_sequence.add_child(
                InTriggerDistanceToVehicle(
                    self.ego_vehicles[0],
                    self._bikes[0],
                    distance=self._bike_start_distance
                )
            )
        bike_parallel = py_trees.composites.Parallel(
            "BikeMovementParallel",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )
        bike_speeds = [0.6, 0.5]
        for index, bike in enumerate(self._bikes):
            bike_parallel.add_child(
                KeepVelocity(bike, bike_speeds[index], force_speed=True)
            )
        bike_sequence.add_child(bike_parallel)
        scenario_flow.add_child(bike_sequence)

        pedestrian_sequence = py_trees.composites.Sequence("PedestrianMovement")
        cross_location = carla.Location(x=-11.8, y=2.7, z=0.0)
        pedestrian_sequence.add_child(
            InTriggerDistanceToLocation(
                self.ego_vehicles[0],
                cross_location,
                distance=self._ped_start_distance
            )
        )
        pedestrian_parallel = py_trees.composites.Parallel(
            "PedestrianMovementParallel",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )
        ped_speeds = [3.2, 3.1, 3.0, 2.9]
        ped_delays = [0.00, 0.05, 0.10, 0.15]
        for index, ped in enumerate(self._peds):
            ped_sequence = py_trees.composites.Sequence(f"Pedestrian{index}Sequence")
            ped_sequence.add_child(Idle(ped_delays[index]))
            ped_sequence.add_child(KeepVelocity(ped, ped_speeds[index]))
            pedestrian_parallel.add_child(ped_sequence)

        pedestrian_sequence.add_child(pedestrian_parallel)
        scenario_flow.add_child(pedestrian_sequence)

        root.add_child(scenario_flow)
        root.add_child(DriveDistance(self.ego_vehicles[0], distance=max(40.0, self._get_route_length() + 5.0)))

        return root

    def _create_test_criteria(self):
        criteria = []
        ego = self.ego_vehicles[0]
        route_start_loc, route_end_loc = self._get_route_anchor_locations()

        if len(self._bikes) > 0 and len(self._peds) > 0:
            bike_actor = self._bikes[0]
            first_pedestrian = self._peds[0]
            last_pedestrian = self._peds[-1]

            from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
                ScooterDecelerateCriterion,
                PedestrianStopCriterion,
                PedestrianResumeCriterion,
                MinTTCAutoCriterion)

            scooter_criterion = ScooterDecelerateCriterion(
                actor=ego,
                scooter_actor=bike_actor,
                route_start_location=route_start_loc,
                route_end_location=route_end_loc,
                trigger_distance=max(self._ped_start_distance + 12.0, 30.0),
                min_speed_drop=2,
                brake_threshold=0.15,
            )
            scooter_criterion.latest_reaction_distance = max(self._ped_start_distance - 2.0, 10.0)
            criteria.append(scooter_criterion)

            stop_criterion = PedestrianStopCriterion(
                actor=ego,
                pedestrian_actor=first_pedestrian,
                route_start_location=route_start_loc,
                route_end_location=route_end_loc,
                trigger_distance=max(self._ped_start_distance + 4.0, 16.0),
                stop_speed_threshold=1.5,
                activation_lateral_margin=2.8,
            )
            stop_criterion.minimum_stop_distance = -0.5
            criteria.append(stop_criterion)

            resume_criterion = PedestrianResumeCriterion(
                actor=ego,
                pedestrian_actor=last_pedestrian,
                route_start_location=route_start_loc,
                route_end_location=route_end_loc,
                resume_speed=3.0,
                safe_lateral_margin=1.5,
            )
            if hasattr(resume_criterion, 'pedestrian_actors'):
                resume_criterion.pedestrian_actors = self._peds
            criteria.append(resume_criterion)

        criteria.append(CollisionTest(ego))
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
