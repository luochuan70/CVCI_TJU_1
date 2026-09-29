#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic evaluation criteria required to analyze if a
scenario was completed successfully or failed.

Criteria should run continuously to monitor the state of a single actor, multiple
actors or environmental parameters. Hence, a termination is not required.

The atomic criteria are implemented with py_trees.
"""

import math
import numpy as np
import py_trees
import shapely.geometry

import sys
import os

import carla
def get_speed(actor):
    vel = actor.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType
from srunner.scenariomanager.timer import GameTime

class Criterion(py_trees.behaviour.Behaviour):

    """
    Base class for all criteria used to evaluate a scenario for success/failure

    Important parameters (PUBLIC):
    - name: Name of the criterion
    - actor: Actor of the criterion
    - optional: Indicates if a criterion is optional (not used for overall analysis)
    - terminate on failure: Whether or not the criteria stops on failure

    - test_status: Used to access the result of the criterion
    - success_value: Result in case of success (e.g. max_speed, zero collisions, ...)
    - acceptable_value: Result that does not mean a failure,  but is not good enough for a success
    - actual_value: Actual result after running the scenario
    - units: units of the 'actual_value'. This is a string and is used by the result writter
    """

    def __init__(self,
                 name,
                 actor,
                 optional=False,
                 terminate_on_failure=False):
        super(Criterion, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.name = name
        self.actor = actor
        self.optional = optional
        self._terminate_on_failure = terminate_on_failure
        self.test_status = "INIT"   # Either "INIT", "RUNNING", "SUCCESS", "ACCEPTABLE" or "FAILURE"

        # Attributes to compare the current state (actual_value), with the expected ones
        self.success_value = 0
        self.acceptable_value = None
        self.actual_value = 0
        self.units = "times"

        self.events = []  # List of events (i.e collision, sidewalk invasion...)

    def initialise(self):
        """
        Initialise the criterion. Can be extended by the user-derived class
        """
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        """
        Terminate the criterion. Can be extended by the user-derived class
        """
        if self.test_status in ('RUNNING', 'INIT'):
            self.test_status = "SUCCESS"

        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class MaxVelocityTest(Criterion):

    """
    This class contains an atomic test for maximum velocity.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - max_velocity_allowed: maximum allowed velocity in m/s
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, max_velocity, optional=False, name="CheckMaximumVelocity"):
        """
        Setup actor and maximum allowed velovity
        """
        super(MaxVelocityTest, self).__init__(name, actor, optional)
        self.success_value = max_velocity

    def update(self):
        """
        Check velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        velocity = CarlaDataProvider.get_velocity(self.actor)

        self.actual_value = max(velocity, self.actual_value)

        if velocity > self.success_value:
            self.test_status = "FAILURE"
        else:
            self.test_status = "SUCCESS"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class DrivenDistanceTest(Criterion):
    """
    This class contains an atomic test to check the driven distance

    Important parameters:
    - actor: CARLA actor to be used for this test
    - distance_success: If the actor's driven distance is more than this value (in meters),
                        the test result is SUCCESS
    - distance_acceptable: If the actor's driven distance is more than this value (in meters),
                           the test result is ACCEPTABLE
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, distance, acceptable_distance=None, optional=False, name="CheckDrivenDistance"):
        """
        Setup actor
        """
        super(DrivenDistanceTest, self).__init__(name, actor, optional)
        self.success_value = distance
        self.acceptable_value = acceptable_distance
        self._last_location = None

    def initialise(self):
        self._last_location = CarlaDataProvider.get_location(self.actor)
        super(DrivenDistanceTest, self).initialise()

    def update(self):
        """
        Check distance
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        location = CarlaDataProvider.get_location(self.actor)

        if location is None:
            return new_status

        if self._last_location is None:
            self._last_location = location
            return new_status

        self.actual_value += location.distance(self._last_location)
        self._last_location = location

        if self.actual_value > self.success_value:
            self.test_status = "SUCCESS"
        elif (self.acceptable_value is not None and
              self.actual_value > self.acceptable_value):
            self.test_status = "ACCEPTABLE"
        else:
            self.test_status = "RUNNING"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Set final status
        """
        if self.test_status != "SUCCESS":
            self.test_status = "FAILURE"
        self.actual_value = round(self.actual_value, 2)
        super(DrivenDistanceTest, self).terminate(new_status)


class AverageVelocityTest(Criterion):

    """
    This class contains an atomic test for average velocity.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - avg_velocity_success: If the actor's average velocity is more than this value (in m/s),
                            the test result is SUCCESS
    - avg_velocity_acceptable: If the actor's average velocity is more than this value (in m/s),
                               the test result is ACCEPTABLE
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, velocity, acceptable_velocity=None, optional=False,
                 name="CheckAverageVelocity"):
        """
        Setup actor and average velovity expected
        """
        super(AverageVelocityTest, self).__init__(name, actor, optional)
        self.success_value = velocity
        self.acceptable_value = acceptable_velocity
        self._last_location = None
        self._distance = 0.0

    def initialise(self):
        self._last_location = CarlaDataProvider.get_location(self.actor)
        super(AverageVelocityTest, self).initialise()

    def update(self):
        """
        Check velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        location = CarlaDataProvider.get_location(self.actor)

        if location is None:
            return new_status

        if self._last_location is None:
            self._last_location = location
            return new_status

        self._distance += location.distance(self._last_location)
        self._last_location = location

        elapsed_time = GameTime.get_time()
        if elapsed_time > 0.0:
            self.actual_value = self._distance / elapsed_time

        if self.actual_value > self.success_value:
            self.test_status = "SUCCESS"
        elif (self.acceptable_value is not None and
              self.actual_value > self.acceptable_value):
            self.test_status = "ACCEPTABLE"
        else:
            self.test_status = "RUNNING"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Set final status
        """
        if self.test_status == "RUNNING":
            self.test_status = "FAILURE"
        super(AverageVelocityTest, self).terminate(new_status)


class CollisionTest(Criterion):

    """
    This class contains an atomic test for collisions.

    Args:
    - actor (carla.Actor): CARLA actor to be used for this test
    - other_actor (carla.Actor): only collisions with this actor will be registered
    - other_actor_type (str): only collisions with actors including this type_id will count.
        Additionally, the "miscellaneous" tag can also be used to include all static objects in the scene
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    COLLISION_RADIUS = 5  # Two collisions that happen within this distance count as one
    MAX_ID_TIME = 5  # Two collisions with the same id that happen within this time count as one
    EPSILON = 0.1  # Collisions at lower this speed won't be counted as the actor's fault

    def __init__(self, actor, other_actor=None, other_actor_type=None,
                 optional=False, terminate_on_failure=False, name="CollisionTest"):
        """
        Construction with sensor setup
        """
        super(CollisionTest, self).__init__(name, actor, optional, terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._other_actor = other_actor
        self._other_actor_type = other_actor_type

        # Attributes to store the last collisions's data
        self._collision_sensor = None
        self._collision_id = None
        self._collision_time = None
        self._collision_location = None

    def initialise(self):
        """
        Creates the sensor and callback"""
        world = CarlaDataProvider.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._collision_sensor.listen(lambda event: self._count_collisions(event))
        super(CollisionTest, self).initialise()

    def update(self):
        """
        Check collision count
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        actor_location = CarlaDataProvider.get_location(self.actor)

        # Check if the last collision can be ignored
        if self._collision_location:
            distance_vector = actor_location - self._collision_location
            if distance_vector.length() > self.COLLISION_RADIUS:
                self._collision_location = None
        if self._collision_id:
            elapsed_time = GameTime.get_time() - self._collision_time
            if elapsed_time > self.MAX_ID_TIME:
                self._collision_id = None

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        if self._collision_sensor is not None:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
        self._collision_sensor = None
        super(CollisionTest, self).terminate(new_status)

    def _count_collisions(self, event):     # pylint: disable=too-many-return-statements
        """Update collision count"""
        actor_location = CarlaDataProvider.get_location(self.actor)

        # Check if the care about the other actor
        if self._other_actor and self._other_actor.id != event.other_actor.id:
            return

        if self._other_actor_type:
            if self._other_actor_type == "miscellaneous":  # Special OpenScenario case
                if "traffic" not in event.other_actor.type_id and "static" not in event.other_actor.type_id:
                    return
            elif self._other_actor_type not in event.other_actor.type_id:
                    return

        # To avoid multiple counts of the same collision, filter some of them.
        if self._collision_id == event.other_actor.id:
            return
        if self._collision_location:
            distance_vector = actor_location - self._collision_location
            if distance_vector.length() <= self.COLLISION_RADIUS:
                return

        # If the actor speed is 0, the collision isn't its fault
        if CarlaDataProvider.get_velocity(self.actor) < self.EPSILON:
            return

        # The collision is valid, save the data
        self.test_status = "FAILURE"
        self.actual_value += 1

        self._collision_time = GameTime.get_time()
        self._collision_location = actor_location
        # Read-only CVCI diagnostics: CARLA's result message preserves the ego
        # contact location but drops the frame/control that identifies which
        # controller phase caused it.  Emit those facts to evaluator stdout;
        # do not alter criterion state, filtering, or scoring.
        try:
            ego_transform = self.actor.get_transform()
            ego_control = self.actor.get_control()
            impulse = event.normal_impulse
            impulse_magnitude = (
                impulse.x * impulse.x
                + impulse.y * impulse.y
                + impulse.z * impulse.z
            ) ** 0.5
            print(
                "CVCI_COLLISION_DIAGNOSTIC "
                "frame={} game_time={:.3f} type={} id={} "
                "ego=({:.3f},{:.3f},{:.3f}) yaw={:.3f} speed={:.3f} "
                "steer={:.4f} throttle={:.4f} brake={:.4f} "
                "reverse={} impulse={:.3f}".format(
                    GameTime.get_frame(), GameTime.get_time(),
                    event.other_actor.type_id, event.other_actor.id,
                    actor_location.x, actor_location.y, actor_location.z,
                    ego_transform.rotation.yaw,
                    CarlaDataProvider.get_velocity(self.actor),
                    ego_control.steer, ego_control.throttle,
                    ego_control.brake, ego_control.reverse,
                    impulse_magnitude,
                ),
                flush=True,
            )
        except Exception as diagnostic_error:  # pragma: no cover - diagnostic only
            print(
                "CVCI_COLLISION_DIAGNOSTIC error={}".format(
                    diagnostic_error
                ),
                flush=True,
            )
        if event.other_actor.id != 0: # Number 0: static objects -> ignore it
            self._collision_id = event.other_actor.id

        if ('static' in event.other_actor.type_id or 'traffic' in event.other_actor.type_id) \
                and 'sidewalk' not in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_STATIC
        elif 'vehicle' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_VEHICLE
        elif 'walker' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_PEDESTRIAN
        else:
            return

        collision_event = TrafficEvent(event_type=actor_type, frame=GameTime.get_frame())
        collision_event.set_dict({'other_actor': event.other_actor, 'location': actor_location})
        collision_event.set_message(
            "Agent collided against object with type={} and id={} at (x={}, y={}, z={})".format(
                event.other_actor.type_id,
                event.other_actor.id,
                round(actor_location.x, 3),
                round(actor_location.y, 3),
                round(actor_location.z, 3)))
        self.events.append(collision_event)


class ActorBlockedTest(Criterion):

    """
    This test will fail if the actor has had its linear velocity lower than a specific value for
    a specific amount of time
    Important parameters:
    - actor: CARLA actor to be used for this test
    - min_speed: speed required [m/s]
    - max_time: Maximum time (in seconds) the actor can remain under the speed threshold
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    def __init__(self, actor, min_speed, max_time, name="ActorBlockedTest", optional=False, terminate_on_failure=False):
        """
        Class constructor
        """
        super().__init__(name, actor, optional, terminate_on_failure)
        self._min_speed = min_speed
        self._max_time = max_time
        self._time_last_valid_state = None
        self._active = True
        self.units = None  # We care about whether or not it fails, no units attached

    def update(self):
        """
        Check if the actor speed is above the min_speed
        """
        new_status = py_trees.common.Status.RUNNING

        # Deactivate/Activate checking by blackboard message
        active = py_trees.blackboard.Blackboard().get('AC_SwitchActorBlockedTest')
        if active is not None:
            self._active = active
            self._time_last_valid_state = GameTime.get_time()
            py_trees.blackboard.Blackboard().set("AC_SwitchActorBlockedTest", None, overwrite=True)

        if self._active:
            linear_speed = CarlaDataProvider.get_velocity(self.actor)
            if linear_speed is not None:
                if linear_speed < self._min_speed and self._time_last_valid_state:
                    if (GameTime.get_time() - self._time_last_valid_state) > self._max_time:
                        # The actor has been "blocked" for too long, save the data
                        self.test_status = "FAILURE"

                        vehicle_location = CarlaDataProvider.get_location(self.actor)
                        event = TrafficEvent(event_type=TrafficEventType.VEHICLE_BLOCKED, frame=GameTime.get_frame())
                        event.set_message('Agent got blocked at (x={}, y={}, z={})'.format(
                            round(vehicle_location.x, 3),
                            round(vehicle_location.y, 3),
                            round(vehicle_location.z, 3))
                        )
                        event.set_dict({'location': vehicle_location})
                        self.events.append(event)
                else:
                    self._time_last_valid_state = GameTime.get_time()

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class KeepLaneTest(Criterion):
    """
    This class contains an atomic test for keeping lane.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, optional=False, name="CheckKeepLane"):
        """
        Construction with sensor setup
        """
        super(KeepLaneTest, self).__init__(name, actor, optional)

        world = self.actor.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self._lane_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._lane_sensor.listen(lambda event: self._count_lane_invasion(event))

    def update(self):
        """
        Check lane invasion count
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actual_value > 0:
            self.test_status = "FAILURE"
        else:
            self.test_status = "SUCCESS"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        if self._lane_sensor is not None:
            self._lane_sensor.destroy()
            self._lane_sensor = None
        super(KeepLaneTest, self).terminate(new_status)

    def _count_lane_invasion(self, event):
        """
        Callback to update lane invasion count
        """
        self.actual_value += 1


class ReachedRegionTest(Criterion):

    """
    This class contains the reached region test
    The test is a success if the actor reaches a specified region

    Important parameters:
    - actor: CARLA actor to be used for this test
    - min_x, max_x, min_y, max_y: Bounding box of the checked region
    """

    def __init__(self, actor, min_x, max_x, min_y, max_y, name="ReachedRegionTest"):
        """
        Setup trigger region (rectangle provided by
        [min_x,min_y] and [max_x,max_y]
        """
        super(ReachedRegionTest, self).__init__(name, actor)
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self.actor)
        if location is None:
            return new_status

        in_region = False
        if self.test_status != "SUCCESS":
            in_region = (location.x > self._min_x and location.x < self._max_x) and (
                location.y > self._min_y and location.y < self._max_y)
            if in_region:
                self.test_status = "SUCCESS"
            else:
                self.test_status = "RUNNING"

        if self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class OffRoadTest(Criterion):

    """
    Atomic containing a test to detect when an actor deviates from the driving lanes. This atomic can
    fail when actor has spent a specific time outside driving lanes (defined by OpenDRIVE). Simplified
    version of OnSidewalkTest, and doesn't relly on waypoints with *Sidewalk* lane types

    Args:
        actor (carla.Actor): CARLA actor to be used for this test
        duration (float): Time spent at sidewalks before the atomic fails.
            If terminate_on_failure isn't active, this is ignored.
        optional (bool): If True, the result is not considered for an overall pass/fail result
            when using the output argument
        terminate_on_failure (bool): If True, the atomic will fail when the duration condition has been met.
    """

    def __init__(self, actor, duration=0, optional=False, terminate_on_failure=False, name="OffRoadTest"):
        """
        Setup of the variables
        """
        super(OffRoadTest, self).__init__(name, actor, optional, terminate_on_failure)

        self._map = CarlaDataProvider.get_map()
        self._offroad = False

        self._duration = duration
        self._prev_time = None
        self._time_offroad = 0

    def update(self):
        """
        First, transforms the actor's current position to its corresponding waypoint. This is
        filtered to only use waypoints of type Driving or Parking. Depending on these results,
        the actor will be considered to be outside (or inside) driving lanes.

        returns:
            py_trees.common.Status.FAILURE: when the actor has spent a given duration outside driving lanes
            py_trees.common.Status.RUNNING: the rest of the time
        """
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self.actor)

        # Get the waypoint at the current location to see if the actor is offroad
        drive_waypoint = self._map.get_waypoint(
            current_location,
            project_to_road=False
        )
        park_waypoint = self._map.get_waypoint(
            current_location,
            project_to_road=False,
            lane_type=carla.LaneType.Parking
        )
        if drive_waypoint or park_waypoint:
            self._offroad = False
        else:
            self._offroad = True

        # Counts the time offroad
        if self._offroad:
            if self._prev_time is None:
                self._prev_time = GameTime.get_time()
            else:
                curr_time = GameTime.get_time()
                self._time_offroad += curr_time - self._prev_time
                self._prev_time = curr_time
        else:
            self._prev_time = None

        if self._time_offroad > self._duration:
            self.test_status = "FAILURE"

        if self._terminate_on_failure and self.test_status == "FAILURE":
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class EndofRoadTest(Criterion):

    """
    Atomic containing a test to detect when an actor has changed to a different road

    Args:
        actor (carla.Actor): CARLA actor to be used for this test
        duration (float): Time spent after ending the road before the atomic fails.
            If terminate_on_failure isn't active, this is ignored.
        optional (bool): If True, the result is not considered for an overall pass/fail result
            when using the output argument
        terminate_on_failure (bool): If True, the atomic will fail when the duration condition has been met.
    """

    def __init__(self, actor, duration=0, optional=False, terminate_on_failure=False, name="EndofRoadTest"):
        """
        Setup of the variables
        """
        super(EndofRoadTest, self).__init__(name, actor, optional, terminate_on_failure)

        self._map = CarlaDataProvider.get_map()
        self._end_of_road = False

        self._duration = duration
        self._start_time = None
        self._time_end_road = 0
        self._road_id = None

    def update(self):
        """
        First, transforms the actor's current position to its corresponding waypoint. Then the road id
        is compared with the initial one and if that's the case, a time is started

        returns:
            py_trees.common.Status.FAILURE: when the actor has spent a given duration outside driving lanes
            py_trees.common.Status.RUNNING: the rest of the time
        """
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self.actor)
        current_waypoint = self._map.get_waypoint(current_location)

        # Get the current road id
        if self._road_id is None:
            self._road_id = current_waypoint.road_id

        else:
            # Wait until the actor has left the road
            if self._road_id != current_waypoint.road_id or self._start_time:

                # Start counting
                if self._start_time is None:
                    self._start_time = GameTime.get_time()
                    return new_status

                curr_time = GameTime.get_time()
                self._time_end_road = curr_time - self._start_time

                if self._time_end_road > self._duration:
                    self.test_status = "FAILURE"
                    self.actual_value += 1
                    return py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class OnSidewalkTest(Criterion):

    """
    Atomic containing a test to detect sidewalk invasions of a specific actor. This atomic can
    fail when actor has spent a specific time outside driving lanes (defined by OpenDRIVE).

    Args:
        actor (carla.Actor): CARLA actor to be used for this test
        duration (float): Time spent at sidewalks before the atomic fails.
            If terminate_on_failure isn't active, this is ignored.
        optional (bool): If True, the result is not considered for an overall pass/fail result
            when using the output argument
        terminate_on_failure (bool): If True, the atomic will fail when the duration condition has been met.
    """

    def __init__(self, actor, duration=0, optional=False, terminate_on_failure=False, name="OnSidewalkTest"):
        """
        Construction with sensor setup
        """
        super(OnSidewalkTest, self).__init__(name, actor, optional, terminate_on_failure)

        self._map = CarlaDataProvider.get_map()
        self._onsidewalk_active = False
        self._outside_lane_active = False

        self._actor_location = self.actor.get_location()
        self._wrong_sidewalk_distance = 0
        self._wrong_outside_lane_distance = 0
        self._sidewalk_start_location = None
        self._outside_lane_start_location = None
        self._duration = duration
        self._prev_time = None
        self._time_outside_lanes = 0

    def update(self):
        """
        First, transforms the actor's current position as well as its four corners to their
        corresponding waypoints. Depending on their lane type, the actor will be considered to be
        outside (or inside) driving lanes.

        returns:
            py_trees.common.Status.FAILURE: when the actor has spent a given duration outside
                driving lanes and terminate_on_failure is active
            py_trees.common.Status.RUNNING: the rest of the time
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and self.test_status == "FAILURE":
            new_status = py_trees.common.Status.FAILURE

        # Some of the vehicle parameters
        current_tra = CarlaDataProvider.get_transform(self.actor)
        current_loc = current_tra.location
        current_wp = self._map.get_waypoint(current_loc, lane_type=carla.LaneType.Any)

        # Case 1) Car center is at a sidewalk
        if current_wp.lane_type == carla.LaneType.Sidewalk:
            if not self._onsidewalk_active:
                self._onsidewalk_active = True
                self._sidewalk_start_location = current_loc

        # Case 2) Not inside allowed zones (Driving and Parking)
        elif current_wp.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking):

            # Get the vertices of the vehicle
            heading_vec = current_tra.get_forward_vector()
            heading_vec.z = 0
            heading_vec = heading_vec / math.sqrt(math.pow(heading_vec.x, 2) + math.pow(heading_vec.y, 2))
            perpendicular_vec = carla.Vector3D(-heading_vec.y, heading_vec.x, 0)

            extent = self.actor.bounding_box.extent
            x_boundary_vector = heading_vec * extent.x
            y_boundary_vector = perpendicular_vec * extent.y

            bbox = [
                current_loc + carla.Location(x_boundary_vector - y_boundary_vector),
                current_loc + carla.Location(x_boundary_vector + y_boundary_vector),
                current_loc + carla.Location(-1 * x_boundary_vector - y_boundary_vector),
                current_loc + carla.Location(-1 * x_boundary_vector + y_boundary_vector)]

            bbox_wp = [
                self._map.get_waypoint(bbox[0], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bbox[1], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bbox[2], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bbox[3], lane_type=carla.LaneType.Any)]

            lane_type_list = [bbox_wp[0].lane_type, bbox_wp[1].lane_type, bbox_wp[2].lane_type, bbox_wp[3].lane_type]

            # Case 2.1) Not quite outside yet
            if bbox_wp[0].lane_type == (carla.LaneType.Driving or carla.LaneType.Parking) \
                or bbox_wp[1].lane_type == (carla.LaneType.Driving or carla.LaneType.Parking) \
                or bbox_wp[2].lane_type == (carla.LaneType.Driving or carla.LaneType.Parking) \
                    or bbox_wp[3].lane_type == (carla.LaneType.Driving or carla.LaneType.Parking):

                self._onsidewalk_active = False
                self._outside_lane_active = False

            # Case 2.2) At the mini Shoulders between Driving and Sidewalk
            elif carla.LaneType.Sidewalk in lane_type_list:
                if not self._onsidewalk_active:
                    self._onsidewalk_active = True
                    self._sidewalk_start_location = current_loc

            else:
                distance_vehicle_wp = current_loc.distance(current_wp.transform.location)

                # Case 2.3) Outside lane
                if distance_vehicle_wp >= current_wp.lane_width / 2:

                    if not self._outside_lane_active:
                        self._outside_lane_active = True
                        self._outside_lane_start_location = current_loc

                # Case 2.4) Very very edge case (but still inside driving lanes)
                else:
                    self._onsidewalk_active = False
                    self._outside_lane_active = False

        # Case 3) Driving and Parking conditions
        else:
            # Check for false positives at junctions
            if current_wp.is_junction:
                distance_vehicle_wp = math.sqrt(
                    math.pow(current_wp.transform.location.x - current_loc.x, 2) +
                    math.pow(current_wp.transform.location.y - current_loc.y, 2))

                if distance_vehicle_wp <= current_wp.lane_width / 2:
                    self._onsidewalk_active = False
                    self._outside_lane_active = False
                # Else, do nothing, the waypoint is too far to consider it a correct position
            else:

                self._onsidewalk_active = False
                self._outside_lane_active = False

        # Counts the time offroad
        if self._onsidewalk_active or self._outside_lane_active:
            if self._prev_time is None:
                self._prev_time = GameTime.get_time()
            else:
                curr_time = GameTime.get_time()
                self._time_outside_lanes += curr_time - self._prev_time
                self._prev_time = curr_time
        else:
            self._prev_time = None

        if self._time_outside_lanes > self._duration:
            self.test_status = "FAILURE"

        # Update the distances
        distance_vector = CarlaDataProvider.get_location(self.actor) - self._actor_location
        distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

        if distance >= 0.02:  # Used to avoid micro-changes adding to considerable sums
            self._actor_location = CarlaDataProvider.get_location(self.actor)

            if self._onsidewalk_active:
                self._wrong_sidewalk_distance += distance
            elif self._outside_lane_active:
                # Only add if car is outside the lane but ISN'T in a junction
                self._wrong_outside_lane_distance += distance

        # Register the sidewalk event
        if not self._onsidewalk_active and self._wrong_sidewalk_distance > 0:

            self.actual_value += 1

            onsidewalk_event = TrafficEvent(event_type=TrafficEventType.ON_SIDEWALK_INFRACTION, frame=GameTime.get_frame())
            self._set_event_message(
                onsidewalk_event, self._sidewalk_start_location, self._wrong_sidewalk_distance)
            self._set_event_dict(
                onsidewalk_event, self._sidewalk_start_location, self._wrong_sidewalk_distance)

            self._onsidewalk_active = False
            self._wrong_sidewalk_distance = 0
            self.events.append(onsidewalk_event)

        # Register the outside of a lane event
        if not self._outside_lane_active and self._wrong_outside_lane_distance > 0:

            self.actual_value += 1

            outsidelane_event = TrafficEvent(event_type=TrafficEventType.OUTSIDE_LANE_INFRACTION, frame=GameTime.get_frame())
            self._set_event_message(
                outsidelane_event, self._outside_lane_start_location, self._wrong_outside_lane_distance)
            self._set_event_dict(
                outsidelane_event, self._outside_lane_start_location, self._wrong_outside_lane_distance)

            self._outside_lane_active = False
            self._wrong_outside_lane_distance = 0
            self.events.append(outsidelane_event)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        If there is currently an event running, it is registered
        """
        # If currently at a sidewalk, register the event
        if self._onsidewalk_active:

            self.actual_value += 1

            onsidewalk_event = TrafficEvent(event_type=TrafficEventType.ON_SIDEWALK_INFRACTION, frame=GameTime.get_frame())
            self._set_event_message(
                onsidewalk_event, self._sidewalk_start_location, self._wrong_sidewalk_distance)
            self._set_event_dict(
                onsidewalk_event, self._sidewalk_start_location, self._wrong_sidewalk_distance)

            self._onsidewalk_active = False
            self._wrong_sidewalk_distance = 0
            self.events.append(onsidewalk_event)

        # If currently outside of our lane, register the event
        if self._outside_lane_active:

            self.actual_value += 1

            outsidelane_event = TrafficEvent(event_type=TrafficEventType.OUTSIDE_LANE_INFRACTION, frame=GameTime.get_frame())
            self._set_event_message(
                outsidelane_event, self._outside_lane_start_location, self._wrong_outside_lane_distance)
            self._set_event_dict(
                outsidelane_event, self._outside_lane_start_location, self._wrong_outside_lane_distance)

            self._outside_lane_active = False
            self._wrong_outside_lane_distance = 0
            self.events.append(outsidelane_event)

        super(OnSidewalkTest, self).terminate(new_status)

    def _set_event_message(self, event, location, distance):
        """
        Sets the message of the event
        """
        if event.get_type() == TrafficEventType.ON_SIDEWALK_INFRACTION:
            message_start = 'Agent invaded the sidewalk'
        else:
            message_start = 'Agent went outside the lane'

        event.set_message(
            '{} for about {} meters, starting at (x={}, y={}, z={})'.format(
                message_start,
                round(distance, 3),
                round(location.x, 3),
                round(location.y, 3),
                round(location.z, 3)))

    def _set_event_dict(self, event, location, distance):
        """
        Sets the dictionary of the event
        """
        event.set_dict({'location': location, 'distance': distance})


class OutsideRouteLanesTest(Criterion):

    """
    Atomic to detect if the vehicle is either on a sidewalk or at a wrong lane. The distance spent outside
    is computed and it is returned as a percentage of the route distance traveled.

    Args:
        actor (carla.ACtor): CARLA actor to be used for this test
        route (list [carla.Location, connection]): series of locations representing the route waypoints
        optional (bool): If True, the result is not considered for an overall pass/fail result
    """

    ALLOWED_OUT_DISTANCE = 0.5  # At least 0.5, due to the mini-shoulder between lanes and sidewalks
    MAX_VEHICLE_ANGLE = 120.0  # Maximum angle between the yaw and waypoint lane
    MAX_WAYPOINT_ANGLE = 150.0  # Maximum change between the yaw-lane angle between frames
    WINDOWS_SIZE = 3  # Amount of additional waypoints checked (in case the first on fails)

    def __init__(self, actor, route, optional=False, name="OutsideRouteLanesTest"):
        """
        Constructor
        """
        super(OutsideRouteLanesTest, self).__init__(name, actor, optional)
        self.units = "%"

        self._route = route
        self._current_index = 0
        self._route_length = len(self._route)
        self._route_transforms, _ = zip(*self._route)

        self._map = CarlaDataProvider.get_map()
        self._last_ego_waypoint = self._map.get_waypoint(self.actor.get_location())

        self._outside_lane_active = False
        self._wrong_lane_active = False
        self._last_road_id = None
        self._last_lane_id = None
        self._total_distance = 0
        self._wrong_distance = 0
        self._wrong_direction_active = True

        self._traffic_event = None

    def update(self):
        """
        Transforms the actor location and its four corners to waypoints. Depending on its types,
        the actor will be considered to be at driving lanes, sidewalk or offroad.

        returns:
            py_trees.common.Status.FAILURE: when the actor has left driving and terminate_on_failure is active
            py_trees.common.Status.RUNNING: the rest of the time
        """
        new_status = py_trees.common.Status.RUNNING

        # Some of the vehicle parameters
        location = CarlaDataProvider.get_location(self.actor)
        if location is None:
            return new_status

        # Deactivate / activate checking by blackboard message
        active = py_trees.blackboard.Blackboard().get('AC_SwitchWrongDirectionTest')
        if active is not None:
            self._wrong_direction_active = active
            py_trees.blackboard.Blackboard().set("AC_SwitchWrongDirectionTest", None, overwrite=True)

        self._is_outside_driving_lanes(location)
        self._is_at_wrong_lane(location)

        if self._outside_lane_active or (self._wrong_direction_active and self._wrong_lane_active):
            self.test_status = "FAILURE"

        # Get the traveled distance
        for index in range(self._current_index + 1,
                           min(self._current_index + self.WINDOWS_SIZE + 1, self._route_length)):
            # Get the dot product to know if it has passed this location
            route_transform = self._route_transforms[index]
            route_location = route_transform.location

            wp_dir = route_transform.get_forward_vector()  # Waypoint's forward vector
            wp_veh = location - route_location  # vector waypoint - vehicle

            if wp_veh.dot(wp_dir) > 0:
                # Get the distance traveled and add it to the total distance
                prev_route_location = self._route_transforms[self._current_index].location
                new_dist = prev_route_location.distance(route_location)
                self._total_distance += new_dist

                # And to the wrong one if outside route lanes
                if self._outside_lane_active or (self._wrong_direction_active and self._wrong_lane_active):
                    self._wrong_distance += new_dist

                if self._wrong_distance:
                    self._set_traffic_event()

                self._current_index = index

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def _set_traffic_event(self):
        """
        Creates the traffic event / updates it
        """
        if self._traffic_event is None:
            self._traffic_event = TrafficEvent(event_type=TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION, frame=GameTime.get_frame())
            self.events.append(self._traffic_event)

        percentage = self._wrong_distance / self._total_distance * 100
        self.actual_value = round(percentage, 2)

        self._traffic_event.set_message(
            "Agent went outside its route lanes for about {} meters "
            "({}% of the completed route)".format(
                round(self._wrong_distance, 3),
                round(percentage, 2)))

        self._traffic_event.set_dict({
            'distance': self._wrong_distance,
            'percentage': percentage
        })

        self._traffic_event.set_frame(GameTime.get_frame())

    def _is_outside_driving_lanes(self, location):
        """
        Detects if the ego_vehicle is outside driving lanes
        """
        driving_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Driving)
        parking_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Parking)

        driving_distance = location.distance(driving_wp.transform.location)
        if parking_wp is not None:  # Some towns have no parking
            parking_distance = location.distance(parking_wp.transform.location)
        else:
            parking_distance = float('inf')

        if driving_distance >= parking_distance:
            distance = parking_distance
            lane_width = parking_wp.lane_width
        else:
            distance = driving_distance
            lane_width = driving_wp.lane_width

        self._outside_lane_active = bool(distance > (lane_width / 2 + self.ALLOWED_OUT_DISTANCE))

    def _is_at_wrong_lane(self, location):
        """
        Detects if the ego_vehicle has invaded a wrong lane
        """
        waypoint = self._map.get_waypoint(location, lane_type=carla.LaneType.Driving)
        lane_id = waypoint.lane_id
        road_id = waypoint.road_id

        # Lanes and roads are too chaotic at junctions
        if waypoint.is_junction:
            self._wrong_lane_active = False
        elif self._last_road_id != road_id or self._last_lane_id != lane_id:

            if self._last_ego_waypoint.is_junction:
                # Just exited a junction, check the wp direction vs the ego's one
                wp_yaw = waypoint.transform.rotation.yaw % 360
                actor_yaw = self.actor.get_transform().rotation.yaw % 360
                angle = (wp_yaw - actor_yaw) % 360

                if angle < self.MAX_VEHICLE_ANGLE or angle > (360 - self.MAX_VEHICLE_ANGLE):
                    self._wrong_lane_active = False
                else:
                    self._wrong_lane_active = True

            else:
                # Route direction can be considered continuous, check for a big gap.
                last_wp_yaw = self._last_ego_waypoint.transform.rotation.yaw % 360
                wp_yaw = waypoint.transform.rotation.yaw % 360
                angle = (last_wp_yaw - wp_yaw) % 360

                if angle > self.MAX_WAYPOINT_ANGLE and angle < (360 - self.MAX_WAYPOINT_ANGLE):

                    # Is the ego vehicle going back to the lane, or going out? Take the opposite
                    self._wrong_lane_active = not bool(self._wrong_lane_active)

        # Remember the last state
        self._last_lane_id = lane_id
        self._last_road_id = road_id
        self._last_ego_waypoint = waypoint


class WrongLaneTest(Criterion):

    """
    This class contains an atomic test to detect invasions to wrong direction lanes.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """
    MAX_ALLOWED_ANGLE = 120.0
    MAX_WAYPOINT_ANGLE = 150.0

    def __init__(self, actor, optional=False, name="WrongLaneTest"):
        """
        Construction with sensor setup
        """
        super(WrongLaneTest, self).__init__(name, actor, optional)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._map = CarlaDataProvider.get_map()
        self._last_lane_id = None
        self._last_road_id = None

        self._in_lane = True
        self._wrong_distance = 0
        self._actor_location = self.actor.get_location()
        self._previous_lane_waypoint = self._map.get_waypoint(self.actor.get_location())
        self._wrong_lane_start_location = None

    def update(self):
        """
        Check lane invasion count
        """

        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        lane_waypoint = self._map.get_waypoint(self.actor.get_location())
        current_lane_id = lane_waypoint.lane_id
        current_road_id = lane_waypoint.road_id

        if (self._last_road_id != current_road_id or self._last_lane_id != current_lane_id) \
                and not lane_waypoint.is_junction:
            next_waypoint = lane_waypoint.next(2.0)[0]

            if not next_waypoint:
                return new_status

            # The waypoint route direction can be considered continuous.
            # Therefore just check for a big gap in waypoint directions.
            previous_lane_direction = self._previous_lane_waypoint.transform.get_forward_vector()
            current_lane_direction = lane_waypoint.transform.get_forward_vector()

            p_lane_vector = np.array([previous_lane_direction.x, previous_lane_direction.y])
            c_lane_vector = np.array([current_lane_direction.x, current_lane_direction.y])

            waypoint_angle = math.degrees(
                math.acos(np.clip(np.dot(p_lane_vector, c_lane_vector) /
                                  (np.linalg.norm(p_lane_vector) * np.linalg.norm(c_lane_vector)), -1.0, 1.0)))

            if waypoint_angle > self.MAX_WAYPOINT_ANGLE and self._in_lane:

                self.test_status = "FAILURE"
                self._in_lane = False
                self.actual_value += 1
                self._wrong_lane_start_location = self._actor_location

            else:
                # Reset variables
                self._in_lane = True

            # Continuity is broken after a junction so check vehicle-lane angle instead
            if self._previous_lane_waypoint.is_junction:

                vector_wp = np.array([next_waypoint.transform.location.x - lane_waypoint.transform.location.x,
                                      next_waypoint.transform.location.y - lane_waypoint.transform.location.y])

                vector_actor = np.array([math.cos(math.radians(self.actor.get_transform().rotation.yaw)),
                                         math.sin(math.radians(self.actor.get_transform().rotation.yaw))])

                vehicle_lane_angle = math.degrees(
                    math.acos(np.clip(np.dot(vector_actor, vector_wp) / (np.linalg.norm(vector_wp)), -1.0, 1.0)))

                if vehicle_lane_angle > self.MAX_ALLOWED_ANGLE:

                    self.test_status = "FAILURE"
                    self._in_lane = False
                    self.actual_value += 1
                    self._wrong_lane_start_location = self.actor.get_location()

        # Keep adding "meters" to the counter
        distance_vector = self.actor.get_location() - self._actor_location
        distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

        if distance >= 0.02:  # Used to avoid micro-changes adding add to considerable sums
            self._actor_location = CarlaDataProvider.get_location(self.actor)

            if not self._in_lane and not lane_waypoint.is_junction:
                self._wrong_distance += distance

        # Register the event
        if self._in_lane and self._wrong_distance > 0:

            wrong_way_event = TrafficEvent(event_type=TrafficEventType.WRONG_WAY_INFRACTION, frame=GameTime.get_frame())
            self._set_event_message(wrong_way_event, self._wrong_lane_start_location,
                                    self._wrong_distance, current_road_id, current_lane_id)
            self._set_event_dict(wrong_way_event, self._wrong_lane_start_location,
                                 self._wrong_distance, current_road_id, current_lane_id)

            self.events.append(wrong_way_event)
            self._wrong_distance = 0

        # Remember the last state
        self._last_lane_id = current_lane_id
        self._last_road_id = current_road_id
        self._previous_lane_waypoint = lane_waypoint

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        If there is currently an event running, it is registered
        """
        if not self._in_lane:

            lane_waypoint = self._map.get_waypoint(self.actor.get_location())
            current_lane_id = lane_waypoint.lane_id
            current_road_id = lane_waypoint.road_id

            wrong_way_event = TrafficEvent(event_type=TrafficEventType.WRONG_WAY_INFRACTION, frame=GameTime.get_frame())
            self._set_event_message(wrong_way_event, self._wrong_lane_start_location,
                                    self._wrong_distance, current_road_id, current_lane_id)
            self._set_event_dict(wrong_way_event, self._wrong_lane_start_location,
                                 self._wrong_distance, current_road_id, current_lane_id)

            self._wrong_distance = 0
            self._in_lane = True
            self.events.append(wrong_way_event)

        super(WrongLaneTest, self).terminate(new_status)

    def _set_event_message(self, event, location, distance, road_id, lane_id):
        """
        Sets the message of the event
        """

        event.set_message(
            "Agent invaded a lane in opposite direction for {} meters, starting at (x={}, y={}, z={}). "
            "road_id={}, lane_id={}".format(
                round(distance, 3),
                round(location.x, 3),
                round(location.y, 3),
                round(location.z, 3),
                road_id,
                lane_id))

    def _set_event_dict(self, event, location, distance, road_id, lane_id):
        """
        Sets the dictionary of the event
        """
        event.set_dict({
            'location': location,
            'distance': distance,
            'road_id': road_id,
            'lane_id': lane_id})


class InRadiusRegionTest(Criterion):

    """
    The test is a success if the actor is within a given radius of a specified region

    Important parameters:
    - actor: CARLA actor to be used for this test
    - x, y, radius: Position (x,y) and radius (in meters) used to get the checked region
    """

    def __init__(self, actor, x, y, radius, name="InRadiusRegionTest"):
        """
        """
        super(InRadiusRegionTest, self).__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._x = x     # pylint: disable=invalid-name
        self._y = y     # pylint: disable=invalid-name
        self._radius = radius

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self.actor)
        if location is None:
            return new_status

        if self.test_status != "SUCCESS":
            in_radius = math.sqrt(((location.x - self._x)**2) + ((location.y - self._y)**2)) < self._radius
            if in_radius:
                route_completion_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETED, frame=GameTime.get_frame())
                route_completion_event.set_message("Destination was successfully reached")
                self.events.append(route_completion_event)
                self.test_status = "SUCCESS"
            else:
                self.test_status = "RUNNING"

        if self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InRouteTest(Criterion):

    """
    The test is a success if the actor is never outside route. The actor can go outside of the route
    but only for a certain amount of distance

    Important parameters:
    - actor: CARLA actor to be used for this test
    - route: Route to be checked
    - offroad_max: Maximum distance (in meters) the actor can deviate from the route
    - offroad_min: Maximum safe distance (in meters). Might eventually cause failure
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    MAX_ROUTE_PERCENTAGE = 30  # %
    WINDOWS_SIZE = 5  # Amount of additional waypoints checked

    def __init__(self, actor, route, offroad_min=None, offroad_max=30, name="InRouteTest", terminate_on_failure=False):
        """
        """
        super(InRouteTest, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.units = None  # We care about whether or not it fails, no units attached

        self._route = route
        self._offroad_max = offroad_max
        # Unless specified, halve of the max value
        if offroad_min is None:
            self._offroad_min = self._offroad_max / 2
        else:
            self._offroad_min = self._offroad_min

        self._world = CarlaDataProvider.get_world()
        self._route_transforms, _ = zip(*self._route)
        self._route_length = len(self._route)
        self._current_index = 0
        self._out_route_distance = 0
        self._in_safe_route = True

        self._accum_meters = []
        prev_loc = self._route_transforms[0].location
        for i, tran in enumerate(self._route_transforms):
            loc = tran.location
            d = loc.distance(prev_loc)
            accum = 0 if i == 0 else self._accum_meters[i - 1]

            self._accum_meters.append(d + accum)
            prev_loc = loc

        # Blackboard variable
        blackv = py_trees.blackboard.Blackboard()
        _ = blackv.set("InRoute", True)

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self.actor)
        if location is None:
            return new_status

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        elif self.test_status in ('RUNNING', 'INIT'):

            off_route = True

            shortest_distance = float('inf')
            closest_index = -1

            # Get the closest distance
            for index in range(self._current_index,
                               min(self._current_index + self.WINDOWS_SIZE + 1, self._route_length)):
                ref_location = self._route_transforms[index].location
                distance = math.sqrt(((location.x - ref_location.x) ** 2) + ((location.y - ref_location.y) ** 2))
                if distance <= shortest_distance:
                    closest_index = index
                    shortest_distance = distance

            if closest_index == -1 or shortest_distance == float('inf'):
                return new_status

            # Check if the actor is out of route
            if shortest_distance < self._offroad_max:
                off_route = False
                self._in_safe_route = bool(shortest_distance < self._offroad_min)

            # If actor advanced a step, record the distance
            if self._current_index != closest_index:

                new_dist = self._accum_meters[closest_index] - self._accum_meters[self._current_index]

                # If too far from the route, add it and check if its value
                if not self._in_safe_route:
                    self._out_route_distance += new_dist
                    out_route_percentage = 100 * self._out_route_distance / self._accum_meters[-1]
                    if out_route_percentage > self.MAX_ROUTE_PERCENTAGE:
                        off_route = True

                self._current_index = closest_index

            if off_route:
                # Blackboard variable
                blackv = py_trees.blackboard.Blackboard()
                _ = blackv.set("InRoute", False)

                route_deviation_event = TrafficEvent(event_type=TrafficEventType.ROUTE_DEVIATION, frame=GameTime.get_frame())
                route_deviation_event.set_message(
                    "Agent deviated from the route at (x={}, y={}, z={})".format(
                        round(location.x, 3),
                        round(location.y, 3),
                        round(location.z, 3)))
                route_deviation_event.set_dict({'location': location})

                self.events.append(route_deviation_event)

                self.test_status = "FAILURE"
                self.actual_value += 1
                new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class RouteCompletionTest(Criterion):

    """
    Check at which stage of the route is the actor at each tick

    Important parameters:
    - actor: CARLA actor to be used for this test
    - route: Route to be checked
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    WINDOWS_SIZE = 2

    # Thresholds to return that a route has been completed
    DISTANCE_THRESHOLD = 10.0  # meters
    PERCENTAGE_THRESHOLD = 99  # %

    def __init__(self, actor, route, name="RouteCompletionTest", terminate_on_failure=False):
        """
        """
        super(RouteCompletionTest, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.units = "%"
        self.success_value = 100
        self._route = route
        self._map = CarlaDataProvider.get_map()

        self._index = 0
        self._route_length = len(self._route)
        self._route_transforms, _ = zip(*self._route)
        self._route_accum_perc = self._get_acummulated_percentages()

        self.target_location = self._route_transforms[-1].location

        self._traffic_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETION, frame=0)
        self._traffic_event.set_dict({'route_completed': self.actual_value})
        self._traffic_event.set_message("Agent has completed {} of the route".format(self.actual_value))
        self.events.append(self._traffic_event)

    def _get_acummulated_percentages(self):
        """Gets the accumulated percentage of each of the route transforms"""
        accum_meters = []
        prev_loc = self._route_transforms[0].location
        for i, tran in enumerate(self._route_transforms):
            d = tran.location.distance(prev_loc)
            new_d = 0 if i == 0 else accum_meters[i - 1]

            accum_meters.append(d + new_d)
            prev_loc = tran.location

        max_dist = accum_meters[-1]
        return [x / max_dist * 100 for x in accum_meters]

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self.actor)
        if location is None:
            return new_status

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        elif self.test_status in ('RUNNING', 'INIT'):

            for index in range(self._index, min(self._index + self.WINDOWS_SIZE + 1, self._route_length)):
                # Get the dot product to know if it has passed this location
                route_transform = self._route_transforms[index]
                route_location = route_transform.location
                wp_dir = route_transform.get_forward_vector()          # Waypoint's forward vector
                wp_veh = location - route_location                     # vector route - vehicle

                if wp_veh.dot(wp_dir) > 0:
                    self._index = index
                    self.actual_value = self._route_accum_perc[self._index]

            self.actual_value = round(self.actual_value, 2)
            self._traffic_event.set_dict({'route_completed': self.actual_value})
            self._traffic_event.set_message("Agent has completed {} of the route".format(self.actual_value))

            if self.actual_value > self.PERCENTAGE_THRESHOLD \
                    and location.distance(self.target_location) < self.DISTANCE_THRESHOLD:
                self.test_status = "SUCCESS"
                self.actual_value = 100

        elif self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Set test status to failure if not successful and terminate
        """
        self.actual_value = round(self.actual_value, 2)

        self._traffic_event.set_dict({'route_completed': self.actual_value})
        self._traffic_event.set_message("Agent has completed {} of the route".format(self.actual_value))

        if self.test_status == "INIT":
            self.test_status = "FAILURE"
        super(RouteCompletionTest, self).terminate(new_status)


class RunningRedLightTest(Criterion):

    """
    Check if an actor is running a red light

    Important parameters:
    - actor: CARLA actor to be used for this test
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    DISTANCE_LIGHT = 15  # m

    def __init__(self, actor, name="RunningRedLightTest", terminate_on_failure=False):
        """
        Init
        """
        super(RunningRedLightTest, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self._world = CarlaDataProvider.get_world()
        self._map = CarlaDataProvider.get_map()
        self._list_traffic_lights = []
        self._last_red_light_id = None
        self.debug = False

        all_actors = CarlaDataProvider.get_all_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                center, waypoints = self.get_traffic_light_waypoints(_actor)
                self._list_traffic_lights.append((_actor, center, waypoints))

    # pylint: disable=no-self-use
    def is_vehicle_crossing_line(self, seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = shapely.geometry.LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = shapely.geometry.LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)

        return not inter.is_empty

    def update(self):
        """
        Check if the actor is running a red light
        """
        new_status = py_trees.common.Status.RUNNING

        transform = CarlaDataProvider.get_transform(self.actor)
        location = transform.location
        if location is None:
            return new_status

        veh_extent = self.actor.bounding_box.extent.x

        tail_close_pt = self.rotate_point(carla.Vector3D(-0.8 * veh_extent, 0, 0), transform.rotation.yaw)
        tail_close_pt = location + carla.Location(tail_close_pt)

        tail_far_pt = self.rotate_point(carla.Vector3D(-veh_extent - 1, 0, 0), transform.rotation.yaw)
        tail_far_pt = location + carla.Location(tail_far_pt)

        for traffic_light, center, waypoints in self._list_traffic_lights:

            if self.debug:
                z = 2.1
                if traffic_light.state == carla.TrafficLightState.Red:
                    color = carla.Color(155, 0, 0)
                elif traffic_light.state == carla.TrafficLightState.Green:
                    color = carla.Color(0, 155, 0)
                else:
                    color = carla.Color(155, 155, 0)
                self._world.debug.draw_point(center + carla.Location(z=z), size=0.2, color=color, life_time=0.01)
                for wp in waypoints:
                    text = "{}.{}".format(wp.road_id, wp.lane_id)
                    self._world.debug.draw_string(
                        wp.transform.location + carla.Location(x=1, z=z), text, color=color, life_time=0.01)
                    self._world.debug.draw_point(
                        wp.transform.location + carla.Location(z=z), size=0.1, color=color, life_time=0.01)

            center_loc = carla.Location(center)

            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            for wp in waypoints:

                tail_wp = self._map.get_waypoint(tail_far_pt)

                # Calculate the dot product (Might be unscaled, as only its sign is important)
                ve_dir = CarlaDataProvider.get_transform(self.actor).get_forward_vector()
                wp_dir = wp.transform.get_forward_vector()

                # Check the lane until all the "tail" has passed
                if tail_wp.road_id == wp.road_id and tail_wp.lane_id == wp.lane_id and ve_dir.dot(wp_dir) > 0:
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location

                    lft_lane_wp = self.rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp + 90)
                    lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
                    rgt_lane_wp = self.rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp - 90)
                    rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)

                    # Is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (lft_lane_wp, rgt_lane_wp)):

                        self.test_status = "FAILURE"
                        self.actual_value += 1
                        location = traffic_light.get_transform().location
                        red_light_event = TrafficEvent(event_type=TrafficEventType.TRAFFIC_LIGHT_INFRACTION, frame=GameTime.get_frame())
                        red_light_event.set_message(
                            "Agent ran a red light {} at (x={}, y={}, z={})".format(
                                traffic_light.id,
                                round(location.x, 3),
                                round(location.y, 3),
                                round(location.z, 3)))
                        red_light_event.set_dict({'id': traffic_light.id, 'location': location})

                        self.events.append(red_light_event)
                        self._last_red_light_id = traffic_light.id
                        break

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    def get_traffic_light_waypoints(self, traffic_light):
        """
        get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps


class RunningStopTest(Criterion):

    """
    Check if an actor is running a stop sign

    Important parameters:
    - actor: CARLA actor to be used for this test
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    PROXIMITY_THRESHOLD = 4.0  # Stops closer than this distance will be detected [m]
    SPEED_THRESHOLD = 0.1 # Minimum speed to consider the actor has stopped [m/s]
    WAYPOINT_STEP = 0.5  # m

    def __init__(self, actor, name="RunningStopTest", terminate_on_failure=False):
        """
        """
        super(RunningStopTest, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self._world = CarlaDataProvider.get_world()
        self._map = CarlaDataProvider.get_map()
        self._list_stop_signs = []
        self._target_stop_sign = None
        self._stop_completed = False

        self._last_failed_stop = None

        for _actor in CarlaDataProvider.get_all_actors():
            if 'traffic.stop' in _actor.type_id:
                self._list_stop_signs.append(_actor)

    def point_inside_boundingbox(self, point, bb_center, bb_extent, multiplier=1.2):
        """Checks whether or not a point is inside a bounding box."""

        # pylint: disable=invalid-name
        A = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
        B = carla.Vector2D(bb_center.x + multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
        D = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y + multiplier * bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad  # pylint: disable=chained-comparison

    def is_actor_affected_by_stop(self, wp_list, stop):
        """
        Check if the given actor is affected by the stop.
        Without using waypoints, a stop might not be detected if the actor is moving at the lane edge.
        """
        # Quick distance test
        stop_location = stop.get_transform().transform(stop.trigger_volume.location)
        actor_location = wp_list[0].transform.location
        if stop_location.distance(actor_location) > self.PROXIMITY_THRESHOLD:
            return False

        # Check if the any of the actor wps is inside the stop's bounding box.
        # Using more than one waypoint removes issues with small trigger volumes and backwards movement
        stop_extent = stop.trigger_volume.extent
        for actor_wp in wp_list:
            if self.point_inside_boundingbox(actor_wp.transform.location, stop_location, stop_extent):
                return True

        return False

    def _scan_for_stop_sign(self, actor_transform, wp_list):
        """
        Check the stop signs to see if any of them affect the actor.
        Ignore all checks when going backwards or through an opposite direction"""

        actor_direction = actor_transform.get_forward_vector()

        # Ignore all when going backwards
        actor_velocity = self.actor.get_velocity()
        if actor_velocity.dot(actor_direction) < -0.17:  # 100º, just in case
            return None

        # Ignore all when going in the opposite direction
        lane_direction = wp_list[0].transform.get_forward_vector()
        if actor_direction.dot(lane_direction) < -0.17:  # 100º, just in case
            return None

        for stop in self._list_stop_signs:
            if self.is_actor_affected_by_stop(wp_list, stop):
                return stop

    def _get_waypoints(self, actor):
        """Returns a list of waypoints starting from the ego location and a set amount forward"""
        wp_list = []
        steps = int(self.PROXIMITY_THRESHOLD / self.WAYPOINT_STEP)

        # Add the actor location
        wp = self._map.get_waypoint(actor.get_location())
        wp_list.append(wp)

        # And its forward waypoints
        next_wp = wp
        for _ in range(steps):
            next_wps = next_wp.next(self.WAYPOINT_STEP)
            if not next_wps:
                break
            next_wp = next_wps[0]
            wp_list.append(next_wp)

        return wp_list

    def update(self):
        """
        Check if the actor is running a red light
        """
        new_status = py_trees.common.Status.RUNNING

        actor_transform = self.actor.get_transform()
        check_wps = self._get_waypoints(self.actor)

        if not self._target_stop_sign:
            self._target_stop_sign = self._scan_for_stop_sign(actor_transform, check_wps)
            return new_status

        if not self._stop_completed:
            current_speed = CarlaDataProvider.get_velocity(self.actor)
            if current_speed < self.SPEED_THRESHOLD:
                self._stop_completed = True

        if not self.is_actor_affected_by_stop(check_wps, self._target_stop_sign):
            if not self._stop_completed and self._last_failed_stop != self._target_stop_sign.id:
                # did we stop?
                self.actual_value += 1
                self.test_status = "FAILURE"
                stop_location = self._target_stop_sign.get_transform().location
                running_stop_event = TrafficEvent(event_type=TrafficEventType.STOP_INFRACTION, frame=GameTime.get_frame())
                running_stop_event.set_message(
                    "Agent ran a stop with id={} at (x={}, y={}, z={})".format(
                        self._target_stop_sign.id,
                        round(stop_location.x, 3),
                        round(stop_location.y, 3),
                        round(stop_location.z, 3)))
                running_stop_event.set_dict({'id': self._target_stop_sign.id, 'location': stop_location})

                self.events.append(running_stop_event)

                self._last_failed_stop = self._target_stop_sign.id

            # Reset state
            self._target_stop_sign = None
            self._stop_completed = False

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class MinimumSpeedRouteTest(Criterion):

    """
    Check at which stage of the route is the actor at each tick

    Important parameters:
    - actor: CARLA actor to be used for this test
    - route: Route to be checked
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    WINDOWS_SIZE = 2
    RATIO = 1

    def __init__(self, actor, route, checkpoints=1, name="MinimumSpeedRouteTest", terminate_on_failure=False):
        """
        """
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.units = "%"
        self.success_value = 100
        self.actual_value = 100

        self._route = route
        self._accum_dist = []
        prev_trans = None
        for trans, _ in route:
            if prev_trans:
                dist = trans.location.distance(prev_trans.location)
                self._accum_dist.append(dist + self._accum_dist[-1])
            else:
                self._accum_dist.append(0)
            prev_trans = trans
        self._route_length = len(self._route)

        self._checkpoints = checkpoints
        self._checkpoint_dist = self._accum_dist[-1] / self._checkpoints

        self._mean_speed = 0
        self._actor_speed = 0
        self._speed_points = 0

        self._current_dist = 0
        self._checkpoint_values = []

        self._index = 0


    def update(self):
        """
        Check if the actor location is within trigger region
        """
        
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        # Check the actor progress through the route 
        location = CarlaDataProvider.get_location(self.actor)
        if location is None:
            return new_status

        for index in range(self._index, min(self._index + self.WINDOWS_SIZE + 1, self._route_length)):
            # Get the dot product to know if it has passed this location
            route_transform = self._route[index][0]
            route_location = route_transform.location
            wp_dir = route_transform.get_forward_vector()
            wp_veh = location - route_location

            if wp_veh.dot(wp_dir) > 0:
                self._index = index

        if self._accum_dist[self._index] - self._current_dist > self._checkpoint_dist:
            self._set_traffic_event()
            self._current_dist = self._accum_dist[self._index]
            self._mean_speed = 0
            self._actor_speed = 0
            self._speed_points = 0

        # Get the actor speed
        velocity = CarlaDataProvider.get_velocity(self.actor)
        if velocity is None:
            return new_status

        # Get the speed of the surrounding Background Activity
        all_vehicles = CarlaDataProvider.get_all_actors().filter('vehicle*')
        background_vehicles = [v for v in all_vehicles if v.attributes['role_name'] == 'background']

        if background_vehicles:
            frame_mean_speed = 0
            for vehicle in background_vehicles:
                frame_mean_speed += CarlaDataProvider.get_velocity(vehicle)
            frame_mean_speed /= len(background_vehicles)

            # Record the data
            self._mean_speed += frame_mean_speed
            self._actor_speed += velocity
            self._speed_points += 1

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def _set_traffic_event(self):

        if self._speed_points > 0 and self._mean_speed:
            self._mean_speed /= self._speed_points
            self._actor_speed /= self._speed_points
            checkpoint_value = round(self._actor_speed / (self.RATIO * self._mean_speed) * 100, 2)
        else:
            checkpoint_value = 100
        
        self.test_status = "FAILURE"
        self._traffic_event = TrafficEvent(TrafficEventType.MIN_SPEED_INFRACTION, GameTime.get_frame())
        self._traffic_event.set_dict({'percentage': checkpoint_value})
        self._traffic_event.set_message(f"Average speed is {checkpoint_value}% of the surrounding traffic's one")
        self.events.append(self._traffic_event)

        self._checkpoint_values.append(checkpoint_value)

    def terminate(self, new_status):
        """
        Set the actual value as a percentage of the two mean speeds,
        the test status to failure if not successful and terminate
        """
        # Routes end at around 99%, so make sure the last checkpoint is recorded
        if self._accum_dist[self._index] / self._accum_dist[-1] > 0.95:
            self._set_traffic_event()

        if len(self._checkpoint_values):
            self.actual_value = round(sum(self._checkpoint_values) / len(self._checkpoint_values), 2)
        super().terminate(new_status)


class YieldToEmergencyVehicleTest(Criterion):

    """
    Atomic Criterion to detect if the actor yields its lane to the emergency vehicle behind it.
    Detection is done by checking if the ev is in front of the actor

    Args:
        actor (carla.Actor): CARLA actor to be used for this test
        ev (carla.Actor): The emergency vehicle
        optional (bool): If True, the result is not considered for an overall pass/fail result
    """

    WAITING_TIME_THRESHOLD = 15 # Maximum time for actor to block ev

    def __init__(self, actor, ev, optional=False, name="YieldToEmergencyVehicleTest"):
        """
        Constructor
        """
        self._ev = ev
        self._terminated = False
        self._diagnostic_updates = 0
        super().__init__(name, actor, optional)

    def update(self):
        """
        Monitor that the EV ends up in front of the actor

        returns:
            py_trees.common.Status.RUNNING
        """
        new_status = py_trees.common.Status.RUNNING

        actor_location = CarlaDataProvider.get_location(self.actor)
        if not actor_location:
            return new_status
        ev_location = CarlaDataProvider.get_location(self._ev)
        if not ev_location:
            return new_status

        ev_direction = CarlaDataProvider.get_transform(self._ev).get_forward_vector()
        actor_ev_vector = actor_location - ev_location
        relation_dot = ev_direction.dot(actor_ev_vector)

        self._diagnostic_updates += 1
        if (
            os.environ.get("CVCI_YIELD_CRITERION_TRACE", "0").strip().lower()
            in ("1", "true", "yes", "on")
            and self._diagnostic_updates % 10 == 1
        ):
            try:
                ego_speed = get_speed(self.actor)
            except RuntimeError:
                ego_speed = float("nan")
            try:
                ev_speed = get_speed(self._ev)
            except RuntimeError:
                ev_speed = float("nan")
            print(
                "CVCI_YIELD_CRITERION_TRACE frame=%s game_time=%.3f "
                "dot=%.6f ego=(%.3f,%.3f,%.3f) ev=(%.3f,%.3f,%.3f) "
                "ego_speed=%.3f ev_speed=%.3f"
                % (
                    GameTime.get_frame(), GameTime.get_time(), relation_dot,
                    actor_location.x, actor_location.y, actor_location.z,
                    ev_location.x, ev_location.y, ev_location.z,
                    ego_speed, ev_speed,
                ),
                flush=True,
            )

        if relation_dot > 0:
            self.test_status = "FAILURE"
        else:
            self.test_status = "SUCCESS"

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """Set the traffic event, if needed"""
        # Terminates are called multiple times. Do this only once
        if not self._terminated:
            if self.test_status == "FAILURE":
                traffic_event = TrafficEvent(TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE, GameTime.get_frame())
                traffic_event.set_message("Agent failed to yield to an emergency vehicle")
                self.events.append(traffic_event)

            self._terminated = True

        super().terminate(new_status)


class ScenarioTimeoutTest(Criterion):

    """
    Atomic Criterion to detect if the actor has been incapable of finishing an scenario

    Args:
        actor (carla.Actor): CARLA actor to be used for this test
        optional (bool): If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, scenario_name, optional=False, name="ScenarioTimeoutTest"):
        """
        Constructor
        """
        super().__init__(name, actor, optional)
        self.success_value = 0
        self.actual_value = 0
        self._scenario_name = scenario_name

    def update(self):
        """wait"""
        new_status = py_trees.common.Status.RUNNING
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """check the blackboard for the data and update the criteria if one found"""

        blackboard_name = f"ScenarioTimeout_{self._scenario_name}"

        timeout = py_trees.blackboard.Blackboard().get(blackboard_name)
        if timeout:
            self.actual_value = 1
            self.test_status = "FAILURE"

            traffic_event = TrafficEvent(event_type=TrafficEventType.SCENARIO_TIMEOUT, frame=GameTime.get_frame())
            traffic_event.set_message("Agent timed out a scenario")
            self.events.append(traffic_event)
        py_trees.blackboard.Blackboard().set(blackboard_name, None, True)

        super().terminate(new_status)

#--------------------新增Criterion类---------------------------
# minTTC criterion
class MinTTCAutoCriterion(Criterion):

    def __init__(self,
                 actor,
                 other_actors,
                 name="MinTTCAutoCriterion",
                 distance_threshold=50.0,
                 forward_angle_deg=120.0,
                 optional=False,
                 terminate_on_failure=False):
        super(MinTTCAutoCriterion, self).__init__(
            name=name,
            actor=actor,
            optional=optional,
            terminate_on_failure=terminate_on_failure
        )

        self.other_actors = other_actors if other_actors is not None else []
        self.distance_threshold = distance_threshold
        self.forward_angle_deg = forward_angle_deg

        self.actual_value = float("inf")
        self.units = "s"
        self.closest_actor = None
        self.current_target = None
        self.current_ttc = float("inf")

    def initialise(self):
        super(MinTTCAutoCriterion, self).initialise()
        self.actual_value = float("inf")
        self.closest_actor = None
        self.current_target = None
        self.current_ttc = float("inf")
        self.test_status = "RUNNING"

    @staticmethod
    def _dot(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

    @staticmethod
    def _norm(v):
        return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    @staticmethod
    def _get_location(actor):
        loc = actor.get_transform().location
        return (loc.x, loc.y, loc.z)

    @staticmethod
    def _get_velocity(actor):
        vel = actor.get_velocity()
        return (vel.x, vel.y, vel.z)

    @staticmethod
    def _get_forward_vector(actor):
        fv = actor.get_transform().get_forward_vector()
        return (fv.x, fv.y, fv.z)

    @staticmethod
    def _get_actor_extent(actor):
        """获取 actor 的边界框半长和半宽 (x, y)"""
        if hasattr(actor, 'bounding_box'):
            return (actor.bounding_box.extent.x, actor.bounding_box.extent.y)
        # 如果获取不到（例如某些特殊行人），使用默认标准轿车尺寸的一半
        return (2.4, 1.0)

    def _is_candidate_valid(self, candidate):
        """初筛候选目标：剔除距离太远或明显在身后的目标"""
        if candidate is None or candidate.id == self.actor.id or not candidate.is_alive:
            return False

        ego_pos = self._get_location(self.actor)
        tgt_pos = self._get_location(candidate)
        rel_pos = (tgt_pos[0] - ego_pos[0], tgt_pos[1] - ego_pos[1], tgt_pos[2] - ego_pos[2])

        dist = self._norm(rel_pos)
        if dist < 1e-6 or dist > self.distance_threshold:
            return False

        # 粗略的角度过滤，保留前方 forward_angle_deg 视野内的目标
        ego_forward = self._get_forward_vector(self.actor)
        ego_forward_norm = self._norm(ego_forward)
        if ego_forward_norm < 1e-6:
            return True

        rel_dir = (rel_pos[0] / dist, rel_pos[1] / dist, rel_pos[2] / dist)
        cos_theta = self._dot(ego_forward, rel_dir) / ego_forward_norm
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta_deg = math.degrees(math.acos(cos_theta))

        if theta_deg > self.forward_angle_deg / 2.0:
            return False

        return True

    def _compute_pair_ttc(self, target_actor):
        """
        计算纵向投影 TTC（保险杠到保险杠）
        """
        ego_pos = self._get_location(self.actor)
        tgt_pos = self._get_location(target_actor)
        ego_forward = self._get_forward_vector(self.actor)

        # 相对位置
        rel_pos = (tgt_pos[0] - ego_pos[0], tgt_pos[1] - ego_pos[1], tgt_pos[2] - ego_pos[2])

        # 1. 纵向距离：将相对位置投影到自车的正前方向量上
        lon_dist = self._dot(rel_pos, ego_forward)

        # 如果目标中心点在自车后方，TTC 无效
        if lon_dist <= 0:
            return float("inf")

        # 2. 横向距离检测（过滤掉在隔壁车道但纵向距离很近的车）
        lon_proj = (ego_forward[0] * lon_dist, ego_forward[1] * lon_dist, ego_forward[2] * lon_dist)
        lat_vec = (rel_pos[0] - lon_proj[0], rel_pos[1] - lon_proj[1], rel_pos[2] - lon_proj[2])
        lat_dist = self._norm(lat_vec)

        ego_extent = self._get_actor_extent(self.actor)
        tgt_extent = self._get_actor_extent(target_actor)

        # 如果横向距离大于两车半宽之和外加安全余量(0.5m)，说明不在同一轨迹，不会相撞
        lat_margin = 0.5
        if lat_dist > (ego_extent[1] + tgt_extent[1] + lat_margin):
            return float("inf")

        # 3. 计算“保险杠到保险杠”的纵向距离
        # 用中心纵向距离减去两车的半长
        bumper_dist = max(0.0, lon_dist - ego_extent[0] - tgt_extent[0])

        # 4. 计算相对速度投影到自车前方的“接近率” (Closing speed)
        ego_vel = self._get_velocity(self.actor)
        tgt_vel = self._get_velocity(target_actor)
        rel_vel = (tgt_vel[0] - ego_vel[0], tgt_vel[1] - ego_vel[1], tgt_vel[2] - ego_vel[2])

        # 只有相对速度投影在 ego_forward 上的方向与前方相反，才说明在靠近
        closing_speed = -self._dot(rel_vel, ego_forward)

        # 如果前车比我们快，或者速度相同，则不会碰撞
        if closing_speed <= 1e-6:
            return float("inf")

        return bumper_dist / closing_speed

    def _find_most_dangerous_target(self):
        best_actor = None
        best_ttc = float("inf")

        for candidate in self.other_actors:
            if not self._is_candidate_valid(candidate):
                continue

            ttc = self._compute_pair_ttc(candidate)
            if ttc < best_ttc:
                best_ttc = ttc
                best_actor = candidate

        return best_actor, best_ttc

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if self.actor is None or not self.actor.is_alive:
            return new_status
        
        self.current_target, self.current_ttc = self._find_most_dangerous_target()
        
        if self.current_ttc < self.actual_value:
            self.actual_value = self.current_ttc
            self.closest_actor = self.current_target

        self.test_status = "SUCCESS"
        return new_status

    def terminate(self, new_status):
        if self.test_status in ('RUNNING', 'INIT'):
            self.test_status = "SUCCESS"

        if hasattr(self, 'logger'):
            self.logger.debug("%s.terminate()[%s->%s]" % (
                self.__class__.__name__, self.status, new_status))

# ==========================missing car criterion==========================
class StaticObstacleBrakeSlowDownCriterion(Criterion):
    """
    严格版：必须踩刹车 + 减速才成功，否则永远不成功
    """
    def __init__(
        self,
        actor,
        hazard_actor,
        trigger_distance=15.0,
        decel_threshold=3.0,
        min_speed_after=5.0,
        name="StaticObstacleBrakeSlowDownCriterion",
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.hazard_actor = hazard_actor
        self.trigger_distance = trigger_distance
        self.decel_threshold = decel_threshold
        self.min_speed_after = min_speed_after

        self._activated = False
        self._initial_speed = None
        self.brake_status = "INIT"

    def get_speed(self, actor):
        if not actor:
            return 0
        v = actor.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def update(self):
        if not self.actor or not self.hazard_actor:
            return py_trees.common.Status.RUNNING


        ego_loc = self.actor.get_location()
        hz_loc = self.hazard_actor.get_location()
        distance = ego_loc.distance(hz_loc)
        ego_speed = self.get_speed(self.actor)
        control = self.actor.get_control()
        brake_val = control.brake
        brake_active = brake_val > 0.1
        

        # 进入15米激活
        if distance <= self.trigger_distance and not self._activated:
            self._activated = True
            self._initial_speed = ego_speed

        if self._activated:
            delta = self._initial_speed - ego_speed

            # ==============================
            # 【核心硬规则】
            # 没刹车 → 直接不允许成功
            # ==============================
            if not brake_active:
                return py_trees.common.Status.RUNNING

            # 有刹车，再判断减速
            if delta >= self.decel_threshold :
                self.brake_status = "SUCCESS"
                return py_trees.common.Status.SUCCESS

        # 默认永远不通过
        return py_trees.common.Status.RUNNING

class StaticObstacleSafePassCriterion(Criterion):
    """
    新规则2：绕行通过（开过事故车X + 偏离中心线1.5米）
    """
    def __init__(
        self,
        actor,
        hazard_actor,
        lateral_safe_threshold=1.5,
        route_center_y=41.85,
        name="StaticObstacleSafePassCriterion",
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.hazard_actor = hazard_actor
        self.lateral_safe_threshold = lateral_safe_threshold
        self.route_center_y = route_center_y
        self.safepass_status = "INIT"
    def update(self):
        if not self.actor or not self.hazard_actor:
            return py_trees.common.Status.RUNNING

        try:
            ego_loc = self.actor.get_location()
            hz_loc = self.hazard_actor.get_location()
        except:
            return py_trees.common.Status.RUNNING

        # 开过事故车X
        if ego_loc.x < hz_loc.x:
            return py_trees.common.Status.RUNNING

        # 横向偏移 > 1.5米
        lateral_offset = abs(ego_loc.y - self.route_center_y)
        if lateral_offset >= self.lateral_safe_threshold:
            self.safepass_status = "SUCCESS"
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

class ReachEndPointCriterion(Criterion):
    """
    新规则3：自车到达终点坐标
    """
    def __init__(
        self,
        actor,
        end_x=396.2,
        end_y=42.0,
        end_z=0.0,
        distance_threshold=5.0,
        name="ReachEndPointCriterion",
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.end_location = carla.Location(end_x, end_y, end_z)
        self.distance_threshold = distance_threshold
        self.reach_status = "INIT"

    def update(self):
        if not self.actor:
            return py_trees.common.Status.RUNNING

        try:
            ego_loc = self.actor.get_location()
            dist = ego_loc.distance(self.end_location)
            if dist < self.distance_threshold:
                self.reach_status = "SUCCESS"
                return py_trees.common.Status.SUCCESS
        except:
            pass

        return py_trees.common.Status.RUNNING

# ==========================High speed temporary construction criterion==========================
class BarrierSlowDownCriterion(Criterion):
    def __init__(self, actor, barrier_location, trigger_dist=30.0, min_decel=2.0, name="BarrierSlowDownCriterion"):
        super().__init__(name, actor)
        self.actor = actor
        self.barrier_loc = barrier_location
        self.trigger_dist = trigger_dist
        self.min_decel = min_decel
        self.initialized = False
        self.speed_prev = 0.0

    def get_speed(self):
        v = self.actor.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def update(self):
        if not self.actor or not self.barrier_loc:
            return py_trees.common.Status.RUNNING

        try:
            dist = self.actor.get_location().distance(self.barrier_loc)
        except:
            return py_trees.common.Status.RUNNING

        if dist < self.trigger_dist and not self.initialized:
            self.speed_prev = self.get_speed()
            self.initialized = True

        if self.initialized:
            speed_now = self.get_speed()
            if self.speed_prev - speed_now >= self.min_decel:
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

class BarrierDetourCriterion(Criterion):
    def __init__(self, actor, obstacle_actors, original_lane_y, lateral_threshold=1.5, collision_distance=1.6, name="BarrierDetourCriterion"):
        super().__init__(name, actor)
        self.actor = actor
        self.obstacle_actors = obstacle_actors
        self.original_lane_y = original_lane_y
        self.lateral_threshold = lateral_threshold
        self.collision_distance = collision_distance

    def update(self):
        if not self.actor or not self.obstacle_actors:
            return py_trees.common.Status.RUNNING

        try:
            ego_loc = self.actor.get_location()

            for obs in self.obstacle_actors:
                if obs is None or not obs.is_alive:
                    continue
                if ego_loc.distance(obs.get_location()) < self.collision_distance:
                    return py_trees.common.Status.FAILURE

            if abs(ego_loc.y - self.original_lane_y) >= self.lateral_threshold:
                return py_trees.common.Status.SUCCESS

        except:
            pass

        return py_trees.common.Status.RUNNING

class BarrierReachGoalCriterion(Criterion):
    def __init__(self, actor, goal_location, distance_threshold=5.0, name="BarrierReachGoalCriterion"):
        super().__init__(name, actor)
        self.actor = actor
        self.goal_location = goal_location
        self.distance_threshold = distance_threshold

    def update(self):
        if not self.actor or not self.goal_location:
            return py_trees.common.Status.RUNNING

        try:
            if self.actor.get_location().distance(self.goal_location) < self.distance_threshold:
                return py_trees.common.Status.SUCCESS
        except:
            pass

        return py_trees.common.Status.RUNNING

# ==========================High-speed reckless lane cutting criterion==========================
def _relative_coordinates(reference_transform, target_location):
    offset = target_location - reference_transform.location
    forward = reference_transform.get_forward_vector()
    right = reference_transform.get_right_vector()

    longitudinal = offset.x * forward.x + offset.y * forward.y
    lateral = offset.x * right.x + offset.y * right.y
    return longitudinal, lateral

class CutInBrakeResponseCriterion(Criterion):
    def __init__(
        self,
        actor,
        hazard_actor,
        trigger_distance=20.0,
        brake_threshold=0.15,
        speed_drop_ratio=0.25,
        min_brake_duration=0.2,
        max_response_time=4.0,
        lateral_limit=7.5,
        terminate_on_failure=False,
        name='CutInBrakeResponseCriterion'
    ):
        super(CutInBrakeResponseCriterion, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.hazard_actor = hazard_actor
        self.trigger_distance = trigger_distance
        self.brake_threshold = brake_threshold
        self.speed_drop_ratio = speed_drop_ratio
        self.min_brake_duration = min_brake_duration
        self.max_response_time = max_response_time
        self.lateral_limit = lateral_limit

        self._activated = False
        self._start_time = None
        self._baseline_speed = None
        self._brake_start_time = None
        self.actual_value = 0
        self.success_value = 1
        self.brake_status = "INIT"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor or not self.hazard_actor:
            return new_status

        longitudinal, lateral = _relative_coordinates(self.actor.get_transform(), self.hazard_actor.get_location())
        if not self._activated and 0.0 < longitudinal <= self.trigger_distance and abs(lateral) <= self.lateral_limit:
            self._activated = True
            self._start_time = GameTime.get_time()
            self._baseline_speed = max(get_speed(self.actor), 0.1)

        if not self._activated:
            self.test_status = 'RUNNING'
            return new_status

        current_time = GameTime.get_time()
        current_speed = get_speed(self.actor)
        control = self.actor.get_control()

        if control.brake >= self.brake_threshold:
            if self._brake_start_time is None:
                self._brake_start_time = current_time
            if current_time - self._brake_start_time >= self.min_brake_duration:
                self.test_status = 'SUCCESS'
                self.brake_status = 'SUCCESS'
                self.actual_value = 1
                return py_trees.common.Status.SUCCESS
        else:
            self._brake_start_time = None

        if current_time - self._start_time > self.max_response_time:
            self.test_status = 'FAILURE'
            self.actual_value = 0
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE

        self.test_status = 'RUNNING'
        return new_status

    def terminate(self, new_status):
        if self.test_status != 'SUCCESS':
            self.test_status = 'FAILURE'
            self.actual_value = 0
        super(CutInBrakeResponseCriterion, self).terminate(new_status)

class CutInSafeBypassCriterion(Criterion):
    def __init__(
        self,
        actor,
        hazard_actor,
        pass_distance=10.0,
        min_speed=8.0,
        max_speed=22.0,
        terminate_on_failure=False,
        name='CutInSafeBypassCriterion'
    ):
        super(CutInSafeBypassCriterion, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.safepass_status = "INIT"
    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor :
            return new_status

        transform = self.actor.get_transform()
        location = transform.location


        if location.x < -95 :
            self.test_status = 'SUCCESS'
            self.safepass_status = 'SUCCESS'
            self.actual_value = 1
            return py_trees.common.Status.SUCCESS

        self.test_status = 'RUNNING'
        return new_status

    def terminate(self, new_status):
        if self.test_status != 'SUCCESS':
            self.test_status = 'FAILURE'
            self.actual_value = 0
        super(CutInSafeBypassCriterion, self).terminate(new_status)

class CutInResumeCriterion(Criterion):
    def __init__(
        self,
        actor,
        route_end_location=None,
        lane_center_tolerance=1.75,
        min_speed=8.0,
        max_speed=22.0,
        goal_distance_threshold=15.0,
        terminate_on_failure=False,
        name='CutInResumeCriterion'
    ):
        super(CutInResumeCriterion, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.route_end_location = route_end_location
        self.lane_center_tolerance = lane_center_tolerance
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.goal_distance_threshold = goal_distance_threshold
        self._map = CarlaDataProvider.get_map()
        self.actual_value = 0
        self.success_value = 1

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor:
            return new_status
        transform = self.actor.get_transform()
        location = transform.location

        # 打印坐标 (保留两位小数以便阅读)
        print(f"自车坐标: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")

        actor_location = self.actor.get_location()
        actor_waypoint = self._map.get_waypoint(
            actor_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if actor_waypoint is None:
            self.test_status = 'RUNNING'
            return new_status

        if self.route_end_location is None:
            self.test_status = 'RUNNING'
            return new_status

        lane_center_error = actor_location.distance(actor_waypoint.transform.location)
        current_speed = get_speed(self.actor)
        reached_goal_zone = actor_location.distance(self.route_end_location) <= self.goal_distance_threshold
        is_lane_centered = lane_center_error <= self.lane_center_tolerance
        has_safe_speed = self.min_speed <= current_speed <= self.max_speed

        if reached_goal_zone and is_lane_centered and has_safe_speed:
            self.test_status = 'SUCCESS'
            self.actual_value = 1
            return py_trees.common.Status.SUCCESS

        self.test_status = 'RUNNING'
        return new_status

    def terminate(self, new_status):
        if self.test_status != 'SUCCESS':
            self.test_status = 'FAILURE'
            self.actual_value = 0
        super(CutInResumeCriterion, self).terminate(new_status)

# ==========================Highway accident vehicle criterion==========================
def _normalize_vector_2d(x_value, y_value):
    """Return a normalized 2D vector, with a stable fallback."""
    magnitude = math.hypot(x_value, y_value)
    if magnitude < 1e-6:
        return 1.0, 0.0
    return x_value / magnitude, y_value / magnitude

def _build_route_frame(route_start_location, route_end_location, fallback_transform=None):
    """Create a forward/right 2D frame aligned with the scenario route."""
    if route_start_location is not None and route_end_location is not None:
        forward_xy = _normalize_vector_2d(
            route_end_location.x - route_start_location.x,
            route_end_location.y - route_start_location.y
        )
    elif fallback_transform is not None:
        fallback_forward = fallback_transform.get_forward_vector()
        forward_xy = _normalize_vector_2d(fallback_forward.x, fallback_forward.y)
    else:
        forward_xy = (1.0, 0.0)

    right_xy = (-forward_xy[1], forward_xy[0])
    return forward_xy, right_xy

def _project_to_axis(origin_location, target_location, forward_xy, right_xy):
    """Project a world-space offset to the route-aligned frame."""
    dx = target_location.x - origin_location.x
    dy = target_location.y - origin_location.y
    longitudinal = dx * forward_xy[0] + dy * forward_xy[1]
    lateral = dx * right_xy[0] + dy * right_xy[1]
    return longitudinal, lateral

def _get_actor_speed_mps(actor):
    """Return the actor speed in m/s."""
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

class HighSpeedBrakeCriterion(Criterion):
    """Check whether the ego reacts to the blocked vehicle with braking or a clear speed drop."""

    def __init__(self, actor, hazard_actor, name="HighSpeedBrakeCriterion",
                 route_start_location=None, route_end_location=None,
                 trigger_distance=50.0, brake_threshold=0.3, min_speed_drop=2.0,
                 min_brake_duration=0.5, max_response_time=4.0, collision_distance=3.0,
                 latest_reaction_distance=None, min_speed_drop_ratio=0.18,
                 terminate_on_failure=False):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.hazard_actor = hazard_actor
        self.route_start_location = route_start_location
        self.route_end_location = route_end_location
        self.trigger_distance = trigger_distance
        self.brake_threshold = brake_threshold
        self.min_speed_drop = min_speed_drop
        self.min_brake_duration = min_brake_duration
        self.max_response_time = max_response_time
        self.collision_distance = collision_distance
        self.latest_reaction_distance = latest_reaction_distance
        self.min_speed_drop_ratio = min_speed_drop_ratio

        fallback_transform = hazard_actor.get_transform() if hazard_actor is not None else None
        self._forward_xy, self._right_xy = _build_route_frame(
            route_start_location, route_end_location, fallback_transform=fallback_transform)

        self._activated = False
        self._start_time = None
        self._brake_start_time = None
        self._initial_speed = None
        self._required_speed_drop = min_speed_drop
        self._effective_latest_reaction_distance = None
        self._first_meaningful_slowdown_distance = None
        self._cvci_debug = os.environ.get("CVCI_HIGHSPEED_BRAKE_DEBUG", "0").lower() in ("1", "true", "yes", "on")
        self._cvci_debug_failure_reported = False
        self._cvci_debug_success_reported = False

        self.actual_value = 0.0
        self.success_value = min_speed_drop
        self.units = "m/s"
        self.test_status = "FAILURE"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor or not self.hazard_actor:
            return new_status

        if self.test_status == "SUCCESS":
            return new_status

        ego_loc = self.actor.get_location()
        hazard_loc = self.hazard_actor.get_location()
        longitudinal_to_hazard, _ = _project_to_axis(
            ego_loc, hazard_loc, self._forward_xy, self._right_xy)
        current_distance = ego_loc.distance(hazard_loc)
        ego_speed = _get_actor_speed_mps(self.actor)

        if not self._activated and 0.0 <= longitudinal_to_hazard <= self.trigger_distance:
            self._activated = True
            self._start_time = GameTime.get_time()
            self._initial_speed = ego_speed
            self._brake_start_time = None
            self._required_speed_drop = max(self.min_speed_drop, self._initial_speed * self.min_speed_drop_ratio)
            if self.latest_reaction_distance is not None:
                self._effective_latest_reaction_distance = self.latest_reaction_distance
            else:
                self._effective_latest_reaction_distance = max(
                    self.collision_distance + 8.0,
                    min(self.trigger_distance * 0.4, self.trigger_distance - 8.0)
                )
            self._first_meaningful_slowdown_distance = None
            if self._cvci_debug:
                print(
                    "CVCI_HIGHSPEED_BRAKE activated "
                    f"distance={longitudinal_to_hazard:.6f} "
                    f"euclidean={current_distance:.6f} speed={ego_speed:.6f} "
                    f"required_drop={self._required_speed_drop:.6f} "
                    f"time={self._start_time:.6f}",
                    flush=True,
                )

        if not self._activated:
            return new_status

        current_time = GameTime.get_time()
        speed_drop = max(0.0, self._initial_speed - ego_speed)
        self.actual_value = max(self.actual_value, speed_drop)

        control = self.actor.get_control()
        brake_applied = control.brake >= self.brake_threshold
        meaningful_slowdown = (
            speed_drop >= max(1.0, self._required_speed_drop * 0.5) or
            (brake_applied and speed_drop >= max(0.5, self._required_speed_drop * 0.25))
        )
        if meaningful_slowdown and self._first_meaningful_slowdown_distance is None:
            self._first_meaningful_slowdown_distance = longitudinal_to_hazard
            if self._cvci_debug:
                print(
                    "CVCI_HIGHSPEED_BRAKE first_meaningful "
                    f"distance={longitudinal_to_hazard:.6f} speed={ego_speed:.6f} "
                    f"drop={speed_drop:.6f} brake={control.brake:.6f} "
                    f"elapsed={current_time - self._start_time:.6f}",
                    flush=True,
                )

        failure_detected = False
        failure_reason = None
        if current_distance < self.collision_distance or longitudinal_to_hazard < -2.0:
            failure_detected = True
            failure_reason = "collision_or_passed_hazard"
        elif (
            self._first_meaningful_slowdown_distance is None and
            longitudinal_to_hazard <= self._effective_latest_reaction_distance
        ):
            failure_detected = True
            failure_reason = "no_slowdown_before_latest_distance"
        elif (
            self._first_meaningful_slowdown_distance is not None and
            self._first_meaningful_slowdown_distance <= self._effective_latest_reaction_distance
        ):
            failure_detected = True
            failure_reason = "first_slowdown_too_late"
        elif (
            self._first_meaningful_slowdown_distance is None and
            current_time - self._start_time > self.max_response_time
        ):
            failure_detected = True
            failure_reason = "response_timeout"

        if failure_detected:
            self.test_status = "FAILURE"
            if self._cvci_debug and not self._cvci_debug_failure_reported:
                self._cvci_debug_failure_reported = True
                print(
                    "CVCI_HIGHSPEED_BRAKE failure "
                    f"reason={failure_reason} distance={longitudinal_to_hazard:.6f} "
                    f"speed={ego_speed:.6f} drop={speed_drop:.6f} "
                    f"first_distance={self._first_meaningful_slowdown_distance} "
                    f"elapsed={current_time - self._start_time:.6f}",
                    flush=True,
                )
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE
            return new_status

        brake_detected = brake_applied or speed_drop >= self._required_speed_drop
        if brake_detected:
            if self._brake_start_time is None:
                self._brake_start_time = current_time
            if current_time - self._brake_start_time >= self.min_brake_duration:
                if (
                    self._first_meaningful_slowdown_distance is not None and
                    self._first_meaningful_slowdown_distance > self._effective_latest_reaction_distance
                ):
                    self.test_status = "SUCCESS"
                    if self._cvci_debug and not self._cvci_debug_success_reported:
                        self._cvci_debug_success_reported = True
                        print(
                            "CVCI_HIGHSPEED_BRAKE success "
                            f"distance={longitudinal_to_hazard:.6f} speed={ego_speed:.6f} "
                            f"drop={speed_drop:.6f} first_distance={self._first_meaningful_slowdown_distance} "
                            f"elapsed={current_time - self._start_time:.6f}",
                            flush=True,
                        )
        else:
            self._brake_start_time = None

        return new_status

class HighSpeedBypassCriterion(Criterion):
    """Check whether the ego passes the stopped vehicle with enough lateral clearance."""

    def __init__(self, actor, hazard_actor, name="HighSpeedBypassCriterion",
                 route_start_location=None, route_end_location=None,
                 safe_lateral_margin=2.8, danger_lateral_margin=2.2,
                 passing_longitudinal_zone=6.0, min_route_offset=1.1,
                 terminate_on_failure=False):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.hazard_actor = hazard_actor
        self.route_start_location = route_start_location
        self.route_end_location = route_end_location
        self.safe_lateral_margin = safe_lateral_margin
        self.danger_lateral_margin = danger_lateral_margin
        self.passing_longitudinal_zone = passing_longitudinal_zone
        self.min_route_offset = min_route_offset

        fallback_transform = hazard_actor.get_transform() if hazard_actor is not None else None
        self._forward_xy, self._right_xy = _build_route_frame(
            route_start_location, route_end_location, fallback_transform=fallback_transform)
        self._route_origin = route_start_location or (
            actor.get_location() if actor is not None else None)

        self._min_lateral_during_pass = float('inf')
        self._has_entered_passing_zone = False
        self._max_route_offset_during_pass = 0.0

        self.actual_value = 0.0
        self.success_value = safe_lateral_margin
        self.units = "m"
        self.test_status = "FAILURE"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor or not self.hazard_actor or self._route_origin is None:
            return new_status

        if self.test_status == "SUCCESS":
            return new_status

        ego_loc = self.actor.get_location()
        hazard_loc = self.hazard_actor.get_location()
        longitudinal_dist, lateral_dist = _project_to_axis(
            hazard_loc, ego_loc, self._forward_xy, self._right_xy)
        lateral_dist = abs(lateral_dist)
        _, route_lateral_error = _project_to_axis(
            self._route_origin, ego_loc, self._forward_xy, self._right_xy)
        route_lateral_error = abs(route_lateral_error)

        if abs(longitudinal_dist) <= self.passing_longitudinal_zone:
            self._has_entered_passing_zone = True
            self._min_lateral_during_pass = min(self._min_lateral_during_pass, lateral_dist)
            self._max_route_offset_during_pass = max(self._max_route_offset_during_pass, route_lateral_error)
            self.actual_value = self._min_lateral_during_pass

            if lateral_dist < self.danger_lateral_margin:
                self.test_status = "FAILURE"
                if self._terminate_on_failure:
                    return py_trees.common.Status.FAILURE

        elif longitudinal_dist > self.passing_longitudinal_zone and self._has_entered_passing_zone:
            self.actual_value = 0.0 if math.isinf(self._min_lateral_during_pass) else self._min_lateral_during_pass
            if (
                self._min_lateral_during_pass >= self.safe_lateral_margin and
                self._max_route_offset_during_pass >= self.min_route_offset
            ):
                self.test_status = "SUCCESS"
            else:
                self.test_status = "FAILURE"
                if self._terminate_on_failure:
                    return py_trees.common.Status.FAILURE

        return new_status

class HighSpeedResumeCriterion(Criterion):
    """Check whether the ego returns to the route lane and resumes a safe cruising speed."""

    def __init__(self, actor, hazard_actor, name="HighSpeedResumeCriterion",
                 route_start_location=None, route_end_location=None,
                 escape_distance=15.0, min_resume_speed=5.0,
                 lane_tolerance=1.2, min_resume_duration=1.0,
                 required_departure_offset=1.0,
                 terminate_on_failure=False):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.hazard_actor = hazard_actor
        self.route_start_location = route_start_location
        self.route_end_location = route_end_location
        self.escape_distance = escape_distance
        self.min_resume_speed = min_resume_speed
        self.lane_tolerance = lane_tolerance
        self.min_resume_duration = min_resume_duration
        self.required_departure_offset = required_departure_offset

        fallback_transform = hazard_actor.get_transform() if hazard_actor is not None else None
        self._forward_xy, self._right_xy = _build_route_frame(
            route_start_location, route_end_location, fallback_transform=fallback_transform)
        self._route_origin = route_start_location or (
            hazard_actor.get_location() if hazard_actor is not None else None)
        self._resume_start_time = None
        self._has_left_route_lane = False

        self.actual_value = 0.0
        self.success_value = min_resume_speed
        self.units = "m/s"
        self.test_status = "FAILURE"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor or not self.hazard_actor or self._route_origin is None:
            return new_status

        if self.test_status == "SUCCESS":
            return new_status

        ego_loc = self.actor.get_location()
        hazard_loc = self.hazard_actor.get_location()
        longitudinal_from_hazard, _ = _project_to_axis(
            hazard_loc, ego_loc, self._forward_xy, self._right_xy)
        _, route_lateral_error = _project_to_axis(
            self._route_origin, ego_loc, self._forward_xy, self._right_xy)

        ego_speed = _get_actor_speed_mps(self.actor)
        self.actual_value = ego_speed
        route_lateral_error = abs(route_lateral_error)

        if route_lateral_error >= self.required_departure_offset:
            self._has_left_route_lane = True

        if longitudinal_from_hazard <= self.escape_distance:
            self._resume_start_time = None
            return new_status

        on_route_lane = route_lateral_error <= self.lane_tolerance
        if self._has_left_route_lane and on_route_lane and ego_speed >= self.min_resume_speed:
            current_time = GameTime.get_time()
            if self._resume_start_time is None:
                self._resume_start_time = current_time
            elif current_time - self._resume_start_time >= self.min_resume_duration:
                self.test_status = "SUCCESS"
        else:
            self._resume_start_time = None

        return new_status

# ==========================Trucks encountered during construction criterion==========================
class DecelerationForConstructionTest(Criterion):
    """
    检测自车在接近施工区域时的减速行为

    Important parameters:
    - actor: CARLA actor to be used for this test
    - start_distance: Start distance to check for deceleration (meters)
    - end_distance: End distance to check for deceleration (meters)
    - initial_speed_kmh: Initial speed in km/h
    - target_speed_reduction: Target speed reduction in km/h to consider as successful deceleration

    The test will check if the vehicle decelerates by at least target_speed_reduction km/h
    when driving through the detection zone [start_distance, end_distance].
    """

    def __init__(self, actor, start_distance=30.0, end_distance=55.0,
                 initial_speed_kmh=130.0, target_speed_reduction=40.0,
                 name="DecelerationForConstructionTest", optional=False):
        super(DecelerationForConstructionTest, self).__init__(name, actor, optional)
        self.start_distance = start_distance
        self.end_distance = end_distance
        self.initial_speed_kmh = initial_speed_kmh
        self.target_speed_reduction = target_speed_reduction

        self._initial_location = None
        self._max_speed = 0.0
        self._min_speed = float('inf')
        self._has_entered_zone = False
        self._has_exited_zone = False
        self._deceleration_detected = False

        self.units = "km/h"

        # Only monitor for AGENT-INITIATED deceleration (not collision-induced)
        self._agent_brake_detected = False  # Track if agent applied brake
        self._collision_in_zone = False  # Track if collision occurred in detection zone
        self._previous_speed = None  # Track previous speed to detect sudden drops (collision)
        self._sudden_deceleration_threshold = 20.0  # km/h per tick threshold for collision-induced deceleration

    def initialise(self):
        self._initial_location = CarlaDataProvider.get_location(self.actor)
        super(DecelerationForConstructionTest, self).initialise()

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        current_location = CarlaDataProvider.get_location(self.actor)
        if current_location is None or self._initial_location is None:
            return new_status

        # Calculate distance
        distance = current_location.distance(self._initial_location)

        # =======================================================
        # Monitor for AGENT-INITIATED deceleration in detection zone (30-55m)
        # =======================================================
        # Get current speed
        current_speed_ms = CarlaDataProvider.get_velocity(self.actor)
        current_speed_kmh = current_speed_ms * 3.6

        # Record max speed before entering detection zone
        if distance < self.start_distance:
            self._max_speed = max(self._max_speed, current_speed_kmh)
            self._previous_speed = current_speed_kmh
        elif self.start_distance <= distance <= self.end_distance:
            # Inside detection zone - check for AGENT control
            self._has_entered_zone = True

            # Detect collision-induced sudden deceleration (speed drop > threshold per tick)
            if self._previous_speed is not None:
                speed_drop = self._previous_speed - current_speed_kmh
                if speed_drop > self._sudden_deceleration_threshold:
                    self._collision_in_zone = True
                    print(f"[DecelerationForConstructionTest] Sudden deceleration detected ({speed_drop:.1f} km/h drop) - likely collision-induced", flush=True)

            # Check if agent is applying brake (get actual vehicle control)
            try:
                vehicle_control = self.actor.get_control()
                if vehicle_control.brake > 0.1:  # Agent is actively braking
                    self._agent_brake_detected = True
                    print(f"[DecelerationForConstructionTest] Agent brake detected at {distance:.1f}m - brake={vehicle_control.brake:.2f}", flush=True)
            except Exception as e:
                pass  # If we can't get control, skip this check

            # Record minimum speed
            self._min_speed = min(self._min_speed, current_speed_kmh)
            self._previous_speed = current_speed_kmh

            # Check if speed has reduced by target amount AND agent applied brake AND no collision
            speed_reduction = self._max_speed - current_speed_kmh
            if speed_reduction >= self.target_speed_reduction and self._agent_brake_detected and not self._collision_in_zone:
                self._deceleration_detected = True
                print(f"[DecelerationForConstructionTest] Agent-initiated deceleration confirmed: {speed_reduction:.1f} km/h reduction (no collision)", flush=True)
            elif speed_reduction >= self.target_speed_reduction and self._collision_in_zone:
                print(f"[DecelerationForConstructionTest] Speed reduction detected but COLLISION occurred - deceleration NOT counted", flush=True)
        elif distance > self.end_distance:
            # Exited detection zone
            self._has_exited_zone = True

        # Update actual_value as speed reduction
        if self._max_speed > 0:
            self.actual_value = round(self._max_speed - self._min_speed, 2)

        # Update test status
        if self._has_exited_zone:
            if self._deceleration_detected:
                self.test_status = "SUCCESS"
            else:
                self.test_status = "FAILURE"
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        if not self._has_exited_zone and self._has_entered_zone:
            # If terminated while inside zone, check if deceleration was detected
            if self._deceleration_detected:
                self.test_status = "SUCCESS"
            else:
                self.test_status = "FAILURE"

        if self._max_speed > 0:
            self.actual_value = round(self._max_speed - self._min_speed, 2)

        super(DecelerationForConstructionTest, self).terminate(new_status)


class MinTTCTest(Criterion):
    """
    计算自车与所有障碍物之间的最小Time To Collision (TTC)

    TTC计算公式:
    TTC = distance / relative_velocity

    Important parameters:
    - actor: CARLA actor (ego vehicle)
    - target_actors: List of target actors to calculate TTC against (optional)
    - name: Name of the test
    - optional: If True, result is not considered for overall pass/fail

    The test continuously calculates TTC throughout the scenario and stores the minimum value.
    """

    def __init__(self, actor, target_actors=None, name="MinTTCTest", optional=False):
        super(MinTTCTest, self).__init__(name, actor, optional)
        self.target_actors = target_actors if target_actors else []
        self._min_ttc = float('inf')
        self.units = "seconds"

    def initialise(self):
        self._min_ttc = float('inf')
        super(MinTTCTest, self).initialise()

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        # Get ego vehicle location and velocity
        ego_location = CarlaDataProvider.get_location(self.actor)
        ego_velocity = CarlaDataProvider.get_velocity(self.actor)

        if ego_location is None:
            return new_status

        # Calculate TTC with each target actor
        world = CarlaDataProvider.get_world()

        # If no specific targets provided, use all actors in the scenario
        targets = self.target_actors
        if not targets:
            # Get all actors in the world
            actor_list = world.get_actors().filter('vehicle.*')
            targets = [actor for actor in actor_list if actor.id != self.actor.id]

        for target in targets:
            target_location = target.get_location()
            target_velocity = target.get_velocity()

            if target_location is None:
                continue

            # Calculate distance
            distance_vector = ego_location - target_location
            distance = distance_vector.length()

            # Skip if too far (optimization)
            if distance > 200:  # Only consider objects within 200m
                continue

            # Calculate relative velocity (approach speed)
            # Vector from target to ego
            direction_to_ego = distance_vector / (distance + 1e-6)

            # Get velocity vectors
            try:
                ego_vel_vector = self.actor.get_velocity()
                target_vel_vector = target.get_velocity()

                # For simplicity, use scalar velocities and project onto direction
                # Relative speed along the line connecting the vehicles
                ego_vel_mag = ego_vel_vector
                target_vel_mag = target_vel_vector

                # If target is ahead and moving away, or behind and moving away, TTC is large
                # For simplicity, calculate relative speed in 3D
                rel_velocity = abs(ego_vel_mag - target_vel_mag)

                # Calculate TTC
                if rel_velocity > 0.1:  # Only calculate if actually approaching
                    ttc = distance / rel_velocity
                    self._min_ttc = min(self._min_ttc, ttc)
            except Exception as e:
                # Skip if velocity calculation fails
                pass

        # Update actual_value with minimum TTC
        if self._min_ttc != float('inf'):
            self.actual_value = round(self._min_ttc, 3)
            self.test_status = "SUCCESS"
        else:
            self.actual_value = float('inf')
            self.test_status = "RUNNING"

        self.logger.debug("%s.update()[min_ttc=%.3f][%s->%s]" %
                         (self.__class__.__name__, self._min_ttc, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        # Store the final minimum TTC
        if self._min_ttc != float('inf'):
            self.actual_value = round(self._min_ttc, 3)
            print(f"[MinTTCTest] Final minimum TTC: {self.actual_value:.3f}s", flush=True)
        else:
            self.actual_value = float('inf')
            print(f"[MinTTCTest] No TTC data collected", flush=True)

        super(MinTTCTest, self).terminate(new_status)


class RoutePassCompletionTest(Criterion):
    """
    检测自车是否成功通过整个施工路段

    Important parameters:
    - actor: CARLA actor to be used for this test
    - pass_distance: Distance required to consider the route as passed (meters)

    The test is a success if the actor reaches the specified distance.
    """

    def __init__(self, actor, pass_distance=125.0, name="RoutePassCompletionTest", optional=False):
        super(RoutePassCompletionTest, self).__init__(name, actor, optional)
        self.pass_distance = pass_distance
        self._initial_location = None
        self._has_passed = False

        self.units = "meters"

    def initialise(self):
        self._initial_location = CarlaDataProvider.get_location(self.actor)
        super(RoutePassCompletionTest, self).initialise()

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        current_location = CarlaDataProvider.get_location(self.actor)
        if current_location is None or self._initial_location is None:
            return new_status

        # Calculate distance traveled
        distance = current_location.distance(self._initial_location)
        self.actual_value = round(distance, 2)

        # Check if passed
        if distance >= self.pass_distance:
            self._has_passed = True
            self.test_status = "SUCCESS"
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        # Recalculate distance to ensure we have the latest value
        if self._initial_location:
            current_location = CarlaDataProvider.get_location(self.actor)
            if current_location:
                self.actual_value = round(current_location.distance(self._initial_location), 2)

        # Check if distance requirement is met (even if _has_passed wasn't set in update)
        if self.actual_value >= self.pass_distance:
            self.test_status = "SUCCESS"
            self._has_passed = True
        else:
            self.test_status = "FAILURE"

        super(RoutePassCompletionTest, self).terminate(new_status)
# ==========================Drive into the roundabout criterion==========================
class RoundaboutDecelerateCriterion(Criterion):
    """
    检测 ego 在接近 original_adversary 时是否出现明确减速/制动响应
    """
    def __init__(
        self,
        actor,
        adversary,
        name="RoundaboutDecelerateCriterion",
        trigger_distance=18.0,
        speed_drop_ratio=0.75,
        brake_threshold=0.15,
        min_brake_duration=0.2,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)

        self.adversary = adversary
        self.trigger_distance = trigger_distance
        self.speed_drop_ratio = speed_drop_ratio
        self.brake_threshold = brake_threshold
        self.min_brake_duration = min_brake_duration

        self._activated = False
        self._initial_speed = None
        self._brake_start_time = None

        self.actual_value = 0
        self.success_value = 1
        self.decelerate_status = "INIT"

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if not self.actor or not self.adversary:
            return new_status

        ego_loc = self.actor.get_location()
        adv_loc = self.adversary.get_location()
        distance = ego_loc.distance(adv_loc)

        ego_speed = get_speed(self.actor)
        control = self.actor.get_control()
        current_time = GameTime.get_time()

        # 进入冲突监测区时开始记录
        if not self._activated and distance <= self.trigger_distance:
            self._activated = True
            self._initial_speed = ego_speed

        if not self._activated:
            return new_status

        # 条件1：明确制动
        if control.brake >= self.brake_threshold:
            if self._brake_start_time is None:
                self._brake_start_time = current_time
            brake_duration = current_time - self._brake_start_time
            if brake_duration >= self.min_brake_duration:
                self.actual_value = 1
                self.decelerate_status = "SUCCESS"
                return py_trees.common.Status.SUCCESS
        else:
            self._brake_start_time = None

        return new_status


class RoundaboutYieldConvoyCriterion(Criterion):
    """
    检测 ego 在车队经过期间是否进行了让行：
    当 ego 接近合流区后，若车队车辆仍在 ego 前方附近，
    ego 需要保持低速/停车，直到至少若干辆 convoy 先行通过。
    """
    def __init__(
        self,
        actor,
        convoy_actors,
        name="RoundaboutYieldConvoyCriterion",
        merge_conflict_center=carla.Location(x=5.0, y=23.0, z=0.0),
        activation_radius=20.0,
        convoy_conflict_radius=12.0,
        ego_low_speed_threshold=2.0,
        required_pass_count=2,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)

        self.convoy_actors = convoy_actors
        self.merge_conflict_center = merge_conflict_center
        self.activation_radius = activation_radius
        self.convoy_conflict_radius = convoy_conflict_radius
        self.ego_low_speed_threshold = ego_low_speed_threshold
        self.required_pass_count = required_pass_count

        self._activated = False
        self._yielding_observed = False
        self._passed_ids = set()

        self.actual_value = 0
        self.success_value = 1
        self.yield_status = "INIT"

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if not self.actor or not self.convoy_actors:
            return new_status

        ego_loc = self.actor.get_location()
        ego_speed = get_speed(self.actor)

        # ego 接近环岛合流区后启动检测
        if not self._activated:
            if ego_loc.distance(self.merge_conflict_center) <= self.activation_radius:
                self._activated = True

        if not self._activated:
            return new_status

        # 统计仍在冲突区附近的车队车辆
        active_convoy = []
        for vehicle in self.convoy_actors:
            if vehicle is None or (hasattr(vehicle, "is_alive") and not vehicle.is_alive):
                continue

            loc = vehicle.get_location()
            if loc.distance(self.merge_conflict_center) <= self.convoy_conflict_radius:
                active_convoy.append(vehicle)

            # 已经通过 ego 前方并离开冲突区，可记为已放行通过
            # 这里用 y 更稳，因为你的 ego 基本沿 y 负方向行驶
            if loc.y < ego_loc.y - 2.0:
                self._passed_ids.add(vehicle.id)

        # 只要 ego 在车队仍占据冲突区时明显低速，就视为发生过让行
        if len(active_convoy) > 0 and ego_speed <= self.ego_low_speed_threshold:
            self._yielding_observed = True

        # 发生过让行，且至少让若干辆车先通过，则成功
        if self._yielding_observed and len(self._passed_ids) >= self.required_pass_count:
            self.actual_value = 1
            self.yield_status = "SUCCESS"
            return py_trees.common.Status.SUCCESS

        return new_status


class RoundaboutSafePassCriterion(Criterion):
    """
    检测 ego 在完成减速/让行后，是否恢复通行并安全到达终点附近
    """
    def __init__(
        self,
        actor,
        goal_location,
        name="RoundaboutSafePassCriterion",
        goal_dist_threshold=6.0,
        min_resume_speed=2.0,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)

        self.goal_location = goal_location
        self.goal_dist_threshold = goal_dist_threshold
        self.min_resume_speed = min_resume_speed

        self.actual_value = 0
        self.success_value = 1
        self.safe_pass_status = "INIT"

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if not self.actor:
            return new_status

        ego_loc = self.actor.get_location()
        ego_speed = get_speed(self.actor)

        dist_to_goal = ego_loc.distance(self.goal_location)

        if dist_to_goal <= self.goal_dist_threshold and ego_speed >= self.min_resume_speed:
            self.actual_value = 1
            self.safe_pass_status = "SUCCESS"
            return py_trees.common.Status.SUCCESS

        return new_status

# ==========================Four students crossing the road criterion==========================
class ScooterDecelerateCriterion(Criterion):
    """Check whether the ego starts a stable slowdown when approaching the occluding scooter."""

    def __init__(self, actor, scooter_actor, name="ScooterDecelerateCriterion",
                 route_start_location=None, route_end_location=None,
                 trigger_distance=35.0, min_speed_drop=3.0, brake_threshold=0.2,
                 min_brake_duration=0.4, pass_buffer=2.0, latest_reaction_distance=1.0,
                 terminate_on_failure=False):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.scooter_actor = scooter_actor
        self.route_start_location = route_start_location
        self.route_end_location = route_end_location
        self.trigger_distance = trigger_distance
        self.min_speed_drop = min_speed_drop
        self.brake_threshold = brake_threshold
        self.min_brake_duration = min_brake_duration
        self.pass_buffer = pass_buffer
        self.latest_reaction_distance = latest_reaction_distance

        fallback_transform = scooter_actor.get_transform() if scooter_actor is not None else None
        self._forward_xy, self._right_xy = _build_route_frame(
            route_start_location, route_end_location, fallback_transform=fallback_transform)

        self._activated = False
        self._initial_speed = None
        self._reaction_start_time = None
        self._has_valid_slowdown = False
        self._first_meaningful_slowdown_distance = None
        self._slowdown_confirmation_drop = max(0.8, self.min_speed_drop * 0.5)

        self.actual_value = 0.0
        self.success_value = min_speed_drop
        self.units = "m/s"
        self.test_status = "FAILURE"
        self.brake_status = "INIT"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor or not self.scooter_actor:
            return new_status

        if self.test_status == "SUCCESS":
            return new_status

        ego_loc = self.actor.get_location()
        scooter_loc = self.scooter_actor.get_location()
        longitudinal_dist, _ = _project_to_axis(
            ego_loc, scooter_loc, self._forward_xy, self._right_xy)
        ego_speed = _get_actor_speed_mps(self.actor)
        ego_control = self.actor.get_control()

        if os.environ.get("CVCI_DEBUG_CRAZYBIKE_CRITERIA", "0") == "1":
            print("CVCI_CRAZYBIKE_DECEL dist={:.3f} speed={:.3f} brake={:.3f} activated={} actual={:.3f} status={}".format(
                longitudinal_dist, ego_speed, float(getattr(ego_control, "brake", 0.0)), self._activated, self.actual_value, self.test_status
            ))

        if not self._activated and 0.0 <= longitudinal_dist <= self.trigger_distance:
            self._activated = True
            self._initial_speed = ego_speed
            self._reaction_start_time = None
            self._first_meaningful_slowdown_distance = None

        if not self._activated:
            return new_status

        speed_drop = max(0.0, self._initial_speed - ego_speed)
        self.actual_value = max(self.actual_value, speed_drop)

        brake_applied = ego_control.brake >= self.brake_threshold
        meaningful_slowdown = (
            speed_drop >= self._slowdown_confirmation_drop or
            (brake_applied and speed_drop >= max(0.4, self.min_speed_drop * 0.2))
        )
        if meaningful_slowdown and self._first_meaningful_slowdown_distance is None:
            self._first_meaningful_slowdown_distance = longitudinal_dist

        reaction_detected = speed_drop >= self.min_speed_drop

        control = self.actor.get_control()


        if control.brake >= self.brake_threshold:
            if reaction_detected :
                current_time = GameTime.get_time()
                if self._reaction_start_time is None:
                    self._reaction_start_time = current_time
                if current_time - self._reaction_start_time >= self.min_brake_duration:
                    self.test_status = "SUCCESS"
                    self.brake_status = "SUCCESS"


            else:
                self._reaction_start_time = None

        if longitudinal_dist < -self.pass_buffer and not self._has_valid_slowdown:
            self.test_status = "FAILURE"
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE

        return new_status

class PedestrianStopCriterion(Criterion):
    """Check whether the ego slows to a safe low speed for the first emerging child."""

    def __init__(self, actor, pedestrian_actor, name="PedestrianStopCriterion",
                 route_start_location=None, route_end_location=None,
                 trigger_distance=25.0, stop_speed_threshold=1.5,
                 activation_lateral_margin=2.8, pass_buffer=1.0, minimum_stop_distance=4.5,
                 min_speed_drop=1.0, min_stop_duration=0.3, clear_lateral_margin=0.6,
                 terminate_on_failure=False):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.pedestrian_actor = pedestrian_actor
        self.route_start_location = route_start_location
        self.route_end_location = route_end_location
        self.trigger_distance = trigger_distance
        self.stop_speed_threshold = stop_speed_threshold
        self.activation_lateral_margin = activation_lateral_margin
        self.pass_buffer = pass_buffer
        self.minimum_stop_distance = minimum_stop_distance
        self.min_speed_drop = min_speed_drop
        self.min_stop_duration = min_stop_duration
        self.clear_lateral_margin = clear_lateral_margin

        fallback_transform = actor.get_transform() if actor is not None else None
        self._forward_xy, self._right_xy = _build_route_frame(
            route_start_location, route_end_location, fallback_transform=fallback_transform)
        self._route_origin = route_start_location or (
            actor.get_location() if actor is not None else None)

        self._activated = False
        self._has_valid_stop = False
        self._activation_speed = None
        self._stop_start_time = None
        self._first_meaningful_slowdown_distance = None
        self._collision_sensor = None
        self._pedestrian_collision = False
        self.actual_value = float("inf")
        self.success_value = stop_speed_threshold
        self.units = "m/s"
        self.test_status = "FAILURE"

    def initialise(self):
        world = CarlaDataProvider.get_world()
        if world is not None and self.actor is not None:
            blueprint = world.get_blueprint_library().find('sensor.other.collision')
            self._collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
            self._collision_sensor.listen(lambda event: self._on_collision(event))
        super().initialise()

    def terminate(self, new_status):
        if self._collision_sensor is not None:
            try:
                if getattr(self._collision_sensor, "is_listening", False):
                    self._collision_sensor.stop()
            except (RuntimeError, AttributeError):
                pass
            try:
                self._collision_sensor.destroy()
            except RuntimeError:
                pass
        self._collision_sensor = None
        super().terminate(new_status)

    def _on_collision(self, event):
        if event is None or event.other_actor is None:
            return
        if 'walker.pedestrian' not in event.other_actor.type_id:
            return
        self._pedestrian_collision = True
        self.test_status = "FAILURE"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor or not self.pedestrian_actor or self._route_origin is None:
            return new_status

        if self._pedestrian_collision:
            self.test_status = "FAILURE"
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE
            return new_status

        if self.test_status == "SUCCESS":
            return new_status

        ego_loc = self.actor.get_location()
        ped_loc = self.pedestrian_actor.get_location()
        longitudinal_dist, _ = _project_to_axis(
            ego_loc, ped_loc, self._forward_xy, self._right_xy)
        _, ped_lateral_error = _project_to_axis(
            self._route_origin, ped_loc, self._forward_xy, self._right_xy)

        ego_speed = _get_actor_speed_mps(self.actor)
        self.actual_value = min(self.actual_value, ego_speed)

        pedestrian_in_conflict_zone = ped_lateral_error <= self.activation_lateral_margin
        if not self._activated and 0.0 <= longitudinal_dist <= self.trigger_distance and pedestrian_in_conflict_zone:
            self._activated = True
            self._activation_speed = ego_speed
            self._stop_start_time = None
            self._first_meaningful_slowdown_distance = None

        if os.environ.get("CVCI_DEBUG_GHOSTPROBE_CRITERIA", "0") == "1":
            activation_speed = -1.0 if self._activation_speed is None else self._activation_speed
            debug_drop = max(0.0, activation_speed - ego_speed) if self._activation_speed is not None else 0.0
            print(
                "CVCI_GHOSTPROBE_PED_STOP dist={:.3f} ped_lat={:.3f} speed={:.3f} "
                "activated={} activation_speed={:.3f} drop={:.3f} stop_start={} "
                "valid_stop={} status={}".format(
                    longitudinal_dist,
                    ped_lateral_error,
                    ego_speed,
                    self._activated,
                    activation_speed,
                    debug_drop,
                    self._stop_start_time is not None,
                    self._has_valid_stop,
                    self.test_status,
                )
            )

        if not self._activated:
            return new_status

        speed_drop = max(0.0, (self._activation_speed or ego_speed) - ego_speed)
        if speed_drop >= self.min_speed_drop and self._first_meaningful_slowdown_distance is None:
            self._first_meaningful_slowdown_distance = longitudinal_dist

        latest_slowdown_start_distance = max(
            self.minimum_stop_distance + 2.0,
            min(self.trigger_distance * 0.5, self.trigger_distance - 2.0)
        )
        if longitudinal_dist <= latest_slowdown_start_distance and self._first_meaningful_slowdown_distance is None:
            self.test_status = "FAILURE"
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE
            return new_status

        valid_low_speed = (
            ego_speed <= self.stop_speed_threshold and
            speed_drop >= self.min_speed_drop and
            longitudinal_dist > self.minimum_stop_distance
        )
        if valid_low_speed:
            current_time = GameTime.get_time()
            if self._stop_start_time is None:
                self._stop_start_time = current_time
            elif current_time - self._stop_start_time >= self.min_stop_duration:
                self._has_valid_stop = True
        else:
            self._stop_start_time = None

        if self._has_valid_stop and ped_lateral_error <= -self.clear_lateral_margin and not self._pedestrian_collision:
            self.test_status = "SUCCESS"
            return new_status

        if longitudinal_dist < -self.pass_buffer:
            self.test_status = "FAILURE"
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE

        return new_status


class PedestrianResumeCriterion(Criterion):
    """Check whether the ego restarts only after all children have safely crossed."""

    def __init__(self, actor, pedestrian_actor, name="PedestrianResumeCriterion",
                 route_start_location=None, route_end_location=None,
                 pedestrian_actors=None, resume_speed=3.0, stop_speed_threshold=0.5,
                 safe_lateral_margin=1.5, min_resume_duration=1.0,
                 unsafe_resume_speed=1.0, collision_buffer=1.2,
                 terminate_on_failure=False):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.pedestrian_actor = pedestrian_actor
        self.pedestrian_actors = pedestrian_actors if pedestrian_actors is not None else [pedestrian_actor]
        self.route_start_location = route_start_location
        self.route_end_location = route_end_location
        self.resume_speed = resume_speed
        self.stop_speed_threshold = stop_speed_threshold
        self.safe_lateral_margin = safe_lateral_margin
        self.min_resume_duration = min_resume_duration
        self.unsafe_resume_speed = unsafe_resume_speed
        self.collision_buffer = collision_buffer

        fallback_transform = actor.get_transform() if actor is not None else None
        self._forward_xy, self._right_xy = _build_route_frame(
            route_start_location, route_end_location, fallback_transform=fallback_transform)
        self._route_origin = route_start_location or (
            actor.get_location() if actor is not None else None)

        self._has_stopped = False
        self._resume_start_time = None

        self.actual_value = 0.0
        self.success_value = resume_speed
        self.units = "m/s"
        self.test_status = "FAILURE"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor or self._route_origin is None:
            return new_status

        if self.test_status == "SUCCESS":
            return new_status

        ego_loc = self.actor.get_location()
        ego_speed = _get_actor_speed_mps(self.actor)
        self.actual_value = ego_speed
        ego_longitudinal, _ = _project_to_axis(
            self._route_origin, ego_loc, self._forward_xy, self._right_xy)

        pedestrian_states = []
        pedestrian_behind_states = []
        min_distance_to_any = float("inf")
        for pedestrian in self.pedestrian_actors:
            if pedestrian is None:
                continue
            ped_loc = pedestrian.get_location()
            ped_longitudinal, ped_lateral_error = _project_to_axis(
                self._route_origin, ped_loc, self._forward_xy, self._right_xy)
            pedestrian_states.append(ped_lateral_error <= -self.safe_lateral_margin)
            pedestrian_behind_states.append(ped_longitudinal < ego_longitudinal - self.collision_buffer)
            min_distance_to_any = min(min_distance_to_any, ego_loc.distance(ped_loc))

        all_pedestrians_safe = len(pedestrian_states) > 0 and all(pedestrian_states)
        all_pedestrians_behind = len(pedestrian_behind_states) > 0 and all(pedestrian_behind_states)

        if ego_speed <= self.stop_speed_threshold and not all_pedestrians_safe:
            self._has_stopped = True

        unsafe_resume = self._has_stopped and not all_pedestrians_safe and ego_speed >= self.unsafe_resume_speed
        unsafe_clearance = ego_speed >= self.unsafe_resume_speed and min_distance_to_any <= self.collision_buffer
        if unsafe_resume or unsafe_clearance:
            self.test_status = "FAILURE"
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE
            return new_status

        if all_pedestrians_safe and not self._has_stopped:
            # If the ego has already passed all children with safe clearance, allow success.
            if all_pedestrians_behind and ego_speed >= self.resume_speed and min_distance_to_any > self.collision_buffer:
                current_time = GameTime.get_time()
                if self._resume_start_time is None:
                    self._resume_start_time = current_time
                elif current_time - self._resume_start_time >= self.min_resume_duration:
                    self.test_status = "SUCCESS"
            else:
                self._resume_start_time = None
            return new_status

        if self._has_stopped and all_pedestrians_safe and ego_speed >= self.resume_speed \
                and min_distance_to_any > self.collision_buffer:
            current_time = GameTime.get_time()
            if self._resume_start_time is None:
                self._resume_start_time = current_time
            elif current_time - self._resume_start_time >= self.min_resume_duration:
                self.test_status = "SUCCESS"
        else:
            self._resume_start_time = None

        return new_status
# ==========================avoid a disabled vehicle criterion==========================
class BrokenDownVehicleBrakeCriterion(Criterion):

    def __init__(self, actor, hazard_actor, name="BrakeCriterion", 
                 trigger_y=85.0, brake_threshold=0.2, 
                 min_brake_duration=0.3, max_response_time=10.0):
        
        # 必须显式调用父类，且参数顺序为 (name, actor, terminate_on_failure)
        super(BrokenDownVehicleBrakeCriterion, self).__init__(name, actor, terminate_on_failure=False)
        
        self.hazard_actor = hazard_actor
        self.trigger_y = trigger_y
        self.brake_threshold = brake_threshold
        self.min_brake_duration = min_brake_duration
        self.max_response_time = max_response_time
        
        self._activated = False
        self._start_time = None
        self._brake_start_time = None
        self.test_status = "FAILURE"
        
        # 初始状态设为 RUNNING，确保它能被持续 Tick
        self.status = py_trees.common.Status.RUNNING

    def update(self):

        ego_loc = self.actor.get_location()
        current_time = GameTime.get_time()

        # 触发逻辑
        if not self._activated and ego_loc.y <= self.trigger_y:
            self._activated = True

        if self._activated and self.test_status != "SUCCESS":
            control = self.actor.get_control()
            if control.brake >= self.brake_threshold:
                if self._brake_start_time is None:
                    self._brake_start_time = current_time
                
                # 判定成功
                if (current_time - self._brake_start_time) >= self.min_brake_duration:
                    self.test_status = "SUCCESS" # 记录结果，但不 return SUCCESS
            else:
                self._brake_start_time = None

        # 始终返回 RUNNING，确保它能一直 Tick 到底，直到 Parallel 节点整体结束
        return py_trees.common.Status.RUNNING

class BrokenDownVehicleBypassCriterion(Criterion):
    """
    判定自车是否成功绕行：
    道路中心线：y = -12.0
    行驶方向：x 轴负方向 (x 减小的方向)
    """
    def __init__(self, actor, hazard_actor, name="BypassCriterion", route_center_y=-12.0, lateral_threshold=1.5, pass_x_margin=3.0):
        super(BrokenDownVehicleBypassCriterion, self).__init__(name, actor)
        self.hazard_actor = hazard_actor
        self.route_center_y = route_center_y  # 现在中心线是 y
        self.lateral_threshold = lateral_threshold
        self.pass_x_margin = pass_x_margin    # 绕过的距离判定改为 x 轴
        self._has_departed = False
        
        self.test_status = "FAILURE"

    def update(self):
        ego_loc = self.actor.get_location()
        hazard_loc = self.hazard_actor.get_location()
        
        # 1. 横向偏移：现在是 y 轴方向的偏离
        lateral_offset = ego_loc.y - self.route_center_y
        
        # 2. 判定是否开始绕行（偏离 y = -12 中心线）
        if not self._has_departed:
            if abs(lateral_offset) >= self.lateral_threshold:
                self._has_departed = True

        # 3. 判定是否成功绕过
        # 沿 x 负方向行驶，成功意味着：自车的 x < (障碍物的 x - 裕度)
        if self._has_departed and self.test_status != "SUCCESS":
            if ego_loc.x < (hazard_loc.x - self.pass_x_margin):
                self.test_status = "SUCCESS" 

        # 4. 失败判定：如果自车 x 已经越过障碍物但还没偏离中心线
        # 注意：沿 x 负方向，“越过”意味着 ego.x < hazard.x
        if not self._has_departed and ego_loc.x < hazard_loc.x:
            self.test_status = "FAILURE"
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

class BrokenDownVehicleResumeCriterion(Criterion):
    """
    判定自车是否回归中心并到达终点：
    支持多个道路中心线：y = -12.0 或 y = -15.5
    行驶方向：沿 x 轴负方向
    """
    def __init__(self, actor, goal_location, name="ResumeCriterion", 
                 route_centers=[-12.0, -15.5],  # 修改为列表，支持多个中心线
                 goal_dist_threshold=5.0, 
                 center_recover_threshold=2.0, 
                 min_resume_speed=1.0):
        super(BrokenDownVehicleResumeCriterion, self).__init__(name, actor)
        self.goal_location = goal_location
        self.route_centers = route_centers  # 存储所有合法的中心线
        self.goal_dist_threshold = goal_dist_threshold
        self.center_recover_threshold = center_recover_threshold
        self.min_resume_speed = min_resume_speed
        
        self.test_status = "FAILURE"

    def update(self):
        ego_loc = self.actor.get_location()
        vel = self.actor.get_velocity()
        
        # 计算 3D 速度
        ego_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        
        # 实时监控进度
        dist_to_goal = ego_loc.distance(self.goal_location)

        # 【核心逻辑修改】：判断当前 y 坐标是否靠近任何一个预设的中心线
        # 只要有一个中心线的偏移量小于阈值，is_on_center 就会为 True
        is_on_center = any(abs(ego_loc.y - target_y) <= self.center_recover_threshold 
                           for target_y in self.route_centers)

        # 判定逻辑
        if dist_to_goal <= self.goal_dist_threshold:
            # 必须同时满足：1. 在任意一条中心线附近 2. 还有速度
            if is_on_center and ego_speed >= self.min_resume_speed:
                self.test_status = "SUCCESS" 
                return py_trees.common.Status.SUCCESS
            else:
                self.test_status = "FAILURE"
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING
    
# ==========================Slanted motor and children criterion==========================
def get_actor_speed(actor):
    """获取actor的速度（m/s）"""
    velocity = actor.get_velocity()
    return (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
class EbikeDetectionAndDecelerateCriterion(Criterion):
    """
    识别电瓶车并减速：
    当电瓶车开始移动后，ego是否输出明确减速信号（刹车或降低油门）
    评分：25分
    """
    def __init__(
        self,
        actor,
        hazard_actor,
        name="EbikeDetectionAndDecelerateCriterion",
        brake_threshold=0.15,
        throttle_reduction=0.3,
        min_decelerate_duration=0.5,
        max_response_time=5.0,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)

        self.hazard_actor = hazard_actor
        self.brake_threshold = brake_threshold
        self.throttle_reduction = throttle_reduction
        self.min_decelerate_duration = min_decelerate_duration
        self.max_response_time = max_response_time

        self._activated = False
        self._start_time = None
        self._decelerate_start_time = None
        self._initial_throttle = None
        self._completed = False
        self._has_failed = False

        # 永久性公共属性
        self.was_activated = False
        self.was_decelerated = False

        self.test_status = "INIT"
        self.ebick_status = 'INIT'
        self.actual_value = 0
        self.success_value = 1

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if self._has_failed:
            return new_status

        if self._completed:
            if self.ebick_status == "SUCCESS":
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE

        if not self.actor or not self.hazard_actor:
            return new_status

        if self._initial_throttle is None:
            control = self.actor.get_control()
            self._initial_throttle = control.throttle

        hazard_velocity = self.hazard_actor.get_velocity()
        hazard_speed = math.sqrt(hazard_velocity.x**2 + hazard_velocity.y**2 + hazard_velocity.z**2)

        if not self._activated and hazard_speed > 0.5:
            self._activated = True
            self.was_activated = True
            self._start_time = GameTime.get_time()
            print(f"[电瓶车检测] 电瓶车启动，开始检测")

        if not self._activated:
            return new_status

        control = self.actor.get_control()
        current_time = GameTime.get_time()

        is_decelerating = (
            control.brake >= self.brake_threshold or
            (self._initial_throttle - control.throttle) >= self.throttle_reduction
        )

        if is_decelerating:
            if self._decelerate_start_time is None:
                self._decelerate_start_time = current_time
            decelerate_duration = current_time - self._decelerate_start_time
            if decelerate_duration >= self.min_decelerate_duration:
                self.ebick_status = "SUCCESS"
                self.actual_value = 1
                self._completed = True
                self.was_decelerated = True
                print(f"[电瓶车检测] ✅ 成功减速")
                return py_trees.common.Status.SUCCESS
        else:
            self._decelerate_start_time = None

        if current_time - self._start_time > self.max_response_time:
            if not self._has_failed:
                self._has_failed = True
                self.ebick_status = "FAILURE"
                self.actual_value = 0
                self.was_decelerated = False
                print(f"[电瓶车检测] ❌ 超时未减速（不给分，继续运行）")
            return new_status

        return new_status

class PedestrianDetectionAndStopCriterion(Criterion):
    """
    识别行人并刹车：
    当行人距离 ≤ trigger_distance 时，ego必须刹车并完全停止
    评分：50分
    """
    def __init__(
        self,
        actor,
        hazard_actor,
        name="PedestrianDetectionAndStopCriterion",
        trigger_distance=10.0,
        brake_threshold=0.05,
        stop_speed_threshold=0.3,
        min_stop_duration=1.0,
        max_response_time=5.0,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)

        self.hazard_actor = hazard_actor
        self.trigger_distance = trigger_distance
        self.brake_threshold = brake_threshold
        self.stop_speed_threshold = stop_speed_threshold
        self.min_stop_duration = min_stop_duration
        self.max_response_time = max_response_time

        self._activated = False
        self._start_time = None
        self._stop_start_time = None
        self._has_braked = False
        self._is_stopped = False
        self._completed = False
        self._has_failed = False

        # 永久性公共属性
        self.was_activated = False
        self.was_braked = False
        self.was_stopped = False

        self.test_status = "RUNNING"
        self.pedestrain_status = "INIT"
        self.actual_value = 0
        self.success_value = 1

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if self._has_failed:
            return new_status

        if self._completed:
            if self.pedestrain_status == "SUCCESS":
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE

        if not self.actor or not self.hazard_actor:
            return new_status

        ego_loc = CarlaDataProvider.get_location(self.actor)
        ped_loc = CarlaDataProvider.get_location(self.hazard_actor)
        
        if ego_loc is None or ped_loc is None:
            return new_status
            
        distance = ego_loc.distance(ped_loc)
        current_time = GameTime.get_time()

        # 阶段1：距离触发
        if not self._activated:
            if distance <= self.trigger_distance:
                self._activated = True
                self.was_activated = True
                self._start_time = current_time
                print(f"[行人检测] 🚶 触发！距离={distance:.1f}m")
            else:
                return new_status

        control = self.actor.get_control()
        ego_speed = get_actor_speed(self.actor)
        elapsed_time = current_time - self._start_time

        print(f"[行人检测] 距离={distance:.1f}m, 已刹车={self._has_braked}, 已停止={self._is_stopped}, 已用时={elapsed_time:.1f}s")

        is_braking = (control.brake >= self.brake_threshold)
        is_slow_enough = (ego_speed <= self.stop_speed_threshold)

        # 阶段2：必须踩刹车
        if not self._has_braked:
            if is_braking:
                self._has_braked = True
                self.was_braked = True
                print(f"[行人检测] ✅ 踩了刹车！刹车={control.brake:.2f}")
            else:
                if elapsed_time > self.max_response_time:
                    if not self._has_failed:
                        self._has_failed = True
                        self.pedestrain_status = "FAILURE"
                        self.actual_value = 0
                        self.was_braked = False
                        self.was_stopped = False
                        print(f"[行人检测] ❌ 超时 {elapsed_time:.1f}秒未刹车（不给分，继续运行）")
                    return new_status
            return new_status

        # 阶段3：必须完全停止
        if not self._is_stopped:
            if is_slow_enough:
                if self._stop_start_time is None:
                    self._stop_start_time = current_time
                    print(f"[行人检测] 开始停止，速度={ego_speed:.2f}m/s")
                stop_duration = current_time - self._stop_start_time
                if stop_duration >= self.min_stop_duration:
                    self._is_stopped = True
                    self.was_stopped = True
                    self.pedestrain_status = "SUCCESS"
                    self.actual_value = 1
                    self._completed = True
                    print(f"[行人检测] ✅ 已停止 {stop_duration:.1f}秒")
                    return py_trees.common.Status.SUCCESS
            else:
                self._stop_start_time = None
            return new_status

        # 如果已经停止但没有返回成功，这里再返回一次
        if self._is_stopped and not self._completed:
            self.was_stopped = True
            self.pedestrain_status = "SUCCESS"
            self.actual_value = 1
            self._completed = True
            print(f"[行人检测] ✅ 最终成功")
            return py_trees.common.Status.SUCCESS

        return new_status

class ResumeAfterPedestrianCriterion(Criterion):
    """
    离开风险区并恢复通行：
    自车能够安全起步离开
    评分：25分
    """
    def __init__(
        self,
        actor,
        hazard_actor,
        goal_location,
        name="ResumeAfterPedestrianCriterion",
        goal_dist_threshold=50.0,
        min_resume_speed=0.0,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)

        self.hazard_actor = hazard_actor
        self.goal_location = goal_location
        self.goal_dist_threshold = goal_dist_threshold
        self.min_resume_speed = min_resume_speed

        self._completed = False
        
        # 永久性公共属性
        self.was_resumed = False

        self.test_status = "INIT"
        self.resume_status = "INIT"
        self.actual_value = 0
        self.success_value = 1

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if self._completed:
            if self.resume_status == "SUCCESS":
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE

        if not self.actor:
            return new_status

        ego_loc = self.actor.get_location()
        ego_speed = get_actor_speed(self.actor)
        dist_to_goal = ego_loc.distance(self.goal_location)

        if dist_to_goal <= self.goal_dist_threshold and ego_speed >= self.min_resume_speed:
            self.resume_status = "SUCCESS"
            self.actual_value = 1
            self._completed = True
            self.was_resumed = True
            print(f"[恢复通行] ✅ 到达终点")
            return py_trees.common.Status.SUCCESS

        return new_status

# ==========================reverse_vehicle private criterion==========================
class ReverseVehicleBrakeCriterion(Criterion):
    """
    识别并减速：
    当障碍车开始侵入风险区后，ego 是否输出明确制动信号
    """
    def __init__(
        self,
        actor,
        hazard_actor,
        name="ReverseVehicleBrakeCriterion",
        trigger_x=60.0,
        brake_threshold=0.2,
        min_brake_duration=0.3,
        max_response_time=4.0,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)

        self.hazard_actor = hazard_actor
        self.trigger_x = trigger_x
        self.brake_threshold = brake_threshold
        self.min_brake_duration = min_brake_duration
        self.max_response_time = max_response_time

        self._activated = False
        self._start_time = None
        self._brake_start_time = None

        self.brake_status = "INIT"
        self.actual_value = 0
        self.success_value = 1

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if not self.actor or not self.hazard_actor:
            return new_status

        # 障碍车进入主车道/风险区后开始检测
        ego_loc = self.actor.get_location()
        if not self._activated and ego_loc.x <= self.trigger_x:
            self._activated = True
            self._start_time = GameTime.get_time()

        if not self._activated:
            return new_status

        control = self.actor.get_control()
        current_time = GameTime.get_time()

        # brake 达到阈值，开始累计持续时间
        # print("刹车信号{}".format(control.brake))
        if control.brake >= self.brake_threshold:
            if self._brake_start_time is None:
                self._brake_start_time = current_time
            brake_duration = current_time - self._brake_start_time
            if brake_duration >= self.min_brake_duration:
                # print("刹车检测通过")
                self.brake_status = "SUCCESS"
                self.actual_value = 1
                return py_trees.common.Status.SUCCESS
        else:
            # brake 中断则重新计时
            self._brake_start_time = None

        return new_status      

class ReverseVehicleBypassCriterion(Criterion):
    """
    进行安全绕行：
    障碍车占道后，自车是否明确选择左/右绕行并完成通过
    """
    def __init__(
        self,
        actor,
        hazard_actor,
        route_center_y=134.423233,
        name="ReverseVehicleBypassCriterion",
        lateral_threshold=2.0,
        pass_x_margin=3.0,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.hazard_actor = hazard_actor
        self.route_center_y = route_center_y
        self.lateral_threshold = lateral_threshold
        self.pass_x_margin = pass_x_margin

        self._direction = None   # "left" / "right"
        self._has_departed_lane_center = False
        self.bypass_status = "INIT"

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if not self.actor or not self.hazard_actor:
            return new_status

        ego_loc = self.actor.get_location()
        hazard_loc = self.hazard_actor.get_location()

        lateral_offset = ego_loc.y - self.route_center_y

        # 先判断是否出现明确绕行动作
        # print("自车向左绕行，自车位置为{}, lateral_offset为{}, lateral_threshold为{}".format(ego_loc, lateral_offset, self.lateral_threshold))
        if not self._has_departed_lane_center:
            if lateral_offset >= self.lateral_threshold:
                
                self._has_departed_lane_center = True
                self._direction = "left"
            elif lateral_offset <= -self.lateral_threshold:
                self._has_departed_lane_center = True
                self._direction = "right"

        # 再判断是否已经从障碍物前方通过
        if self._has_departed_lane_center:
            if ego_loc.x > hazard_loc.x + self.pass_x_margin:
                self.bypass_status = "SUCCESS"
                return py_trees.common.Status.SUCCESS

        return new_status

class ReverseVehicleResumeCriterion(Criterion):
    """
    离开风险区并恢复通行：
    自车绕行后回归路线，并到达终点附近
    """
    def __init__(
        self,
        actor,
        goal_location,
        route_center_y = 134.423233,
        name="ReverseVehicleResumeCriterion",
        goal_dist_threshold=5.0,
        center_recover_threshold=2.0,
        min_resume_speed=1.0,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.goal_location = goal_location
        self.route_center_y = route_center_y
        self.goal_dist_threshold = goal_dist_threshold
        self.center_recover_threshold = center_recover_threshold
        self.min_resume_speed = min_resume_speed

        self.resume_status = "INIT"

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if not self.actor:
            return new_status

        ego_loc = self.actor.get_location()
        ego_speed = get_speed(self.actor)

        dist_to_goal = ego_loc.distance(self.goal_location)
        center_offset = abs(ego_loc.y - self.route_center_y)
        # print("横向误差：{}".format(center_offset))
        if (
            dist_to_goal <= self.goal_dist_threshold
            and center_offset <= self.center_recover_threshold
            and ego_speed >= self.min_resume_speed
        ):
            self.resume_status = "SUCCESS"
            return py_trees.common.Status.SUCCESS

        return new_status
    
# ==========================crazy motor criterion==========================
class CrazyBikeDecelerateCriterion(Criterion):
    """Check whether ego starts deceleration before getting too close to the cutting-in bike."""

    def __init__(self, actor, bike_actor, name="CrazyBikeDecelerateCriterion",
                 route_start_location=None, route_end_location=None,
                 trigger_distance=24.0, latest_reaction_distance=10.0,
                 min_speed_drop=2.0, brake_threshold=0.15,
                 min_brake_duration=0.3, pass_buffer=1.5,
                 terminate_on_failure=False):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.bike_actor = bike_actor
        self.route_start_location = route_start_location
        self.route_end_location = route_end_location
        self.trigger_distance = trigger_distance
        self.latest_reaction_distance = latest_reaction_distance
        self.min_speed_drop = min_speed_drop
        self.brake_threshold = brake_threshold
        self.min_brake_duration = min_brake_duration
        self.pass_buffer = pass_buffer

        fallback_transform = bike_actor.get_transform() if bike_actor is not None else None
        self._forward_xy, self._right_xy = _build_route_frame(
            route_start_location, route_end_location, fallback_transform=fallback_transform)

        self._activated = False
        self._activation_speed = None
        self._reaction_start_time = None
        self._has_valid_deceleration = False

        self.actual_value = 0.0
        self.success_value = min_speed_drop
        self.units = "m/s"
        self.test_status = "FAILURE"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor or not self.bike_actor:
            return new_status

        if self.test_status == "SUCCESS":
            return new_status

        ego_loc = self.actor.get_location()
        bike_loc = self.bike_actor.get_location()
        longitudinal_dist, _ = _project_to_axis(
            ego_loc, bike_loc, self._forward_xy, self._right_xy)
        ego_speed = _get_actor_speed_mps(self.actor)
        ego_control = self.actor.get_control()

        if os.environ.get("CVCI_DEBUG_CRAZYBIKE_CRITERIA", "0") == "1":
            print("CVCI_CRAZYBIKE_DECEL dist={:.3f} speed={:.3f} brake={:.3f} activated={} activation_speed={} actual={:.3f} status={}".format(
                longitudinal_dist, ego_speed, float(getattr(ego_control, "brake", 0.0)), self._activated, self._activation_speed, self.actual_value, self.test_status
            ))

        if not self._activated and 0.0 <= longitudinal_dist <= self.trigger_distance:
            self._activated = True
            self._activation_speed = ego_speed
            self._reaction_start_time = None

        if not self._activated:
            return new_status

        speed_drop = max(0.0, (self._activation_speed or ego_speed) - ego_speed)
        self.actual_value = max(self.actual_value, speed_drop)

        reaction_detected = (
            speed_drop >= self.min_speed_drop or
            ego_control.brake >= self.brake_threshold
        )

        if reaction_detected:
            current_time = GameTime.get_time()
            if self._reaction_start_time is None:
                self._reaction_start_time = current_time
            elif current_time - self._reaction_start_time >= self.min_brake_duration:
                self._has_valid_deceleration = True
                self.test_status = "SUCCESS"
        else:
            self._reaction_start_time = None

        if longitudinal_dist <= self.latest_reaction_distance and not self._has_valid_deceleration:
            self.test_status = "FAILURE"
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE
            return new_status

        if longitudinal_dist < -self.pass_buffer and not self._has_valid_deceleration:
            self.test_status = "FAILURE"
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE

        return new_status

class CrazyBikeNoCollisionCriterion(Criterion):
    """Check whether ego avoids colliding with the cutting-in bike."""

    def __init__(self, actor, bike_actor, name="CrazyBikeNoCollisionCriterion",
                 terminate_on_failure=False):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.bike_actor = bike_actor
        self._collision_sensor = None

        self.actual_value = 0
        self.success_value = 0
        self.units = "times"
        self.test_status = "SUCCESS"

    def initialise(self):
        world = CarlaDataProvider.get_world()
        if world is not None and self.actor is not None:
            blueprint = world.get_blueprint_library().find('sensor.other.collision')
            self._collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
            self._collision_sensor.listen(lambda event: self._on_collision(event))
        super().initialise()

    def terminate(self, new_status):
        if self._collision_sensor is not None:
            try:
                self._collision_sensor.destroy()
            except RuntimeError:
                pass
        self._collision_sensor = None
        super().terminate(new_status)

    def _on_collision(self, event):
        if event is None or event.other_actor is None:
            return
        if self.bike_actor is None:
            return
        if event.other_actor.id != self.bike_actor.id:
            return

        self.actual_value += 1
        self.test_status = "FAILURE"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if self.test_status != "FAILURE":
            self.test_status = "SUCCESS"
        elif self._terminate_on_failure:
            return py_trees.common.Status.FAILURE
        return new_status

class CrazyBikeResumeCriterion(Criterion):
    """Check whether ego safely resumes cruising after passing the bike."""

    def __init__(self, actor, bike_actor, name="CrazyBikeResumeCriterion",
                 route_start_location=None, route_end_location=None,
                 escape_distance=12.0, resume_speed=5.0,
                 min_resume_duration=1.0, lane_tolerance=1.8,
                 terminate_on_failure=False):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.bike_actor = bike_actor
        self.route_start_location = route_start_location
        self.route_end_location = route_end_location
        self.escape_distance = escape_distance
        self.resume_speed = resume_speed
        self.min_resume_duration = min_resume_duration
        self.lane_tolerance = lane_tolerance

        fallback_transform = actor.get_transform() if actor is not None else None
        self._forward_xy, self._right_xy = _build_route_frame(
            route_start_location, route_end_location, fallback_transform=fallback_transform)
        self._route_origin = route_start_location or (
            actor.get_location() if actor is not None else None)

        self._has_been_behind_bike = False
        self._resume_start_time = None
        self._collision_sensor = None
        self._bike_collision = False

        self.actual_value = 0.0
        self.success_value = resume_speed
        self.units = "m/s"
        self.test_status = "FAILURE"

    def initialise(self):
        world = CarlaDataProvider.get_world()
        if world is not None and self.actor is not None:
            blueprint = world.get_blueprint_library().find('sensor.other.collision')
            self._collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
            self._collision_sensor.listen(lambda event: self._on_collision(event))
        super().initialise()

    def terminate(self, new_status):
        if self._collision_sensor is not None:
            try:
                if getattr(self._collision_sensor, "is_listening", False):
                    self._collision_sensor.stop()
            except (RuntimeError, AttributeError):
                pass
            try:
                self._collision_sensor.destroy()
            except RuntimeError:
                pass
        self._collision_sensor = None
        super().terminate(new_status)

    def _on_collision(self, event):
        if event is None or event.other_actor is None:
            return
        if self.bike_actor is None:
            return
        if event.other_actor.id != self.bike_actor.id:
            return
        self._bike_collision = True
        self.test_status = "FAILURE"

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if not self.actor or not self.bike_actor:
            return new_status

        if self._bike_collision:
            self.test_status = "FAILURE"
            if self._terminate_on_failure:
                return py_trees.common.Status.FAILURE
            return new_status

        if self.test_status == "SUCCESS":
            return new_status

        ego_loc = self.actor.get_location()
        bike_loc = self.bike_actor.get_location()
        ego_speed = _get_actor_speed_mps(self.actor)
        self.actual_value = ego_speed

        longitudinal_from_bike, _ = _project_to_axis(
            bike_loc, ego_loc, self._forward_xy, self._right_xy)

        if os.environ.get("CVCI_DEBUG_CRAZYBIKE_CRITERIA", "0") == "1":
            print("CVCI_CRAZYBIKE_RESUME dist_from_bike={:.3f} speed={:.3f} behind={} status={}".format(
                longitudinal_from_bike, ego_speed, self._has_been_behind_bike, self.test_status
            ))

        if longitudinal_from_bike < 0.0:
            self._has_been_behind_bike = True

        if not self._has_been_behind_bike or longitudinal_from_bike <= self.escape_distance:
            self._resume_start_time = None
            return new_status

        if ego_speed >= self.resume_speed:
            current_time = GameTime.get_time()
            if self._resume_start_time is None:
                self._resume_start_time = current_time
            elif current_time - self._resume_start_time >= self.min_resume_duration:
                self.test_status = "SUCCESS"
        else:
            self._resume_start_time = None

        return new_status
# ==========================Blind spot hidden car criterion==========================
class IntersectionCollisionLeftTurnBrakeCriterion(Criterion):
    """
    检测减速行为：
    1. 当与障碍车距离小于 trigger_distance 时激活。
    2. 必须检测到主动刹车信号 (brake > threshold)。
    3. 速度减小量达到目标值。
    4. 排除掉因碰撞导致的异常降速。
    """
    def __init__(
        self,
        actor,
        hazard_actor,
        name="IntersectionCollisionLeftTurnBrakeCriterion",
        trigger_distance=40.0,
        target_speed_reduction=3.0,   # m/s
        brake_threshold=0.1,          # 刹车踏板阈值
        collision_speed_drop=2.0,     # 单帧降速超过此值判定为碰撞 (m/s per tick)
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)

        self.hazard_actor = hazard_actor
        self.trigger_distance = trigger_distance
        self.target_speed_reduction = target_speed_reduction
        self.brake_threshold = brake_threshold
        self.collision_speed_drop = collision_speed_drop

        self._activated = False
        self._initial_speed = None
        self._last_speed = None
        self._brake_detected = False
        self._collision_occurred = False

        self.test_status = "INIT"
        self.brake_status = "INIT"
        self.actual_value = 0.0
        self.success_value = target_speed_reduction

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if not self.actor or not self.hazard_actor:
            return new_status

        # 获取位置和速度
        ego_loc = self.actor.get_location()
        hazard_loc = self.hazard_actor.get_location()
        current_distance = ego_loc.distance(hazard_loc)
        
        vel = self.actor.get_velocity()
        current_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        # 1. 触发逻辑
        if not self._activated and current_distance <= self.trigger_distance:
            self._activated = True
            self._initial_speed = current_speed
            self._last_speed = current_speed
            print(f"[{self.name}] 激活：初始速度 {current_speed:.2f} m/s", flush=True)

        if not self._activated:
            return new_status

        # 2. 实时监测刹车信号和碰撞风险
        control = self.actor.get_control()
        
        # 记录是否踩过刹车
        if control.brake > self.brake_threshold:
            self._brake_detected = True

        # 检测异常降速（单帧降速过快通常是碰撞）
        if self._last_speed is not None:
            speed_delta = self._last_speed - current_speed
            if speed_delta > self.collision_speed_drop:
                self._collision_occurred = True
                print(f"[{self.name}] 警告：检测到疑似碰撞（瞬时降速 {speed_delta:.2f} m/s）", flush=True)

        # 3. 判定逻辑
        speed_reduction = self._initial_speed - current_speed
        self.actual_value = round(speed_reduction, 2)

        # 成功条件：降速足够 + 有刹车动作 + 无碰撞
        if speed_reduction >= self.target_speed_reduction:
            if self._brake_detected and not self._collision_occurred:
                print(f"[{self.name}] 成功：主动减速 {speed_reduction:.2f} m/s", flush=True)
                self.test_status = "SUCCESS"
                self.brake_status = "SUCCESS"
                return py_trees.common.Status.SUCCESS

        self._last_speed = current_speed
        return new_status

    def terminate(self, new_status):
        if self.test_status != "SUCCESS":
            self.test_status = "FAILURE"
        super().terminate(new_status)
class IntersectionCollisionLeftTurnResumeCriterion(Criterion):
    """
    离开风险区并恢复通行：
    自车减速后回归路线，并到达终点附近
    """
    def __init__(
        self,
        actor,
        goal_location,
        route_center_x=45.0,
        name="IntersectionCollisionLeftTurnResumeCriterion",
        goal_dist_threshold=3.0,
        center_recover_threshold=2.0,
        min_resume_speed=1.0,
        terminate_on_failure=False
    ):
        super().__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.goal_location = goal_location
        self.route_center_x = route_center_x
        self.goal_dist_threshold = goal_dist_threshold
        self.center_recover_threshold = center_recover_threshold
        self.min_resume_speed = min_resume_speed

        self.test_status = "INIT"
        self.safepass_status = "INIT"

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if not self.actor:
            return new_status

        ego_loc = self.actor.get_location()
        ego_speed = get_speed(self.actor)

        dist_to_goal = ego_loc.distance(self.goal_location)
        # center_offset = abs(ego_loc.x - self.route_center_x)

        if (
            dist_to_goal <= self.goal_dist_threshold
            and ego_loc.x >= self.route_center_x
            and ego_speed >= self.min_resume_speed
        ):
            self.test_status = "SUCCESS"
            self.safepass_status = "SUCCESS"
            return py_trees.common.Status.SUCCESS

        return new_status
