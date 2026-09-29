#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a statistics manager for the CARLA AD leaderboard
"""

from __future__ import print_function

from dictor import dictor
import math

from srunner.scenariomanager.traffic_events import TrafficEventType

from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict
from leaderboard.utils.facts_creator import (
    extract_common_facts,
    extract_private_facts_lane_closure,
    extract_private_facts_frontcar_disappearance,
    extract_private_facts_static_barrier,
    extract_private_facts_reverse_vehicle,
    extract_private_facts_high_speed_accident,
    extract_private_facts_ghost_probe,
    extract_private_facts_broken_down_vehicle,
    extract_private_facts_ebike_pedestrian_cross,
    extract_private_facts_left_turn,
    extract_private_facts_roundabout_merge_conflict,
    extract_private_facts_crazy_bike,
    extract_private_facts_high_speed_cutting
)
from leaderboard.utils.facts_to_score import (
    score_frontcar_disappear_accident,
    score_static_barrier,
    score_reverse_vehicle,
    compute_lane_closure_score,
    score_high_speed_accident,
    score_ghost_probe,
    score_broken_down_vehicle,
    score_ebike_pedestrian_cross,
    score_left_turn,
    score_roundabout_merge_conflict,
    score_crazy_bike,
    score_roundabout_merge_conflict, 
    score_high_speed_cutting
)

# 通过 PRIVATE_FACT_EXTRACTORS 和 SCENARIO_SCORERS 动态获取不同场景的 facts extractor 和 scorer
PRIVATE_FACT_EXTRACTORS = {
    "LaneClosureWithTruck": extract_private_facts_lane_closure,
    "CarDisappearDiagonalAccident": extract_private_facts_frontcar_disappearance,
    'CutInCollision': extract_private_facts_high_speed_cutting,
    "StaticBarrier": extract_private_facts_static_barrier,
    "HighSpeedAccident": extract_private_facts_high_speed_accident,
    "RoundaboutMergeConflict": extract_private_facts_roundabout_merge_conflict,
    "GhostProbeScenario": extract_private_facts_ghost_probe,
    "BrokenDownVehicle": extract_private_facts_broken_down_vehicle,
    "EbikeAndPedestrianCross": extract_private_facts_ebike_pedestrian_cross,
    "ReverseVehicle": extract_private_facts_reverse_vehicle,
    "IntersectionCollisionLeftTurn": extract_private_facts_left_turn,
    "CrazyBikeScenario": extract_private_facts_crazy_bike,
}

SCENARIO_SCORERS = {
    "LaneClosureWithTruck": compute_lane_closure_score,
    "CarDisappearDiagonalAccident": score_frontcar_disappear_accident,
    'CutInCollision': score_high_speed_cutting,
    "StaticBarrier": score_static_barrier,
    "HighSpeedAccident": score_high_speed_accident,
    "RoundaboutMergeConflict": score_roundabout_merge_conflict,
    "GhostProbeScenario": score_ghost_probe,
    "BrokenDownVehicle": score_broken_down_vehicle,
    "EbikeAndPedestrianCross": score_ebike_pedestrian_cross,
    "ReverseVehicle": score_reverse_vehicle,
    "IntersectionCollisionLeftTurn": score_left_turn,
    "CrazyBikeScenario": score_crazy_bike,
}


def extract_private_facts(criteria_list, scenario_type):
    """动态提取 private facts"""
    if scenario_type not in PRIVATE_FACT_EXTRACTORS:
        raise ValueError("Unsupported scenario_type: {}".format(scenario_type))
    extractor = PRIVATE_FACT_EXTRACTORS[scenario_type]
    return extractor(criteria_list)


def compute_scenario_score(common_facts, private_facts, scenario_type):
    """动态计算 challenge score"""
    if scenario_type not in SCENARIO_SCORERS:
        raise ValueError("Unsupported scenario_type: {}".format(scenario_type))
    scorer = SCENARIO_SCORERS[scenario_type]
    return scorer(common_facts, private_facts)


def evaluate_scenario(criteria_list, scenario_type):
    """先提取 facts，再计算分数。

    The 0629 challenge scorer only defines private facts for the frozen
    extreme-scenario families.  The official Bench2Drive 220 XML contains
    ordinary scenario types that intentionally do not belong to that scorer.
    Those routes must still receive their normal leaderboard statistics; an
    unsupported challenge type must not turn a completed route into an
    evaluator crash.
    """
    common_facts = extract_common_facts(criteria_list)
    if (
        scenario_type not in PRIVATE_FACT_EXTRACTORS
        or scenario_type not in SCENARIO_SCORERS
    ):
        print(
            "CVCI_CHALLENGE_SCORE skipped unsupported ordinary scenario_type={} "
            "(standard score preserved)".format(scenario_type)
        )
        return common_facts, {}, {
            "base_score": 0.0,
            "gate": 1.0,
            "penalty": common_facts.get("collision_penalty", 1.0),
            "final_score": 0.0,
            "scoring_mode": "standard_only_unsupported_scenario",
            "scenario_type": scenario_type,
        }
    private_facts = extract_private_facts(criteria_list, scenario_type)
    score_detail = compute_scenario_score(common_facts, private_facts, scenario_type)
    return common_facts, private_facts, score_detail


PENALTY_VALUE_DICT = {
    # Traffic events that subtract a set amount of points.
    TrafficEventType.COLLISION_PEDESTRIAN: 0.5,
    TrafficEventType.COLLISION_VEHICLE: 0.6,
    TrafficEventType.COLLISION_STATIC: 0.65,
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 0.7,
    TrafficEventType.STOP_INFRACTION: 0.8,
    TrafficEventType.SCENARIO_TIMEOUT: 0.7,
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 0.7
}

PENALTY_PERC_DICT = {
    # Traffic events that subtract a varying amount of points. This is the per unit value.
    # 'increases' means that the higher the value, the higher the penalty.
    # 'decreases' means that the ideal value is 100 and the lower the value, the higher the penalty.
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: [0, 'increases'],  # All route traversed through outside lanes is ignored
    # TrafficEventType.MIN_SPEED_INFRACTION: [0.7, 'decreases'],
    TrafficEventType.MIN_SPEED_INFRACTION: [0.7, 'unused'],
}

PENALTY_NAME_DICT = {
    TrafficEventType.COLLISION_STATIC: 'collisions_layout',
    TrafficEventType.COLLISION_PEDESTRIAN: 'collisions_pedestrian',
    TrafficEventType.COLLISION_VEHICLE: 'collisions_vehicle',
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 'red_light',
    TrafficEventType.STOP_INFRACTION: 'stop_infraction',
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: 'outside_route_lanes',
    TrafficEventType.MIN_SPEED_INFRACTION: 'min_speed_infractions',
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 'yield_emergency_vehicle_infractions',
    TrafficEventType.SCENARIO_TIMEOUT: 'scenario_timeouts',
    TrafficEventType.ROUTE_DEVIATION: 'route_dev',
    TrafficEventType.VEHICLE_BLOCKED: 'vehicle_blocked',
}

# Limit the entry status to some values. Eligible should always be gotten from this table
ENTRY_STATUS_VALUES = ['Started', 'Finished', 'Rejected', 'Crashed', 'Invalid']
ELIGIBLE_VALUES = {'Started': False, 'Finished': True, 'Rejected': False, 'Crashed': False, 'Invalid': False}

# Dictionary mapping a route failure with the 'entry status' and 'status'
FAILURE_MESSAGES = {
    "Simulation": ["Crashed", "Simulation crashed"],
    "Sensors": ["Rejected", "Agent's sensors were invalid"],
    "Agent_init": ["Started", "Agent couldn't be set up"],
    "Agent_runtime": ["Started", "Agent crashed"]
}

ROUND_DIGITS = 3
ROUND_DIGITS_SCORE = 6


class RouteRecord():
    def __init__(self):
        self.index = -1
        self.route_id = None
        self.scenario_name = None
        self.weather_id = None
        self.save_name = None
        self.town_name = None
        self.status = 'Started'
        self.num_infractions = 0

        self.infractions = {}
        for event_name in PENALTY_NAME_DICT.values():
            self.infractions[event_name] = []
        self.infractions['route_timeout'] = []

        self.scores = {
            'score_route': 0,
            'score_penalty': 0,
            'score_composed': 0,
            'score_challenge_base': 0,
            'score_challenge_gate': 0,
            'score_challenge_penalty': 0,
            'score_challenge': 0
        }

        self.common_facts = {}
        self.private_facts = {}
        self.score_detail = {}

        self.meta = {
            'route_length': 0,
            'duration_game': 0,
            'duration_system': 0,
            'scenario_type': ''
        }

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)


class GlobalRecord():
    def __init__(self):
        self.index = -1
        self.route_id = -1
        self.status = 'Perfect'

        self.infractions = {}
        for event_name in PENALTY_NAME_DICT.values():
            self.infractions[event_name] = 0
        self.infractions['route_timeout'] = 0

        self.scores_mean = {
            'score_composed': 0,
            'score_route': 0,
            'score_penalty': 0,
            'score_challenge_base': 0,
            'score_challenge_gate': 0,
            'score_challenge_penalty': 0,
            'score_challenge': 0
        }
        self.scores_std_dev = self.scores_mean.copy()

        self.meta = {
            "total_length": 0,
            "duration_game": 0,
            "duration_system": 0,
            'exceptions': [],
            'scenario_types': [],
            'challenge_common_facts': [],
            'challenge_private_facts': [],
            'challenge_score_breakdown': []
        }

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)


class Checkpoint():

    def __init__(self):
        self.global_record = {}
        self.progress = []
        self.records = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d['global_record'] = self.global_record.to_json() if self.global_record else {}
        d['progress'] = self.progress
        d['records'] = [x.to_json() for x in self.records if x.index != -1]  # Index -1 = Route in progress
        return d


class Results():

    def __init__(self):
        self.checkpoint = Checkpoint()
        self.entry_status = "Started"
        self.eligible = ELIGIBLE_VALUES[self.entry_status]
        self.sensors = []
        self.values = []
        self.labels = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d['_checkpoint'] = self.checkpoint.to_json()
        d['entry_status'] = self.entry_status
        d['eligible'] = self.eligible
        d['sensors'] = self.sensors
        d['values'] = self.values
        d['labels'] = self.labels
        return d


def to_route_record(record_dict):
    record = RouteRecord()
    for key, value in record_dict.items():
        setattr(record, key, value)
    return record


def compute_route_length(route):
    route_length = 0.0
    previous_location = None

    for transform, _ in route:
        location = transform.location
        if previous_location:
            dist_vec = location - previous_location
            route_length += dist_vec.length()
        previous_location = location

    return route_length


class StatisticsManager(object):
    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """

    def __init__(self, endpoint, debug_endpoint):
        self._scenario = None
        self._route_length = 0
        self._total_routes = 0
        self._results = Results()
        self._endpoint = endpoint
        self._debug_endpoint = debug_endpoint

    def add_file_records(self, endpoint):
        """Reads a file and saves its records onto the statistics manager"""
        data = fetch_dict(endpoint)

        if data:
            route_records = dictor(data, '_checkpoint.records')
            if route_records:
                for record in route_records:
                    self._results.checkpoint.records.append(to_route_record(record))

    def clear_records(self):
        """Cleans up the file"""
        if not self._endpoint.startswith(('http:', 'https:', 'ftp:')):
            with open(self._endpoint, 'w') as fd:
                fd.truncate(0)

    def sort_records(self):
        """Sorts the route records according to their route id (e.g. RouteScenario_0_rep0)"""
        self._results.checkpoint.records.sort(key=lambda x: (
            int(x.route_id.split('_')[1]),
            int(x.route_id.split('_rep')[-1])
        ))

        for i, record in enumerate(self._results.checkpoint.records):
            record.index = i

    def write_live_results(self, index, ego_speed, ego_control, ego_location):
        """Writes live results"""
        route_record = self._results.checkpoint.records[index]

        all_events = []
        if self._scenario:
            for node in self._scenario.get_criteria():
                all_events.extend(node.events)

        all_events.sort(key=lambda e: e.get_frame(), reverse=True)

        with open(self._debug_endpoint, 'w') as f:
            f.write(
                "Route id: {}\n\n"
                "Scenario: {}\n\n"
                "Town name: {}\n\n"
                "Weather id: {}\n\n"
                "Save name: {}\n\n"
                "Scores:\n"
                "    Driving score:         {:.3f}\n"
                "    Route completion:      {:.3f}\n"
                "    Infraction penalty:    {:.3f}\n"
                "    Challenge final:       {:.3f}\n"
                "    Challenge base:        {:.3f}\n"
                "    Challenge gate:        {:.3f}\n"
                "    Challenge penalty:     {:.3f}\n\n"
                "    Route length:          {:.3f}\n"
                "    Game duration:         {:.3f}\n"
                "    System duration:       {:.3f}\n\n"
                "Ego:\n"
                "    Throttle:              {:.3f}\n"
                "    Brake:                 {:.3f}\n"
                "    Steer:                 {:.3f}\n\n"
                "    Speed:                 {:.3f} km/h\n\n"
                "    Location:              ({:.3f} {:.3f} {:.3f})\n\n"
                "Total infractions: {}\n"
                "Last 5 infractions:\n".format(
                    route_record.route_id,
                    route_record.scenario_name,
                    route_record.town_name,
                    route_record.weather_id,
                    route_record.save_name,
                    route_record.scores["score_composed"],
                    route_record.scores["score_route"],
                    route_record.scores["score_penalty"],
                    route_record.scores.get("score_challenge", 0.0),
                    route_record.scores.get("score_challenge_base", 0.0),
                    route_record.scores.get("score_challenge_gate", 0.0),
                    route_record.scores.get("score_challenge_penalty", 0.0),
                    route_record.meta["route_length"],
                    route_record.meta["duration_game"],
                    route_record.meta["duration_system"],
                    ego_control.throttle,
                    ego_control.brake,
                    ego_control.steer,
                    ego_speed * 3.6,
                    ego_location.x,
                    ego_location.y,
                    ego_location.z,
                    route_record.num_infractions
                )
            )

            for e in all_events[:5]:
                # Prevent showing the ROUTE_COMPLETION event.
                event_type = e.get_type()
                if event_type == TrafficEventType.ROUTE_COMPLETION:
                    continue

                string = "    " + str(e.get_type()).replace("TrafficEventType.", "")
                if event_type in PENALTY_VALUE_DICT:
                    string += " (penalty: " + str(PENALTY_VALUE_DICT[event_type]) + ")\n"
                elif event_type in PENALTY_PERC_DICT:
                    string += " (value: " + str(round(e.get_dict()['percentage'], 3)) + "%)\n"

                f.write(string)

    def save_sensors(self, sensors):
        self._results.sensors = sensors

    def save_entry_status(self, entry_status):
        if entry_status not in ENTRY_STATUS_VALUES:
            raise ValueError("Found an invalid value for 'entry_status'")
        self._results.entry_status = entry_status
        self._results.eligible = ELIGIBLE_VALUES[entry_status]

    def save_progress(self, route_index, total_routes):
        self._results.checkpoint.progress = [route_index, total_routes]
        self._total_routes = total_routes

    def create_route_data(self, route_id, scenario_name, weather_id, save_name, town_name, index):
        """
        Creates the basic route data.
        This is done at the beginning to ensure the data is saved, even if a crash occurs.
        """
        route_record = RouteRecord()
        route_record.route_id = route_id
        route_record.scenario_name = scenario_name
        route_record.weather_id = weather_id
        route_record.save_name = save_name
        route_record.town_name = town_name

        # Check if we have to overwrite an element (when resuming), or create a new one
        route_records = self._results.checkpoint.records
        if index < len(route_records):
            self._results.checkpoint.records[index] = route_record
        else:
            self._results.checkpoint.records.append(route_record)

    def set_scenario(self, scenario):
        """Sets the scenario from which the statistics will be taken"""
        self._scenario = scenario
        self._route_length = round(compute_route_length(scenario.route), ROUND_DIGITS)

    def remove_scenario(self):
        """Removes the scenario"""
        self._scenario = None
        self._route_length = 0

    def compute_route_statistics(self, route_index, duration_time_system=-1, duration_time_game=-1, failure_message=""):
        """
        Compute the current statistics by evaluating all relevant scenario criteria.
        Failure message will not be empty if an external source has stopped the simulations
        (i.e. simulation crash). For the rest of the cases, it will be filled by this function.
        """

        def set_infraction_message():
            infraction_name = PENALTY_NAME_DICT[event.get_type()]
            route_record.infractions[infraction_name].append(event.get_message())

        def set_score_penalty(score_penalty):
            event_value = event.get_dict()['percentage']
            penalty_value, penalty_type = PENALTY_PERC_DICT[event.get_type()]

            if penalty_type == "decreases":
                score_penalty *= (1 - (1 - penalty_value) * (1 - event_value / 100))
            elif penalty_type == "increases":
                score_penalty *= (1 - (1 - penalty_value) * event_value / 100)
            elif penalty_type == "unused":
                pass
            else:
                raise ValueError("Found a criteria with an unknown penalty type")

            return score_penalty

        route_record = self._results.checkpoint.records[route_index]
        route_record.index = route_index

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0

        for event_name in PENALTY_NAME_DICT.values():
            route_record.infractions[event_name] = []
        route_record.infractions['route_timeout'] = []

        # Update the route meta
        route_record.meta['route_length'] = self._route_length
        route_record.meta['duration_game'] = round(duration_time_game, ROUND_DIGITS)
        route_record.meta['duration_system'] = round(duration_time_system, ROUND_DIGITS)

        # Update the route infractions
        if self._scenario:
            if self._scenario.timeout_node.timeout:
                route_record.infractions['route_timeout'].append('Route timeout.')
                failure_message = "Agent timed out"

            for node in self._scenario.get_criteria():
                for event in node.events:
                    # Traffic events that subtract a set amount of points
                    if event.get_type() in PENALTY_VALUE_DICT:
                        score_penalty *= PENALTY_VALUE_DICT[event.get_type()]
                        set_infraction_message()

                    # Traffic events that subtract a varying amount of points
                    elif event.get_type() in PENALTY_PERC_DICT:
                        score_penalty = set_score_penalty(score_penalty)
                        set_infraction_message()

                    # Traffic events that stop the simulation
                    elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                        failure_message = "Agent deviated from the route"
                        set_infraction_message()

                    elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                        failure_message = "Agent got blocked"
                        set_infraction_message()

                    elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                        score_route = event.get_dict()['route_completed']
                        target_reached = score_route >= 100

        # Original leaderboard scores
        route_record.scores['score_route'] = round(score_route, ROUND_DIGITS_SCORE)
        route_record.scores['score_penalty'] = round(score_penalty, ROUND_DIGITS_SCORE)
        route_record.scores['score_composed'] = round(max(score_route * score_penalty, 0.0), ROUND_DIGITS_SCORE)

        # Challenge-specific facts and score
        if self._scenario:
            criteria_list = self._scenario.get_criteria()
            scenario_type = self._scenario.config.scenario_configs[0].type
            common_facts, private_facts, score_detail = evaluate_scenario(criteria_list, scenario_type)

            route_record.common_facts = common_facts
            route_record.private_facts = private_facts
            route_record.score_detail = score_detail
            route_record.meta['scenario_type'] = scenario_type

            route_record.scores['score_challenge_base'] = round(score_detail.get('base_score', 0.0), ROUND_DIGITS_SCORE)
            route_record.scores['score_challenge_gate'] = round(score_detail.get('gate', 0.0), ROUND_DIGITS_SCORE)
            route_record.scores['score_challenge_penalty'] = round(score_detail.get('penalty', 0.0), ROUND_DIGITS_SCORE)
            route_record.scores['score_challenge'] = round(score_detail.get('final_score', 0.0), ROUND_DIGITS_SCORE)

            print(common_facts, "\n", private_facts, "\n", score_detail)

        # Update result
        route_record.num_infractions = sum([len(route_record.infractions[key]) for key in route_record.infractions])

        if target_reached:
            route_record.status = 'Completed' if route_record.num_infractions > 0 else 'Perfect'
        else:
            route_record.status = 'Failed'
            if failure_message:
                route_record.status += ' - ' + failure_message

        # Add the new data, or overwrite a previous result (happens when resuming the simulation)
        record_len = len(self._results.checkpoint.records)
        if route_index == record_len:
            self._results.checkpoint.records.append(route_record)
        elif route_index < record_len:
            self._results.checkpoint.records[route_index] = route_record
        else:
            raise ValueError("Not enough entries in the route record")

    def compute_global_statistics(self):
        """Computes and saves the global statistics of the routes"""

        def get_infractions_value(route_record, key):
            # Special case for the % based criteria. Extract the meters from the message.
            if key == PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]:
                if not route_record.infractions[key]:
                    return 0.0
                return float(route_record.infractions[key][0].split(" ")[8]) / 1000

            return len(route_record.infractions[key])

        global_record = GlobalRecord()
        global_result = global_record.status

        route_records = self._results.checkpoint.records

        # Calculate the score means and result
        for route_record in route_records:
            global_record.scores_mean['score_route'] += route_record.scores['score_route'] / self._total_routes
            global_record.scores_mean['score_penalty'] += route_record.scores['score_penalty'] / self._total_routes
            global_record.scores_mean['score_composed'] += route_record.scores['score_composed'] / self._total_routes

            global_record.scores_mean['score_challenge_base'] += route_record.scores.get('score_challenge_base', 0.0) / self._total_routes
            global_record.scores_mean['score_challenge_gate'] += route_record.scores.get('score_challenge_gate', 0.0) / self._total_routes
            global_record.scores_mean['score_challenge_penalty'] += route_record.scores.get('score_challenge_penalty', 0.0) / self._total_routes
            global_record.scores_mean['score_challenge'] += route_record.scores.get('score_challenge', 0.0) / self._total_routes

            global_record.meta['total_length'] += route_record.meta['route_length']
            global_record.meta['duration_game'] += route_record.meta['duration_game']
            global_record.meta['duration_system'] += route_record.meta['duration_system']

            scenario_type = route_record.meta.get('scenario_type', '')
            if scenario_type and scenario_type not in global_record.meta['scenario_types']:
                global_record.meta['scenario_types'].append(scenario_type)

            global_record.meta['challenge_common_facts'].append({
                'route_id': route_record.route_id,
                'scenario_type': scenario_type,
                'facts': route_record.common_facts
            })

            global_record.meta['challenge_private_facts'].append({
                'route_id': route_record.route_id,
                'scenario_type': scenario_type,
                'facts': route_record.private_facts
            })

            global_record.meta['challenge_score_breakdown'].append({
                'route_id': route_record.route_id,
                'scenario_type': scenario_type,
                'score_detail': route_record.score_detail
            })

            # Downgrade the global result if need be ('Perfect' -> 'Completed' -> 'Failed')
            route_result = 'Failed' if 'Failed' in route_record.status else route_record.status
            if route_result == 'Failed':
                global_record.meta['exceptions'].append(
                    (route_record.route_id, route_record.index, route_record.status)
                )
                global_result = route_result
            elif global_result == 'Perfect' and route_result != 'Perfect':
                global_result = route_result

        for item in global_record.scores_mean:
            global_record.scores_mean[item] = round(global_record.scores_mean[item], ROUND_DIGITS_SCORE)

        global_record.status = global_result

        # Calculate the score standard deviation
        if self._total_routes == 1:
            for key in global_record.scores_std_dev:
                global_record.scores_std_dev[key] = 0
        else:
            for route_record in route_records:
                for key in global_record.scores_std_dev:
                    diff = route_record.scores.get(key, 0.0) - global_record.scores_mean[key]
                    global_record.scores_std_dev[key] += math.pow(diff, 2)

            for key in global_record.scores_std_dev:
                value = round(
                    math.sqrt(global_record.scores_std_dev[key] / float(self._total_routes - 1)),
                    ROUND_DIGITS
                )
                global_record.scores_std_dev[key] = value

        # Calculate the number of infractions per km
        km_driven = 0
        for route_record in route_records:
            km_driven += route_record.meta['route_length'] / 1000 * route_record.scores['score_route'] / 100
            for key in global_record.infractions:
                global_record.infractions[key] += get_infractions_value(route_record, key)

        km_driven = max(km_driven, 0.001)

        for key in global_record.infractions:
            # Special case for the % based criteria.
            if key != PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]:
                global_record.infractions[key] /= km_driven
            global_record.infractions[key] = round(global_record.infractions[key], ROUND_DIGITS)

        # Save the global records
        self._results.checkpoint.global_record = global_record

        # Change the values and labels. These MUST HAVE A MATCHING ORDER
        self._results.values = [
            str(global_record.scores_mean['score_composed']),
            str(global_record.scores_mean['score_route']),
            str(global_record.scores_mean['score_penalty']),
            str(global_record.scores_mean['score_challenge_base']),
            str(global_record.scores_mean['score_challenge_gate']),
            str(global_record.scores_mean['score_challenge_penalty']),
            str(global_record.scores_mean['score_challenge']),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.COLLISION_PEDESTRIAN]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.COLLISION_VEHICLE]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.COLLISION_STATIC]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.TRAFFIC_LIGHT_INFRACTION]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.STOP_INFRACTION]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.ROUTE_DEVIATION]]),
            str(global_record.infractions['route_timeout']),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.VEHICLE_BLOCKED]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.SCENARIO_TIMEOUT]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.MIN_SPEED_INFRACTION]]),
        ]

        self._results.labels = [
            "Avg. driving score",
            "Avg. route completion",
            "Avg. infraction penalty",
            "Avg. challenge base score",
            "Avg. challenge gate",
            "Avg. challenge penalty",
            "Avg. challenge final score",
            "Collisions with pedestrians",
            "Collisions with vehicles",
            "Collisions with layout",
            "Red lights infractions",
            "Stop sign infractions",
            "Off-road infractions",
            "Route deviations",
            "Route timeouts",
            "Agent blocked",
            "Yield emergency vehicles infractions",
            "Scenario timeouts",
            "Min speed infractions"
        ]

        # Change the entry status and eligible
        entry_status = 'Finished'
        for route_record in route_records:
            route_status = route_record.status
            if 'Simulation crashed' in route_status:
                entry_status = 'Crashed'
            elif "Agent's sensors were invalid" in route_status:
                entry_status = 'Rejected'

        self.save_entry_status(entry_status)

    def validate_and_write_statistics(self, sensors_initialized, crashed):
        """
        Makes sure that all the relevant data is there.
        Changes the 'entry status' to 'Invalid' if this isn't the case.
        """
        error_message = ""
        if sensors_initialized and not self._results.sensors:
            error_message = "Missing 'sensors' data"

        elif not self._results.values:
            error_message = "Missing 'values' data"

        elif self._results.entry_status == 'Started':
            error_message = "'entry_status' has the 'Started' value"

        else:
            global_records = self._results.checkpoint.global_record
            progress = self._results.checkpoint.progress
            route_records = self._results.checkpoint.records

            if not global_records:
                error_message = "Missing 'global_records' data"

            elif not progress:
                error_message = "Missing 'progress' data"

            elif not crashed and (progress[0] != progress[1] or progress[0] != len(route_records)):
                error_message = "'progress' data doesn't match its expected value"

            else:
                for record in route_records:
                    if record.status == 'Started':
                        error_message = "Found a route record with missing data"
                        break

        if error_message:
            print("\n\033[91mThe statistics are badly formed. Setting their status to 'Invalid':")
            print("> {}\033[0m\n".format(error_message))

            self.save_entry_status('Invalid')

        self.write_statistics()

    def write_statistics(self):
        """
        Writes the results into the endpoint. Meant to be used only for partial evaluations,
        use 'validate_and_write_statistics' for the final one as it only validates the data.
        """
        save_dict(self._endpoint, self._results.to_json())
