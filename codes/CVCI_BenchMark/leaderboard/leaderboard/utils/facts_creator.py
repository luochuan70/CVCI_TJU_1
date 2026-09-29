from srunner.scenariomanager.traffic_events import TrafficEventType


def extract_common_facts(criteria_list):
    """
    通用 facts 提取：
    - 不再提取 min_ttc
    - 不再为 TTC 提供 penalty 依据
    - 只保留 collision penalty，且与 Bench2Drive 一致
    """
    common_facts = {
        "collision": False,
        "collision_penalty": 1.0,
        "outside_route": False,
        "running_red_light": False,
        "running_stop": False,
        "agent_blocked": False,
        "route_completed": False,
    }

    collision_penalty = 1.0

    for criterion in criteria_list:
        name = criterion.name

        # 原有的通用布尔 facts
        if name == "OutsideRouteLanesTest":
            common_facts["outside_route"] = (criterion.test_status == "FAILURE")

        elif name == "RunningRedLightTest":
            common_facts["running_red_light"] = (criterion.test_status == "FAILURE")

        elif name == "RunningStopTest":
            common_facts["running_stop"] = (criterion.test_status == "FAILURE")

        elif name == "AgentBlockedTest":
            common_facts["agent_blocked"] = (criterion.test_status == "FAILURE")

        elif name == "RouteCompletionTest":
            common_facts["route_completed"] = (criterion.test_status == "SUCCESS")

        # 遍历所有 criterion 的 events，提取碰撞 penalty
        if hasattr(criterion, "events"):
            for event in criterion.events:
                event_type = event.get_type()

                if event_type == TrafficEventType.COLLISION_PEDESTRIAN:
                    common_facts["collision"] = True
                    collision_penalty *= 0.5

                elif event_type == TrafficEventType.COLLISION_VEHICLE:
                    common_facts["collision"] = True
                    collision_penalty *= 0.6

                elif event_type == TrafficEventType.COLLISION_STATIC:
                    common_facts["collision"] = True
                    collision_penalty *= 0.65

    common_facts["collision_penalty"] = collision_penalty
    return common_facts


# missing car_private_facats extracts
def extract_private_facts_frontcar_disappearance(criteria_list):
    facts = {
        "slow_down": False,
        "safe_bypass": False,
        "reach_end_point": False
    }

    for criterion in criteria_list:
        if criterion.name == "StaticObstacleBrakeSlowDownCriterion":
            facts["slow_down"] = (criterion.brake_status == "SUCCESS")

        if criterion.name == "StaticObstacleSafePassCriterion":
            facts["safe_bypass"] = (criterion.safepass_status == "SUCCESS")

        if criterion.name == "ReachEndPointCriterion":
            facts["reach_end_point"] = (criterion.reach_status == "SUCCESS")

    return facts


# High speed temporary construction_private_facats extracts
def extract_private_facts_static_barrier(criteria_list):
    facts = {
        "barrier_slow_down": False,
        "detour": False,
        "reach_goal": False
    }

    for criterion in criteria_list:
        if criterion.name == "BarrierSlowDownCriterion":
            facts["barrier_slow_down"] = (criterion.test_status == "SUCCESS")

        if criterion.name == "BarrierDetourCriterion":
            facts["detour"] = (criterion.test_status == "SUCCESS")

        if criterion.name == "BarrierReachGoalCriterion":
            facts["reach_goal"] = (criterion.test_status == "SUCCESS")

    return facts


# High-speed reckless lane cutting_private_facats extracts
def extract_private_facts_high_speed_cutting(criteria_list):
    facts = {
        "brake_response": False,
        "safe_bypass": False,
    }

    for criterion in criteria_list:
        if criterion.name == "CutInBrakeResponseCriterion":
            facts["brake_response"] = (criterion.brake_status == "SUCCESS")
        elif criterion.name == "CutInSafeBypassCriterion":
            facts["safe_bypass"] = (criterion.safepass_status == "SUCCESS")

    return facts


# Highway accident vehicle_private_facats extracts
def extract_private_facts_high_speed_accident(criteria_list):
    facts = {
        "brake_response": False,
        "safe_bypass": False,
        "resume_route": False,
    }

    for criterion in criteria_list:
        if criterion.name == "HighSpeedBrakeCriterion":
            facts["brake_response"] = (criterion.test_status == "SUCCESS")
        elif criterion.name == "HighSpeedBypassCriterion":
            facts["safe_bypass"] = (criterion.test_status == "SUCCESS")
        elif criterion.name == "HighSpeedResumeCriterion":
            facts["resume_route"] = (criterion.test_status == "SUCCESS")

    return facts


# Trucks encountered during construction_private_facats extracts
def extract_private_facts_lane_closure(criteria_list):
    facts = {
        "deceleration_detected": False,
        "speed_reduction": 0.0,
        "distance_traveled": 0.0,
    }

    for criterion in criteria_list:
        if criterion.name == "DecelerationForConstructionTest":
            speed_reduction = criterion.actual_value
            facts["speed_reduction"] = speed_reduction

            if speed_reduction >= 30.0 or criterion.test_status == "SUCCESS":
                facts["deceleration_detected"] = True

        elif criterion.name == "RoutePassCompletionTest":
            facts["distance_traveled"] = criterion.actual_value

    return facts


# Drive into the roundabout_private_facats extracts
def extract_private_facts_roundabout_merge_conflict(criteria_list):
    facts = {
        "decelerate_response": False,
        "yield_convoy": False,
        "safe_pass": False,
    }

    for criterion in criteria_list:
        if criterion.name == "RoundaboutDecelerateCriterion":
            facts["decelerate_response"] = (criterion.decelerate_status == "SUCCESS")

        elif criterion.name == "RoundaboutYieldConvoyCriterion":
            facts["yield_convoy"] = (criterion.yield_status == "SUCCESS")

        elif criterion.name == "RoundaboutSafePassCriterion":
            facts["safe_pass"] = (criterion.safe_pass_status == "SUCCESS")

    return facts


# Four students crossing the road_private_facats extracts
def extract_private_facts_ghost_probe(criteria_root):
    facts = {
        "scooter_decelerate": False,
        "pedestrian_stop": False,
        "pedestrian_resume": False,
    }

    nodes = criteria_root.iterate() if hasattr(criteria_root, "iterate") else criteria_root

    for criterion in nodes:
        if hasattr(criterion, "name"):
            if criterion.name == "ScooterDecelerateCriterion":
                facts["scooter_decelerate"] = (criterion.brake_status == "SUCCESS")
            elif criterion.name == "PedestrianStopCriterion":
                facts["pedestrian_stop"] = (criterion.test_status == "SUCCESS")
            elif criterion.name == "PedestrianResumeCriterion":
                facts["pedestrian_resume"] = (criterion.test_status == "SUCCESS")

    return facts


# avoid a disabled vehicle_private_facats extracts
def extract_private_facts_broken_down_vehicle(criteria_list):
    facts = {
        "brake_response": False,
        "safe_bypass": False,
        "resume_route": False,
    }

    for criterion in criteria_list:
        if criterion.name == "BrakeCriterion":
            facts["brake_response"] = (criterion.test_status == "SUCCESS")

        elif criterion.name == "BypassCriterion":
            facts["safe_bypass"] = (criterion.test_status == "SUCCESS")

        elif criterion.name == "ResumeCriterion":
            facts["resume_route"] = (criterion.test_status == "SUCCESS")

    return facts


# Slanted motor and children_private_facats extracts
def extract_private_facts_ebike_pedestrian_cross(criteria_list):
    facts = {
        "ebike_decelerate": False,
        "pedestrian_stop": False,
        "resume_route": False,
    }

    for criterion in criteria_list:
        if criterion.name == "EbikeDetectionAndDecelerateCriterion":
            facts["ebike_decelerate"] = (criterion.test_status == "SUCCESS")

        elif criterion.name == "PedestrianDetectionAndStopCriterion":
            facts["pedestrian_stop"] = (criterion.test_status == "SUCCESS")

        elif criterion.name == "ResumeAfterPedestrianCriterion":
            facts["resume_route"] = (criterion.test_status == "SUCCESS")

    return facts


# reverse vehicle_private_facats extracts
def extract_private_facts_reverse_vehicle(criteria_list):
    facts = {
        "brake_response": False,
        "safe_bypass": False,
        "resume_route": False,
    }

    for criterion in criteria_list:
        if criterion.name == "ReverseVehicleBrakeCriterion":
            facts["brake_response"] = (criterion.brake_status == "SUCCESS")

        elif criterion.name == "ReverseVehicleBypassCriterion":
            facts["safe_bypass"] = (criterion.bypass_status == "SUCCESS")

        elif criterion.name == "ReverseVehicleResumeCriterion":
            facts["resume_route"] = (criterion.resume_status == "SUCCESS")

    return facts


# crazy motor_private_facats extracts
def extract_private_facts_crazy_bike(criteria_list):
    facts = {
        "decelerate_response": False,
        # The PDF assigns a separate 20-point no-collision item.  Start from
        # the safe state and revoke it when this scenario's criteria expose a
        # pedestrian/vehicle/static collision event.
        "no_collision": True,
        "resume_route": False,
    }

    for criterion in criteria_list:
        if criterion.name == "CrazyBikeDecelerateCriterion":
            facts["decelerate_response"] = (criterion.test_status == "SUCCESS")

        elif criterion.name == "CrazyBikeResumeCriterion":
            facts["resume_route"] = (criterion.test_status == "SUCCESS")

        for event in getattr(criterion, "events", []) or []:
            if event.get_type() in (
                TrafficEventType.COLLISION_PEDESTRIAN,
                TrafficEventType.COLLISION_VEHICLE,
                TrafficEventType.COLLISION_STATIC,
            ):
                facts["no_collision"] = False

    return facts


# Blind spot hidden car_private_facats extracts
def extract_private_facts_left_turn(criteria_list):
    facts = {
        "brake_response": False,
        "safe_bypass": False,
    }

    for criterion in criteria_list:
        if criterion.name == "IntersectionCollisionLeftTurnBrakeCriterion":
            facts["brake_response"] = (criterion.brake_status == "SUCCESS")

        elif criterion.name == "IntersectionCollisionLeftTurnResumeCriterion":
            facts["safe_bypass"] = (criterion.safepass_status == "SUCCESS")

    return facts
