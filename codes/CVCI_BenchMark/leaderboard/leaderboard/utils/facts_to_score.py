def compute_collision_penalty(common_facts):
    """
    仅保留与 Bench2Drive 一致的碰撞惩罚：
    - 撞行人: 0.5
    - 撞车辆: 0.6
    - 撞静态物体: 0.65

    如果没有碰撞，则 penalty=1.0
    """
    return float(common_facts.get("collision_penalty", 1.0))


def build_score_result(base_score, common_facts):
    """
    统一的结果封装：
    - 不再使用 gate
    - 不再使用 TTC penalty
    - final_score = base_score * collision_penalty

    为兼容 statistics_manager.py，保留 gate 字段，但固定为 1.0
    """
    penalty = compute_collision_penalty(common_facts)
    gate = 1.0
    final_score = base_score * penalty

    return {
        "base_score": round(base_score, 6),
        "gate": round(gate, 6),          # 兼容旧 statistics_manager，实际不起作用
        "penalty": round(penalty, 6),    # 现在仅表示 collision penalty
        "final_score": round(final_score, 6),
    }


# =========================
# Scenario specific scorers
# =========================

# missing car
def score_frontcar_disappear_accident(common_facts, private_facts):
    base_score = 0.0

    if private_facts["slow_down"]:
        base_score += 50.0

    if private_facts["safe_bypass"]:
        base_score += 30.0

    if private_facts["reach_end_point"]:
        base_score += 20.0

    return build_score_result(base_score, common_facts)


# High speed temporary construction
def score_static_barrier(common_facts, private_facts):
    base_score = 0.0

    if private_facts["barrier_slow_down"]:
        base_score += 55.0

    if private_facts["detour"]:
        base_score += 20.0

    if private_facts["reach_goal"]:
        base_score += 25.0

    return build_score_result(base_score, common_facts)


# High-speed reckless lane cutting
def score_high_speed_cutting(common_facts, private_facts):
    base_score = 0.0

    if private_facts.get("brake_response"):
        base_score += 60.0
    if private_facts.get("safe_bypass"):
        base_score += 40.0

    return build_score_result(base_score, common_facts)


# Highway accident vehicle
def score_high_speed_accident(common_facts, private_facts):
    base_score = 0.0

    if private_facts["brake_response"]:
        base_score += 55.0
    if private_facts["safe_bypass"]:
        base_score += 30.0
    if private_facts["resume_route"]:
        base_score += 15.0

    return build_score_result(base_score, common_facts)


# Trucks encountered during construction
def compute_lane_closure_score(common_facts, private_facts):
    base_score = 0.0

    if private_facts.get("deceleration_detected", False):
        base_score += 80.0

    # The rule table awards the 20-point item only after the complete target
    # distance has been passed.  Do not turn an uncompleted pass into a hidden
    # fractional score that is absent from Introduction.pdf.
    distance_traveled = private_facts.get("distance_traveled", 0.0)
    target_distance = 64.0
    if float(distance_traveled) >= target_distance:
        base_score += 20.0

    return build_score_result(base_score, common_facts)


# Drive into the roundabout
def score_roundabout_merge_conflict(common_facts, private_facts):
    base_score = 0.0

    if private_facts["decelerate_response"]:
        base_score += 55.0
    if private_facts["yield_convoy"]:
        base_score += 20.0
    if private_facts["safe_pass"]:
        base_score += 25.0

    return build_score_result(base_score, common_facts)


# Four students crossing the road
def score_ghost_probe(common_facts, private_facts):
    base_score = 0.0

    if private_facts["scooter_decelerate"]:
        base_score += 25.0
    if private_facts["pedestrian_stop"]:
        base_score += 55.0
    if private_facts["pedestrian_resume"]:
        base_score += 20.0

    return build_score_result(base_score, common_facts)


# avoid a disabled vehicle
def score_broken_down_vehicle(common_facts, private_facts):
    base_score = 0.0

    if private_facts["brake_response"]:
        base_score += 40.0
    if private_facts["safe_bypass"]:
        base_score += 40.0
    if private_facts["resume_route"]:
        base_score += 20.0

    return build_score_result(base_score, common_facts)


# Slanted motor and children
def score_ebike_pedestrian_cross(common_facts, private_facts):
    base_score = 0.0

    if private_facts.get("ebike_decelerate", False):
        base_score += 25.0
    if private_facts.get("pedestrian_stop", False):
        base_score += 50.0
    if private_facts.get("resume_route", False):
        base_score += 25.0

    return build_score_result(base_score, common_facts)


# reverse vehicle
def score_reverse_vehicle(common_facts, private_facts):
    base_score = 0.0

    if private_facts["brake_response"]:
        base_score += 55.0
    if private_facts["safe_bypass"]:
        base_score += 20.0
    if private_facts["resume_route"]:
        base_score += 25.0

    return build_score_result(base_score, common_facts)


# crazy motor
def score_crazy_bike(common_facts, private_facts):
    base_score = 0.0

    if private_facts.get("decelerate_response", False):
        base_score += 55.0
    if private_facts.get("no_collision", False):
        base_score += 20.0
    if private_facts.get("resume_route", False):
        base_score += 25.0

    return build_score_result(base_score, common_facts)


# Blind spot hidden car
def score_left_turn(common_facts, private_facts):
    base_score = 0.0

    if private_facts["brake_response"]:
        base_score += 70.0
    if private_facts["safe_bypass"]:
        base_score += 30.0

    return build_score_result(base_score, common_facts)
