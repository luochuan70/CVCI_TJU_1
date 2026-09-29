"""Authoritative ownership map for the twelve CVCI macro rule systems.

The runtime classifier and controller remain observation-only.  This registry
contains no route ids, files, map coordinates, actors, or evaluator truth; it
only makes the one-owner/one-controller invariant explicit and auditable.
"""


DEDICATED_MACRO_RULES = {
    "avoid_a_disabled_vehicle": {
        "owner_symbol": "_disabled_scene_memory",
        "control_symbol": "_rule_avoid_a_disabled_vehicle",
    },
    "blind_spot_hidden_car": {
        "owner_symbol": "_blind_spot_route_owner",
        "control_symbol": "_rule_blind_spot_hidden_car",
    },
    "crazy_motor": {
        "owner_symbol": "_crazy_motor_memory",
        "control_symbol": "_rule_crazy_motor",
    },
    "four_students_crossing_the_road": {
        "owner_symbol": "_vru_crossing_memory",
        "control_symbol": "_rule_four_students_crossing_the_road",
    },
    "high_speed_reckless_lane_cutting": {
        "owner_symbol": "_reckless_cut_in_route_owner",
        "control_symbol": "_rule_high_speed_reckless_lane_cutting",
    },
    "high_speed_temporary_construction": {
        "owner_symbol": "_highspeed_construction_memory",
        "control_symbol": "_rule_high_speed_temporary_construction",
    },
    "highway_accident_vehicle": {
        "owner_symbol": "_highway_accident_memory",
        "control_symbol": "_rule_highway_accident_vehicle",
    },
    "missing_car": {
        "owner_symbol": "_missing_car_event_memory",
        "control_symbol": "_rule_missing_car",
    },
    "reverse_vehicle": {
        "owner_symbol": "_reverse_vehicle_route_memory",
        "control_symbol": "_rule_reverse_vehicle",
    },
    "roundabout": {
        "owner_symbol": "_roundabout_route_memory",
        "control_symbol": "_rule_roundabout",
    },
    "slanted_motor_and_children": {
        "owner_symbol": "_slanted_motor_children_route_owner",
        "control_symbol": "_rule_slanted_motor_and_children",
    },
    "trucks_encountered_during_construction": {
        "owner_symbol": "_lane_closure_truck_memory",
        "control_symbol": "_rule_trucks_encountered_during_construction",
    },
}


def validate_dedicated_macro_rules(expected_macros=None):
    """Fail closed when a known macro lacks a unique owner or controller."""
    expected = set(expected_macros or DEDICATED_MACRO_RULES)
    actual = set(DEDICATED_MACRO_RULES)
    if actual != expected:
        raise RuntimeError(
            "CVCI dedicated macro registry mismatch: missing=%s extra=%s"
            % (sorted(expected - actual), sorted(actual - expected))
        )
    owners = [item["owner_symbol"] for item in DEDICATED_MACRO_RULES.values()]
    controls = [item["control_symbol"] for item in DEDICATED_MACRO_RULES.values()]
    if len(owners) != len(set(owners)) or len(controls) != len(set(controls)):
        raise RuntimeError("CVCI dedicated macro owner/control symbols must be unique")
    if any(not owner or not control for owner, control in zip(owners, controls)):
        raise RuntimeError("CVCI dedicated macro owner/control symbol is empty")
    return True


validate_dedicated_macro_rules()
