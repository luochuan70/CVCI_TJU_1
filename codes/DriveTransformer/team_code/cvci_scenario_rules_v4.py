import math

try:
    from DriveTransformer.team_code.cvci_rule_config import SCENARIO_RULE_CONFIG
    from DriveTransformer.team_code.cvci_scenario_context import RuleAction
except ModuleNotFoundError:
    from cvci_rule_config import SCENARIO_RULE_CONFIG
    from cvci_scenario_context import RuleAction


class BaseV3Policy:
    name = "v3_fallback"


class ScenarioSpecificOverride:
    macro_scenario = "unknown"

    def __init__(self, config):
        self.config = config

    def apply(self, context):
        return RuleAction()


class TrucksConstructionRule(ScenarioSpecificOverride):
    macro_scenario = "trucks_encountered_during_construction"

    def _construction_steer_bias(self, context):
        cfg = self.config
        base_bias = float(cfg.get("construction_steer_bias", -0.18))
        left_clear = context.risk_flags.get("left_clear", True)
        right_clear = context.risk_flags.get("right_clear", True)

        front_laterals = []
        for obj in context.detections:
            if float(obj.get("score", 0.0)) < float(cfg.get("obstacle_confidence", 0.30)):
                continue
            cls = str(obj.get("class_name", "")).lower()
            if cls not in ("truck", "van", "car", "traffic_cone", "others"):
                continue
            box = obj.get("box_lidar") or {}
            x = float(box.get("x", 999.0))
            y = float(box.get("y", 999.0))
            if 0.0 <= x <= 24.0 and abs(y) <= 4.0:
                front_laterals.append(y)

        if front_laterals:
            mean_y = sum(front_laterals) / float(len(front_laterals))
            if mean_y < -0.35 and right_clear:
                return abs(base_bias)
            if mean_y > 0.35 and left_clear:
                return -abs(base_bias)

        if not left_clear and right_clear:
            return abs(base_bias)
        if not right_clear and left_clear:
            return -abs(base_bias)
        return base_bias

    def apply(self, context):
        cfg = self.config
        if not cfg.get("enabled", True):
            return RuleAction()
        if context.confidence < float(cfg.get("min_confidence", 0.7)):
            return RuleAction()

        front_obstacle = context.risk_flags.get("front_obstacle", False)
        front_vehicle = context.risk_flags.get("front_vehicle", False)
        side_risk = context.risk_flags.get("side_risk", False)
        phase = context.phase
        speed = float(context.ego_speed)

        active = phase in ("approach", "interaction", "recovery")
        if not active:
            return RuleAction()

        target_speed = float(cfg["approach_speed"])
        throttle_scale = 1.0
        brake = None
        steer_scale = float(cfg.get("steer_scale", 0.75))
        steer_bias = self._construction_steer_bias(context)
        reason = "construction_approach_speed_cap"

        if front_obstacle or front_vehicle or side_risk:
            target_speed = float(cfg["interaction_speed"])
            throttle_scale = 0.75
            reason = "construction_detected_corridor_risk"
        if speed > target_speed + 4.0:
            brake = 0.10 if brake is None else max(brake, 0.10)
        if context.risk_flags.get("immediate_hazard", False):
            brake = max(brake or 0.0, float(cfg.get("brake_near_obstacle", 0.45)))
            throttle_scale = 0.30
            steer_bias *= 0.75
            reason = "construction_immediate_hazard"
        if phase == "recovery" and speed < 1.0:
            target_speed = max(target_speed, 5.0)
            throttle_scale = 1.0
            brake = None
            steer_scale = 1.0
            steer_bias = 0.0
            reason = "construction_clear_recovery_release"

        return RuleAction(
            target_speed=target_speed,
            throttle_scale=throttle_scale,
            brake=brake,
            steer_scale=steer_scale,
            steer_smoothing=float(cfg.get("steer_smoothing", 0.4)),
            steer_bias=steer_bias,
            steer_limit=float(cfg.get("construction_steer_limit", 0.35)),
            hold_frames=int(cfg.get("hold_frames", 20)),
            reason=reason,
            active_rule=self.macro_scenario,
        )


class ScenarioRuleRegistry:
    def __init__(self, config=None):
        self.config = config or SCENARIO_RULE_CONFIG
        self.rules = {
            TrucksConstructionRule.macro_scenario: TrucksConstructionRule(
                self.config.get(TrucksConstructionRule.macro_scenario, {})
            )
        }

    def apply(self, context):
        rule = self.rules.get(context.macro_scenario)
        if rule is None:
            return RuleAction()
        action = rule.apply(context)
        if action.target_speed is not None and not math.isfinite(float(action.target_speed)):
            return RuleAction()
        return action
