"""Online, route-relative local-path tracking for ordinary-scene rules.

The controller consumes only the current route-navigation tangent, ego speed,
compass and timestamp. Paths use a local frame whose x axis is the online
route direction observed when a path cycle starts. Route identity, map
coordinates, XML waypoints and evaluator metadata are deliberately absent.
"""

from __future__ import absolute_import

import math
from collections import deque


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _clip(value, low, high):
    return max(low, min(high, value))


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _xy(value):
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def route_lateral_offset(tick_data):
    """Return the signed ego offset from the online route segment.

    The sign is the same as the controller's local ``y_m`` convention.  This
    uses the live position and navigation nodes only; it has no map, XML or
    route-identity dependency.
    """
    if not isinstance(tick_data, dict):
        return None
    position = _xy(tick_data.get("pos"))
    near = _xy(tick_data.get("command_near_xy"))
    far = _xy(tick_data.get("command_far_xy"))
    if position is None or near is None or far is None:
        return None
    dx, dy = far[0] - near[0], far[1] - near[1]
    norm = math.hypot(dx, dy)
    if norm <= 1.0:
        return None
    axis_x, axis_y = dx / norm, dy / norm
    delta_x, delta_y = position[0] - near[0], position[1] - near[1]
    return axis_x * delta_y - axis_y * delta_x


def dense_route_lateral_offset(tick_data):
    """Return signed cross-track error to the nearest live dense-route segment."""
    if not isinstance(tick_data, dict):
        return None
    position = _xy(tick_data.get("pos"))
    points = [_xy(item) for item in list(tick_data.get("route_horizon_xy") or [])]
    points = [point for point in points if point is not None]
    if position is None or len(points) < 2:
        return None
    best = None
    for left, right in zip(points[:-1], points[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 0.04:
            continue
        fraction = _clip(
            ((position[0] - left[0]) * dx + (position[1] - left[1]) * dy)
            / length_sq,
            0.0,
            1.0,
        )
        projected = (left[0] + fraction * dx, left[1] + fraction * dy)
        delta_x = position[0] - projected[0]
        delta_y = position[1] - projected[1]
        distance_sq = delta_x * delta_x + delta_y * delta_y
        if best is None or distance_sq < best[0]:
            norm = math.sqrt(length_sq)
            lateral = (dx / norm) * delta_y - (dy / norm) * delta_x
            best = (distance_sq, lateral)
    return None if best is None else float(best[1])


def route_waypoint_observation(tick_data):
    """Return live near/far heading evidence without route identity or map data."""
    if not isinstance(tick_data, dict):
        return None
    position = _xy(tick_data.get("pos"))
    near = _xy(tick_data.get("command_near_xy"))
    far = _xy(tick_data.get("command_far_xy"))
    compass = _number(tick_data.get("compass"), float("nan"))
    if position is None or near is None or far is None or not math.isfinite(compass):
        return None

    def state(target):
        dx, dy = target[0] - position[0], target[1] - position[1]
        return {
            "distance_m": math.hypot(dx, dy),
            "heading_error_rad": _wrap(math.atan2(dx, -dy) - compass),
        }

    return {"near": state(near), "far": state(far)}


def route_horizon_observation(tick_data):
    """Summarize a live dense route without exposing route identity."""
    if not isinstance(tick_data, dict):
        return None
    position = _xy(tick_data.get("pos"))
    points = [_xy(item) for item in list(tick_data.get("route_horizon_xy") or [])]
    points = [point for point in points if point is not None]
    if position is None or len(points) < 3:
        return None
    nearest = min(
        range(len(points)),
        key=lambda index: math.hypot(
            points[index][0] - position[0], points[index][1] - position[1]
        ),
    )
    points = points[nearest:]
    if len(points) < 3:
        return None
    distance = sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(points[:-1], points[1:])
    )
    return {
        "point_count": len(points),
        "available_distance_m": float(distance),
    }


def route_lane_change_observation(tick_data):
    """Recognize one bounded lateral shift in the live dense route.

    Sequential lane-change routes contain a short diagonal transition and
    then resume the original heading in the adjacent lane.  A junction turn
    does not resume the baseline heading, while an ordinary straight route
    has no sufficiently large diagonal segment.  The observation consumes
    only the live navigation horizon already supplied to the agent.
    """
    if not isinstance(tick_data, dict):
        return None
    position = _xy(tick_data.get("pos"))
    points = [_xy(item) for item in list(tick_data.get("route_horizon_xy") or [])]
    points = [point for point in points if point is not None]
    if position is None or len(points) < 8:
        return None
    nearest = min(
        range(len(points)),
        key=lambda index: math.hypot(
            points[index][0] - position[0], points[index][1] - position[1]
        ),
    )
    points = points[nearest:]
    if len(points) < 8:
        return None

    segments = []
    cumulative = 0.0
    for left, right in zip(points[:-1], points[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        length = math.hypot(dx, dy)
        if length < 0.15:
            continue
        segments.append({
            "start_distance_m": cumulative,
            "length_m": length,
            "heading_rad": math.atan2(dx, -dy),
            "start": left,
            "end": right,
        })
        cumulative += length
    if len(segments) < 7 or cumulative < 48.0:
        return None

    baseline = []
    for segment in segments:
        baseline.append(segment["heading_rad"])
        if segment["start_distance_m"] + segment["length_m"] >= 5.0:
            break
    sin_mean = sum(math.sin(value) for value in baseline) / len(baseline)
    cos_mean = sum(math.cos(value) for value in baseline) / len(baseline)
    baseline_heading = math.atan2(sin_mean, cos_mean)
    deviations = [
        _wrap(segment["heading_rad"] - baseline_heading)
        for segment in segments
    ]

    for index, deviation in enumerate(deviations):
        start_distance = segments[index]["start_distance_m"]
        if not (25.0 <= start_distance <= 47.0):
            continue
        if not (0.40 <= abs(deviation) <= 0.85):
            continue
        direction = 1.0 if deviation > 0.0 else -1.0
        end = index
        while end + 1 < len(segments):
            following = deviations[end + 1]
            if not (
                0.25 <= abs(following) <= 0.95
                and following * direction > 0.0
            ):
                break
            end += 1
        transition_length = sum(
            segment["length_m"] for segment in segments[index:end + 1]
        )
        recovery_length = sum(
            segments[item]["length_m"]
            for item in range(end + 1, min(len(segments), end + 8))
            if abs(deviations[item]) <= 0.14
        )
        start = segments[index]["start"]
        finish = segments[end]["end"]
        axis_x, axis_y = math.sin(baseline_heading), -math.cos(baseline_heading)
        dx, dy = finish[0] - start[0], finish[1] - start[1]
        lateral_shift = axis_x * dy - axis_y * dx
        if (
            4.0 <= transition_length <= 8.0
            and recovery_length >= 4.0
            and 2.5 <= abs(lateral_shift) <= 4.5
        ):
            return {
                "start_distance_m": float(start_distance),
                "heading_change_rad": float(deviation),
                "transition_length_m": float(transition_length),
                "lateral_shift_m": float(lateral_shift),
                "recovery_length_m": float(recovery_length),
                "available_distance_m": float(cumulative),
                "direction": "left" if lateral_shift > 0.0 else "right",
            }
    return None


def route_turn_observation(tick_data, threshold_rad=0.30, sustain_segments=2):
    """Locate the first sustained turn in the live navigation horizon.

    The result is an online geometric observation, not route metadata.  It is
    intentionally expressed as distance along the current dense route so a
    junction rule can approach the stop line instead of reacting to a far
    waypoint chord while still tens of metres away.
    """
    if not isinstance(tick_data, dict):
        return None
    position = _xy(tick_data.get("pos"))
    points = [_xy(item) for item in list(tick_data.get("route_horizon_xy") or [])]
    points = [point for point in points if point is not None]
    if position is None or len(points) < 5:
        return None
    nearest = min(
        range(len(points)),
        key=lambda index: math.hypot(
            points[index][0] - position[0], points[index][1] - position[1]
        ),
    )
    points = points[nearest:]
    if len(points) < 5:
        return None

    segments = []
    cumulative = 0.0
    for left, right in zip(points[:-1], points[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        length = math.hypot(dx, dy)
        if length < 0.15:
            continue
        segments.append({
            "start_distance_m": cumulative,
            "length_m": length,
            "heading_rad": math.atan2(dx, -dy),
        })
        cumulative += length
    if len(segments) < 4:
        return None

    # Average the first few metres as a circular baseline.  This rejects a
    # single noisy navigation node but retains the beginning of a real bend.
    baseline = []
    for segment in segments:
        baseline.append(segment["heading_rad"])
        if segment["start_distance_m"] + segment["length_m"] >= 3.0:
            break
    sin_mean = sum(math.sin(value) for value in baseline) / len(baseline)
    cos_mean = sum(math.cos(value) for value in baseline) / len(baseline)
    baseline_heading = math.atan2(sin_mean, cos_mean)
    changes = [_wrap(item["heading_rad"] - baseline_heading) for item in segments]
    required = max(1, int(sustain_segments))
    threshold = max(0.10, abs(_number(threshold_rad, 0.30)))
    for index in range(0, len(segments) - required + 1):
        window = changes[index:index + required]
        if all(abs(value) >= threshold for value in window) and all(
            value * window[0] > 0.0 for value in window
        ):
            return {
                "distance_to_turn_m": float(segments[index]["start_distance_m"]),
                "heading_change_rad": float(sum(window) / len(window)),
                "direction": "left" if window[0] > 0.0 else "right",
                "sustained_segments": required,
                "available_distance_m": float(cumulative),
            }
    return {
        "distance_to_turn_m": None,
        "heading_change_rad": 0.0,
        "direction": "straight",
        "sustained_segments": 0,
        "available_distance_m": float(cumulative),
    }


class OrdinaryLocalPathController(object):
    """Track a short local polyline with bounded proportional steering."""

    def __init__(self, route_pid_steer=None):
        self.route_pid_steer = route_pid_steer
        self.reset()

    def reset(self):
        self.path_key = None
        self.axis = None
        self.x_m = 0.0
        self.y_m = 0.0
        self.last_timestamp = None
        self.last_error = 0.0
        self.last_steer = 0.0
        self.live_lateral_samples = deque(maxlen=7)
        self.live_lateral_bias_m = 0.0
        self.route_waypoint_phase = None
        self.route_waypoint_last_steer = 0.0
        self.dense_pose_samples = deque(maxlen=9)
        self.dense_odometry_xy = [0.0, 0.0]
        self.dense_last_timestamp = None

    @staticmethod
    def _forward_axis(compass):
        return math.sin(compass), -math.cos(compass)

    def _route_axis(self, tick_data, compass):
        near = _xy(tick_data.get("command_near_xy"))
        far = _xy(tick_data.get("command_far_xy"))
        if near is not None and far is not None:
            dx, dy = far[0] - near[0], far[1] - near[1]
            norm = math.hypot(dx, dy)
            if norm > 1.0:
                return dx / norm, dy / norm
        return self._forward_axis(compass)

    def route_waypoint_step(self, directive, tick_data, fallback_steer=0.0):
        """Rate-limited pure-pursuit steering from live navigation waypoints."""
        observation = route_waypoint_observation(tick_data)
        if observation is None:
            raise ValueError("ordinary live route waypoint inputs unavailable")
        near_distance = observation["near"]["distance_m"]
        near_error = observation["near"]["heading_error_rad"]
        far_distance = observation["far"]["distance_m"]
        far_error = observation["far"]["heading_error_rad"]
        use_far = bool(
            near_distance <= _number(directive.get("near_switch_distance_m"), 2.0)
            or (abs(near_error) > 2.2 and far_distance > near_distance)
        )
        far_lifecycle_switch = bool(
            directive.get("prefer_lower_error_far", False)
            and far_distance <= _number(
                directive.get("far_lifecycle_max_distance_m"), 25.0
            )
            and abs(far_error) + _number(
                directive.get("far_lifecycle_error_margin_rad"), 0.10
            ) < abs(near_error)
        )
        use_far = bool(use_far or far_lifecycle_switch)
        heading_blend_far = _clip(
            _number(directive.get("heading_blend_far"), 0.0), 0.0, 1.0
        )
        blended_heading = False
        if not use_far and heading_blend_far > 0.0:
            # Blend directions on the unit circle rather than averaging raw
            # wrapped angles.  The option is strictly opt-in; historical
            # directives without it retain byte-for-byte near/far selection.
            near_weight = 1.0 - heading_blend_far
            error = math.atan2(
                near_weight * math.sin(near_error)
                + heading_blend_far * math.sin(far_error),
                near_weight * math.cos(near_error)
                + heading_blend_far * math.cos(far_error),
            )
            blended_heading = True
        else:
            error = far_error if use_far else near_error
        speed = max(0.0, _number(tick_data.get("speed"), 0.0))
        kp = _clip(_number(directive.get("steer_kp"), 0.72), 0.2, 1.5)
        model_blend = _clip(_number(directive.get("model_steer_blend"), 0.20), 0.0, 0.6)
        limit = _clip(abs(_number(directive.get("steer_limit"), 0.80)), 0.2, 1.0)
        high_speed_start = _number(directive.get("high_speed_start_mps"), 7.5)
        if speed >= high_speed_start:
            limit = min(
                limit,
                _clip(_number(directive.get("high_speed_steer_limit"), 0.28), 0.15, 0.5),
            )
        requested = (1.0 - model_blend) * kp * error + model_blend * _number(
            fallback_steer, 0.0
        )
        requested = _clip(requested, -limit, limit)
        phase = str(directive.get("phase", "ordinary_route_waypoint"))
        if self.route_waypoint_phase != phase:
            self.route_waypoint_phase = phase
            self.route_waypoint_last_steer = _number(fallback_steer, 0.0)
        rate = _clip(_number(directive.get("steer_rate_per_frame"), 0.12), 0.02, 0.25)
        steer = _clip(
            requested,
            self.route_waypoint_last_steer - rate,
            self.route_waypoint_last_steer + rate,
        )
        self.route_waypoint_last_steer = steer
        return {
            "valid": True,
            "steer": steer,
            "requested_steer": requested,
            "heading_error_rad": error,
            "target": (
                "near_far_blend" if blended_heading
                else "far" if use_far else "near"
            ),
            "heading_blend_far": float(heading_blend_far),
            "far_lifecycle_switch": far_lifecycle_switch,
            "near_distance_m": near_distance,
            "far_distance_m": far_distance,
            "speed_mps": speed,
            "active_steer_limit": limit,
            "source": "online_near_far_rate_limited_pure_pursuit",
        }

    def _dense_filtered_position(self, directive, tick_data, timestamp):
        """Suppress frame-wise GNSS jumps without adding motion lag.

        Each recent sensor position is advanced to the current frame by the
        speed/IMU odometry accumulated after that sample.  The coordinate-wise
        median therefore rejects GNSS outliers while a moving vehicle does not
        inherit the ordinary trailing-window delay.
        """
        raw = _xy(tick_data.get("pos"))
        compass = _number(tick_data.get("compass"), float("nan"))
        now = _number(timestamp, float("nan"))
        if raw is None or not (math.isfinite(compass) and math.isfinite(now)):
            return raw, 0
        if self.dense_last_timestamp is not None:
            dt = _clip(now - self.dense_last_timestamp, 0.0, 0.20)
            speed = max(0.0, _number(tick_data.get("speed"), 0.0))
            forward = self._forward_axis(compass)
            self.dense_odometry_xy[0] += speed * forward[0] * dt
            self.dense_odometry_xy[1] += speed * forward[1] * dt
        self.dense_last_timestamp = now
        odometry = tuple(self.dense_odometry_xy)
        frames = max(1, min(9, int(directive.get("pose_filter_frames", 1))))
        if self.dense_pose_samples.maxlen != frames:
            self.dense_pose_samples = deque(
                list(self.dense_pose_samples)[-frames:], maxlen=frames
            )
        self.dense_pose_samples.append((raw, odometry))
        estimates = [
            (
                sample[0][0] + odometry[0] - sample[1][0],
                sample[0][1] + odometry[1] - sample[1][1],
            )
            for sample in self.dense_pose_samples
        ]
        ordered_x = sorted(item[0] for item in estimates)
        ordered_y = sorted(item[1] for item in estimates)
        middle = len(estimates) // 2
        return (ordered_x[middle], ordered_y[middle]), len(estimates)

    def dense_route_pid_step(
        self, directive, tick_data, fallback_steer=0.0, timestamp=None
    ):
        """Use CARLA PID on a lookahead point from the live dense route."""
        if self.route_pid_steer is None:
            raise ValueError("CARLA route PID callback unavailable")
        raw_position = _xy(tick_data.get("pos"))
        position, pose_sample_count = self._dense_filtered_position(
            directive,
            tick_data,
            timestamp if timestamp is not None else tick_data.get("timestamp"),
        )
        points = [_xy(item) for item in list(tick_data.get("route_horizon_xy") or [])]
        points = [point for point in points if point is not None]
        if position is None or len(points) < 3:
            raise ValueError("ordinary dense route horizon unavailable")
        nearest = min(
            range(len(points)),
            key=lambda index: math.hypot(
                points[index][0] - position[0], points[index][1] - position[1]
            ),
        )
        points = points[nearest:]
        speed = max(0.0, _number(tick_data.get("speed"), 0.0))
        lookahead = _clip(
            _number(directive.get("lookahead_base_m"), 5.0)
            + speed * _number(directive.get("lookahead_speed_s"), 0.45),
            _number(directive.get("lookahead_min_m"), 5.0),
            _number(directive.get("lookahead_max_m"), 9.0),
        )
        cumulative = 0.0
        target_index = min(1, len(points) - 1)
        for index, (left, right) in enumerate(
            zip(points[:-1], points[1:]), start=1
        ):
            cumulative += math.hypot(
                right[0] - left[0], right[1] - left[1]
            )
            target_index = index
            if cumulative >= lookahead:
                break
        target = points[target_index]
        following = points[min(target_index + 1, len(points) - 1)]
        target_route_offset = _number(
            directive.get("target_route_offset_m"), 0.0
        )
        if abs(target_route_offset) > 1e-6:
            tangent_x = following[0] - target[0]
            tangent_y = following[1] - target[1]
            tangent_norm = math.hypot(tangent_x, tangent_y)
            if tangent_norm > 1e-3:
                # Positive offset is the signed left side used by
                # route_lateral_offset().  Shift both points so the public
                # CARLA PID sees the same dense tangent, not a kink.
                offset_x = -tangent_y / tangent_norm * target_route_offset
                offset_y = tangent_x / tangent_norm * target_route_offset
                target = (target[0] + offset_x, target[1] + offset_y)
                following = (
                    following[0] + offset_x,
                    following[1] + offset_y,
                )
        synthetic = dict(tick_data)
        synthetic["pos"] = list(position)
        synthetic["command_near_xy"] = target
        synthetic["command_far_xy"] = following
        limit = _clip(
            abs(_number(directive.get("steer_limit"), 0.48)), 0.2, 0.65
        )
        requested = self.route_pid_steer(
            phase=str(directive.get("phase", "class44_dense_route_follow")),
            tick_data=synthetic,
            route_lateral_error=None,
            target_route_offset=0.0,
            steer_limit=limit,
            fallback_steer=_number(fallback_steer, 0.0),
        )
        requested = _clip(_number(requested, fallback_steer), -limit, limit)
        rate = _clip(
            _number(directive.get("steer_rate_per_frame"), 0.10), 0.02, 0.20
        )
        steer = _clip(
            requested,
            self.route_waypoint_last_steer - rate,
            self.route_waypoint_last_steer + rate,
        )
        self.route_waypoint_last_steer = steer
        return {
            "valid": True,
            "steer": steer,
            "requested_steer": requested,
            "lookahead_m": lookahead,
            "target_xy": list(target),
            "following_xy": list(following),
            "horizon_point_count": len(points),
            "speed_mps": speed,
            "raw_position_xy": list(raw_position),
            "filtered_position_xy": list(position),
            "pose_filter_sample_count": int(pose_sample_count),
            "target_route_offset_m": target_route_offset,
            "source": "carla_public_pid_dense_online_route",
        }

    @staticmethod
    def _validated_waypoints(directive):
        result = []
        for item in directive.get("waypoints", []):
            point = _xy(item)
            if point is None:
                continue
            if result and point[0] <= result[-1][0]:
                raise ValueError("ordinary local-path x coordinates must increase")
            result.append(point)
        if len(result) < 2 or result[0] != (0.0, 0.0):
            raise ValueError("ordinary local path must start at local origin")
        return result

    @staticmethod
    def _interpolate(waypoints, target_x):
        if target_x <= waypoints[0][0]:
            return waypoints[0]
        for left, right in zip(waypoints[:-1], waypoints[1:]):
            if target_x <= right[0]:
                span = right[0] - left[0]
                ratio = (target_x - left[0]) / span
                return target_x, left[1] + ratio * (right[1] - left[1])
        # Keep a forward-facing target after the nominal endpoint. Clamping
        # to the endpoint would put the lookahead point behind an overshooting
        # vehicle and command a sharp turn exactly when heading should settle.
        left, right = waypoints[-2], waypoints[-1]
        slope = (right[1] - left[1]) / (right[0] - left[0])
        return target_x, right[1] + slope * (target_x - right[0])

    def step(self, directive, tick_data, timestamp, fallback_steer=0.0):
        waypoints = self._validated_waypoints(directive)
        compass = _number(tick_data.get("compass"), float("nan"))
        speed = _number(tick_data.get("speed"), 0.0)
        now = _number(timestamp, float("nan"))
        if not (math.isfinite(compass) and math.isfinite(now)):
            raise ValueError("ordinary local-path live motion inputs unavailable")

        use_dense_route_pid = bool(directive.get("use_dense_route_pid"))
        path_key = (
            int(directive.get("class_id", 0)),
            int(directive.get("cycle", 0)),
            str(directive.get("path_id", "")),
        )
        if self.path_key != path_key or self.axis is None:
            self.reset()
            self.path_key = path_key
            self.axis = self._route_axis(tick_data, compass)
            self.last_timestamp = now
            self.last_steer = _number(fallback_steer, 0.0)
        else:
            delta_t = _clip(now - self.last_timestamp, 0.0, 0.20)
            forward = self._forward_axis(compass)
            along = self.axis[0] * forward[0] + self.axis[1] * forward[1]
            lateral = self.axis[0] * forward[1] - self.axis[1] * forward[0]
            # A dense-route directive is parameterized by travelled route arc,
            # not the projection onto the stop-pose tangent.  The latter loses
            # progress on a bend and can postpone rejoin until route end.
            self.x_m += speed * delta_t if use_dense_route_pid else speed * along * delta_t
            self.y_m += speed * lateral * delta_t
            self.last_timestamp = now

        forward = self._forward_axis(compass)
        heading = math.atan2(
            self.axis[0] * forward[1] - self.axis[1] * forward[0],
            self.axis[0] * forward[0] + self.axis[1] * forward[1],
        )
        lookahead = _clip(_number(directive.get("lookahead_m"), 2.5), 1.5, 6.0)
        target_x, target_y = self._interpolate(waypoints, self.x_m + lookahead)
        desired_heading = math.atan2(target_y - self.y_m, max(0.25, target_x - self.x_m))
        error = _wrap(desired_heading - heading)
        kp = _clip(_number(directive.get("steer_kp"), 1.15), 0.2, 2.5)
        kd = _clip(_number(directive.get("steer_kd"), 0.04), 0.0, 0.3)
        requested = kp * error + kd * (error - self.last_error)
        steer_limit = _clip(abs(_number(directive.get("steer_limit"), 0.38)), 0.1, 0.6)
        requested = _clip(requested, -steer_limit, steer_limit)
        final_x, final_y = waypoints[-1]
        steering_source = "speed_compass_route_tangent_odometry"
        route_pid_target = None
        route_lateral = None
        odometry_route_lateral = None
        live_route_lateral = None
        filtered_live_route_lateral = None
        route_pid_min_progress = _number(
            directive.get("route_pid_min_progress_m"), 0.0
        )
        route_pid_active = bool(
            directive.get("use_carla_route_pid")
            and self.route_pid_steer is not None
            and self.x_m >= route_pid_min_progress
        )
        dense_route_feedback = None
        if route_pid_active:
            # ``route_lateral_offset`` can jump when the navigation planner
            # advances its near waypoint.  Freeze its path-start value and
            # add the continuous IMU/speed odometry displacement instead.
            # This keeps CARLA PID's feedback in one coordinate frame.
            odometry_route_lateral = (
                _number(directive.get("origin_route_lateral_m"), 0.0)
                + self.y_m
            )
            route_lateral = odometry_route_lateral
            if bool(directive.get("fuse_live_route_lateral")):
                live_route_lateral = (
                    dense_route_lateral_offset(tick_data)
                    if use_dense_route_pid
                    else route_lateral_offset(tick_data)
                )
                if live_route_lateral is not None:
                    self.live_lateral_samples.append(float(live_route_lateral))
                if self.live_lateral_samples:
                    ordered = sorted(self.live_lateral_samples)
                    live_median = ordered[len(ordered) // 2]
                    filtered_live_route_lateral = live_median
                    innovation = live_median - (
                        odometry_route_lateral + self.live_lateral_bias_m
                    )
                    # GNSS-derived position can jump by almost a metre in one
                    # frame while stopped.  A persistent bias is nevertheless
                    # real evidence that speed/IMU integration has drifted.
                    # Correct no more than 5 cm per frame: isolated jumps are
                    # rejected by the median, while sustained metre-scale
                    # drift is removed before the construction corridor.
                    self.live_lateral_bias_m += _clip(innovation, -0.05, 0.05)
                    route_lateral = (
                        odometry_route_lateral + self.live_lateral_bias_m
                    )
                    if bool(directive.get("trust_live_route_lateral")):
                        # Class05's long staged path can accumulate several
                        # metres of wheel/heading integration error before
                        # rejoin.  The seven-sample median has already removed
                        # single-frame GNSS/navigation jumps, so this opt-in
                        # path must control the measured lane position rather
                        # than slowly chasing it with the generic bias filter.
                        route_lateral = live_median
            corridor = _number(directive.get("radar_corridor_m"), 0.0)
            hold_end = _number(directive.get("hold_end_m"), final_x)
            rejoin_end = _number(directive.get("rejoin_end_m"), hold_end)
            if bool(directive.get("yield_pocket")):
                route_pid_target = _number(directive.get("pocket_absolute_m"), 0.0)
            elif (
                self.x_m < hold_end
                and bool(directive.get("pid_follow_waypoint_entry", False))
            ):
                # Most construction paths intentionally acquire their full
                # corridor immediately.  Class05 has a measured straight
                # approach before its finite conflict, so its explicit local
                # waypoints must own entry timing instead of being ignored by
                # the absolute route-offset PID target.
                route_pid_target = _number(
                    directive.get("origin_route_lateral_m"), 0.0
                ) + target_y
            elif self.x_m < hold_end:
                # The construction-side warning mesh starts before a short
                # pure-pursuit curve can converge.  Ask CARLA's public PID for
                # the absolute online corridor immediately; longitudinal
                # progress still owns the later rejoin transition.
                route_pid_target = corridor
            elif bool(directive.get("pid_step_rejoin")):
                # CARLA PID already provides closed-loop smoothing. Feeding
                # it another slowly moving cosine target caused a second lag
                # and left the vehicle partly in the opposing lane.
                route_pid_target = _number(
                    directive.get("emergency_rejoin_target_m"), 0.0
                )
            elif self.x_m < rejoin_end and rejoin_end > hold_end:
                fraction = _clip((self.x_m - hold_end) / (rejoin_end - hold_end), 0.0, 1.0)
                route_pid_target = corridor * 0.5 * (1.0 + math.cos(math.pi * fraction))
            else:
                route_pid_target = 0.0
            if use_dense_route_pid and route_horizon_observation(tick_data) is not None:
                dense_route_feedback = self.dense_route_pid_step(
                    {
                        "phase": str(directive.get(
                            "dense_route_pid_phase",
                            directive.get("route_pid_phase", "ordinary_local_path"),
                        )),
                        "target_route_offset_m": route_pid_target,
                        "lookahead_base_m": _number(
                            directive.get("dense_lookahead_base_m"), 4.5
                        ),
                        "lookahead_speed_s": _number(
                            directive.get("dense_lookahead_speed_s"), 0.35
                        ),
                        "lookahead_min_m": _number(
                            directive.get("dense_lookahead_min_m"), 4.5
                        ),
                        "lookahead_max_m": _number(
                            directive.get("dense_lookahead_max_m"), 7.5
                        ),
                        "steer_limit": steer_limit,
                        "steer_rate_per_frame": _number(
                            directive.get("steer_rate_per_frame"), 0.06
                        ),
                        "pose_filter_frames": int(
                            directive.get("dense_pose_filter_frames", 5)
                        ),
                    },
                    tick_data,
                    fallback_steer=requested,
                    timestamp=now,
                )
                requested = _number(dense_route_feedback.get("steer"), requested)
                steering_source = str(dense_route_feedback.get("source"))
            else:
                requested = self.route_pid_steer(
                    phase=str(directive.get("route_pid_phase", "ordinary_local_path")),
                    tick_data=tick_data,
                    route_lateral_error=route_lateral,
                    target_route_offset=route_pid_target,
                    steer_limit=steer_limit,
                    fallback_steer=requested,
                )
                requested = _clip(_number(requested, requested), -steer_limit, steer_limit)
                steering_source = "carla_public_pid_route_offset"
        rate = _clip(_number(directive.get("steer_rate_per_frame"), 0.06), 0.01, 0.15)
        steer = _clip(requested, self.last_steer - rate, self.last_steer + rate)
        self.last_error = error
        self.last_steer = steer

        lateral_tolerance = _clip(
            _number(directive.get("finish_lateral_tolerance_m"), 0.50), 0.20, 1.0
        )
        heading_tolerance = _clip(
            _number(directive.get("finish_heading_tolerance_rad"), 0.15), 0.05, 0.35
        )
        if (
            bool(directive.get("finish_use_live_route_lateral"))
            and filtered_live_route_lateral is None
        ):
            live_route_lateral = (
                dense_route_lateral_offset(tick_data)
                if use_dense_route_pid
                else route_lateral_offset(tick_data)
            )
            if live_route_lateral is not None:
                self.live_lateral_samples.append(float(live_route_lateral))
            if self.live_lateral_samples:
                ordered = sorted(self.live_lateral_samples)
                filtered_live_route_lateral = ordered[len(ordered) // 2]
        finish_lateral_error = self.y_m - final_y
        if (
            bool(directive.get("finish_use_live_route_lateral"))
            and filtered_live_route_lateral is not None
        ):
            finish_lateral_error = filtered_live_route_lateral
        finish_heading_error = heading
        if bool(directive.get("finish_use_live_route_heading")):
            live_axis = self._route_axis(tick_data, compass)
            live_dot = live_axis[0] * forward[0] + live_axis[1] * forward[1]
            live_cross = (
                live_axis[0] * forward[1] - live_axis[1] * forward[0]
            )
            finish_heading_error = math.atan2(live_cross, live_dot)
        done = bool(
            self.x_m >= final_x
            and abs(finish_lateral_error) <= lateral_tolerance
            and abs(finish_heading_error) <= heading_tolerance
        )
        return {
            "valid": True,
            "path_key": list(path_key),
            "x_m": self.x_m,
            "y_m": self.y_m,
            "heading_error_rad": error,
            "heading_rad": heading,
            "target_x_m": target_x,
            "target_y_m": target_y,
            "steer": steer,
            "done": done,
            "source": steering_source,
            "route_pid_target_m": route_pid_target,
            "route_lateral_m": route_lateral,
            "odometry_route_lateral_m": (
                odometry_route_lateral
                if route_pid_active else None
            ),
            "live_route_lateral_m": live_route_lateral,
            "filtered_live_route_lateral_m": filtered_live_route_lateral,
            "live_lateral_bias_m": self.live_lateral_bias_m,
            "finish_lateral_error_m": finish_lateral_error,
            "finish_heading_error_rad": finish_heading_error,
            "dense_route_feedback": dense_route_feedback,
        }
