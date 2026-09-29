import py_trees
import carla
import math
import numpy as np
# 导入 Scenario_Runner 的核心类
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import WaypointFollower
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToLocation, DriveDistance
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (IntersectionCollisionLeftTurnBrakeCriterion, 
                                                                     IntersectionCollisionLeftTurnResumeCriterion,
                                                                     MinTTCAutoCriterion)


class EgoSpeedControl(py_trees.behaviour.Behaviour):
    """
    主车速度保持节点（适合有坡度）
    - 未接管时：仅做纵向速度保持
    - 检测到人工输入后：永久放权并返回 SUCCESS
    - 不使用 set_target_velocity
    - 不修改 steer，避免覆盖人工转向/route 跟踪
    """
    def __init__(
        self,
        ego_vehicle,
        target_speed=10.0,
        throttle_gain=0.20,
        brake_gain=0.10,
        max_throttle=1,
        max_brake=0.50,
        takeover_steer_threshold=0.02,
        takeover_throttle_threshold=0.02,
        takeover_brake_threshold=0.02,
        name="EgoSpeedControl"
    ):
        super(EgoSpeedControl, self).__init__(name)
        self.ego_vehicle = ego_vehicle
        self.target_speed = target_speed

        self.throttle_gain = throttle_gain
        self.brake_gain = brake_gain
        self.max_throttle = max_throttle
        self.max_brake = max_brake

        self.takeover_steer_threshold = takeover_steer_threshold
        self.takeover_throttle_threshold = takeover_throttle_threshold
        self.takeover_brake_threshold = takeover_brake_threshold

        self._taken_over = False

    def _get_speed(self):
        v = self.ego_vehicle.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def update(self):
        if not self.ego_vehicle or not self.ego_vehicle.is_alive:
            return py_trees.common.Status.FAILURE

        if self._taken_over:
            return py_trees.common.Status.SUCCESS

        current_control = self.ego_vehicle.get_control()

        # 接管判定：方向/油门/刹车任一超过阈值，立即放权
        # 注意：这仍然不是最理想的“原始输入检测”，但比只看 throttle/brake 好很多
        if (
            abs(current_control.steer) > self.takeover_steer_threshold or
            current_control.throttle > self.takeover_throttle_threshold or
            current_control.brake > self.takeover_brake_threshold
        ):
            print("[EgoSpeedControl] Manual takeover detected, release control.")
            self._taken_over = True
            return py_trees.common.Status.SUCCESS

        current_speed = self._get_speed()
        speed_error = self.target_speed - current_speed

        new_control = carla.VehicleControl()

        # 不碰方向，避免覆盖人工/上层横向控制
        new_control.steer = current_control.steer
        new_control.hand_brake = False
        new_control.reverse = False
        new_control.manual_gear_shift = False

        if speed_error >= 0.0:
            # 速度偏低：补油
            throttle_cmd = np.clip(speed_error * self.throttle_gain, 0.0, self.max_throttle)
            new_control.throttle = float(throttle_cmd)
            new_control.brake = 0.0
        else:
            # 速度偏高：轻刹
            brake_cmd = np.clip((-speed_error) * self.brake_gain, 0.0, self.max_brake)
            new_control.throttle = 0.0
            new_control.brake = float(brake_cmd)

        self.ego_vehicle.apply_control(new_control)
        return py_trees.common.Status.RUNNING


class IntersectionCollisionLeftTurn(BasicScenario):
    """
    场景描述：自车在Route模式下以约 40km/h 直行。
    NPC从左侧进入路口并左转。若自车不避让，将在终点发生碰撞。
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):

        self._init_speed = float(
            config.other_parameters.get("init_speed", {}).get("value", 10.0)
        )
        self._npc_speed = float(
            config.other_parameters.get("npc_speed", {}).get("value", 11.0)
        )
        self._trigger_npc_point = carla.Location(
            x=-35.0,
            y=205.1,
            z=2.5
        )
        super(IntersectionCollisionLeftTurn, self).__init__(
            "IntersectionCollisionLeftTurn",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

        self.scenario_type = "IntersectionCollisionLeftTurn"

        if self.ego_vehicles and self.ego_vehicles[0]:
            ego = self.ego_vehicles[0]
            transform = ego.get_transform()
            yaw = transform.rotation.yaw * (math.pi / 180)
            velocity = carla.Vector3D(
                math.cos(yaw) * self._init_speed,
                math.sin(yaw) * self._init_speed,
                0
            )
            ego.set_target_velocity(velocity)


    def _initialize_actors(self, config):
        """
        根据 XML 配置自动生成 Actor
        """
        
        # 使用 standard 模式从 config 生成 Actor，这能避免重复生成和物理冲突
        for actor_conf in config.other_actors:
            actor = CarlaDataProvider.request_new_actor(actor_conf.model, actor_conf.transform)
            if actor:
                # 开启灯光（推荐：示宽灯 + 近光灯）
                actor.set_light_state(carla.VehicleLightState(
                    carla.VehicleLightState.Position |
                    carla.VehicleLightState.LowBeam 
                ))

                self.other_actors.append(actor)
        if self.ego_vehicles:
            ego = self.ego_vehicles[0]
            ego.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Position |
                carla.VehicleLightState.LowBeam 
            ))


    def _create_behavior(self):
        """
        构建行为树：手动计算圆弧路径，并强制投影为合法的 Waypoint
        """
        # ————————场景调节条件——————————
        # NPC 速度
        self._npc_speed = self._npc_speed

        # 碰撞点
        self._collision_location = carla.Location(
            x=43.2,
            y=205.3,
            z=0.5
        )
        # --------------------------------------------

        all_traffic_lights = CarlaDataProvider.get_world().get_actors().filter('*traffic_light*')
        for light in all_traffic_lights:
            # 锁定绿灯 100 秒
            light.set_state(carla.TrafficLightState.Green)
            light.set_green_time(1000.0)

        root = py_trees.composites.Parallel(
            "IntersectionCollisionLeftTurnBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )

        # 安全检查
        if not self.ego_vehicles or len(self.other_actors) < 1:
            return root

        ego = self.ego_vehicles[0]
        left_car = self.other_actors[0]

        # ================= 分支1: 主车控制逻辑 =================
        ego_control_node = EgoSpeedControl(ego, self._init_speed)
        root.add_child(ego_control_node)

        # ===============================
        # 2️⃣ NPC 行为
        # ===============================
        npc_sequence = py_trees.composites.Sequence("NPCSequence")

        # 触发条件
        trigger = InTriggerDistanceToLocation(
            ego,
            self._trigger_npc_point,
            distance=3.0,
            name="TriggerDistance"
        )

        # ===============================
        # 3️⃣ NPC 左转（WaypointFollower）
        # ===============================
        amap = CarlaDataProvider.get_map()

        entry_loc = carla.Location(x=24.5, y=185.4, z=0.5)
        end_loc = carla.Location(x=43.2, y=205.3, z=0.5)

        custom_plan = []
        # 左转
        # Bézier 曲线
        p0 = entry_loc
        p1 = carla.Location(x=24.5, y=205.3, z=0.5)
        p2 = end_loc

        for i in range(1, 40):
            t = i / 40.0
            x = (1-t)**2*p0.x + 2*(1-t)*t*p1.x + t**2*p2.x
            y = (1-t)**2*p0.y + 2*(1-t)*t*p1.y + t**2*p2.y

            loc = carla.Location(x=x, y=y, z=0.5)
            wp = amap.get_waypoint(loc, project_to_road=True)
            wp.transform.location = loc

            custom_plan.append((wp, 0))  # RoadOption.LEFT 也行
        
        # 到达终点后直行
        for j in range(1, 80):   # 延伸80米
            x = end_loc.x + j
            y = end_loc.y

            loc = carla.Location(x=x, y=y, z=0.5)
            wp = amap.get_waypoint(loc, project_to_road=True)
            wp.transform.location = loc

            custom_plan.append((wp, 4))  # LANEFOLLOW

        npc_move = WaypointFollower(
            left_car,
            self._npc_speed,
            plan=custom_plan
        )

        npc_sequence.add_child(trigger)
        npc_sequence.add_child(npc_move)   

        root.add_child(npc_sequence)
        ego_drive_distance = DriveDistance(ego, 100)
        root.add_child(ego_drive_distance)
        return root

    def _create_test_criteria(self):
        criteria = []

        ego = self.ego_vehicles[0]
        # hazard = self.other_actors[0]   # 你的占道车辆
        goal_loc = carla.Location(x=43.2, y=205.3, z=0.5)

        criteria.append(
            IntersectionCollisionLeftTurnBrakeCriterion(
                actor=self.ego_vehicles[0],
                hazard_actor=self.other_actors[0],
            )
        )

        criteria.append(
            IntersectionCollisionLeftTurnResumeCriterion(
                actor=ego,
                goal_location=goal_loc,
                route_center_x=40,
                goal_dist_threshold=3.0,
                center_recover_threshold=2.0,
                min_resume_speed=1.0
            )
        )
        criteria.append(
            MinTTCAutoCriterion(
                actor=self.ego_vehicles[0],
                other_actors=self.other_actors,
                distance_threshold=40.0,
                forward_angle_deg=140.0,
                terminate_on_failure=False)
        )
        return criteria

    def __del__(self):
        """
        清理工作：移除生成的 Actor
        """
        self.remove_all_actors()

# =========================================================
# 动态注册场景到 Route 
# =========================================================
try:
    from srunner.scenarios import route_scenario
    route_scenario.IntersectionCollisionLeftTurn = IntersectionCollisionLeftTurn
except ImportError:
    print("Warning: Could not import route_scenario for dynamic registration.")