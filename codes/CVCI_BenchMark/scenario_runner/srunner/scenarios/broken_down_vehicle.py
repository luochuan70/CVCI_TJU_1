import py_trees
import carla
import sys
import math

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToLocation
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (Criterion, 
    BrokenDownVehicleBrakeCriterion, BrokenDownVehicleBypassCriterion, BrokenDownVehicleResumeCriterion,MinTTCAutoCriterion)



# --- 行为 1：维持自车速度 ---
class MaintainEgoVelocity(AtomicBehavior):
    def __init__(self, ego_vehicle, target_speed, goal_location, name="MaintainEgoVelocity"):
        super(MaintainEgoVelocity, self).__init__(name)
        self._ego_vehicle = ego_vehicle
        self._target_speed = target_speed 
        self._goal_location = goal_location
        self._has_been_controlled = False 

    def update(self):
        if self._ego_vehicle is None or not self._ego_vehicle.is_alive:
            return py_trees.common.Status.FAILURE
        
        ego_loc = self._ego_vehicle.get_location()
        if ego_loc.distance(self._goal_location) < 5.0:
            return py_trees.common.Status.SUCCESS

        if self._has_been_controlled:
            return py_trees.common.Status.RUNNING

        control = self._ego_vehicle.get_control()
        v = self._ego_vehicle.get_velocity()
        current_speed = math.sqrt(v.x**2 + v.y**2)

        if current_speed > 2.0 and (abs(control.throttle) > 0.2 or abs(control.brake) > 0.2):
            self._has_been_controlled = True
            return py_trees.common.Status.RUNNING

        self._ego_vehicle.set_target_velocity(carla.Vector3D(x=self._target_speed, y=0, z=0))
        return py_trees.common.Status.RUNNING

# --- 行为 2：NPC 触发后匀速行驶（替代原有的坐标同步） ---
class NPCConstantDrive(AtomicBehavior):
    def __init__(self, actor, ego_vehicle, target_speed, trigger_loc, goal_loc, name="NPCConstantDrive"):
        super(NPCConstantDrive, self).__init__(name)
        self._actor = actor
        self._ego_vehicle = ego_vehicle
        self._target_speed = target_speed # 使用自车的初速度
        self._trigger_loc = trigger_loc
        self._goal_loc = goal_loc
        self._triggered = False

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.FAILURE

        ego_loc = self._ego_vehicle.get_location()

        # 场景结束逻辑
        if ego_loc.distance(self._goal_loc) < 5.0:
            return py_trees.common.Status.SUCCESS

        # 触发判断：自车经过触发点
        if not self._triggered:
            # 这里使用 X 轴判断（假设车辆沿 X 轴负方向行驶，即 x 减小）
            if ego_loc.x <= self._trigger_loc.x+50.0: # 提前50米触发
                self._triggered = True
            else:
                # 触发前保持静止
                self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
                return py_trees.common.Status.RUNNING

        # 触发后，持续设置速度
        self._actor.set_target_velocity(carla.Vector3D(x=self._target_speed, y=0, z=0))
        return py_trees.common.Status.RUNNING

# --- 主场景类 ---
class BrokenDownVehicle(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        self._xml_speed = config.ego_vehicles[0].speed
        # 定义关键坐标点
        self.goal_location = carla.Location(x=-152.7, y=-12.0, z=0.5) 
        self.trigger_location = carla.Location(x=-80.0, y=-12.0, z=0.5) # 自车开到这里，NPC启动
        
        super(BrokenDownVehicle, self).__init__("BrokenDownVehicle", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        for actor_conf in config.other_actors:
            actor = CarlaDataProvider.request_new_actor(actor_conf.model, actor_conf.transform)
            if actor:
                actor.set_simulate_physics(True) # 保持物理开启
                self.other_actors.append(actor)

    def _create_behavior(self):
        root = py_trees.composites.Parallel("ScenarioRoot", 
                                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        ego = self.ego_vehicles[0]
        # 统一速度矢量方向（假设 X 轴负方向为前进方向）
        target_v = -abs(self._xml_speed) if self._xml_speed != 0 else -10.0
        
        # 1. 自车控速
        root.add_child(MaintainEgoVelocity(ego, target_v, self.goal_location))

        # 2. 右侧车道 NPC：由原来的 Sync 改为现在的 ConstantDrive
        # 假设 config 里前几个 NPC 是右侧车道的车辆
        for i in range(min(3, len(self.other_actors))):
            if self.other_actors[i]:
                root.add_child(NPCConstantDrive(
                    self.other_actors[i], 
                    ego, 
                    target_speed=target_v, 
                    trigger_loc=self.trigger_location,
                    goal_loc=self.goal_location
                ))

        # 3. 基础触发器
        root.add_child(InTriggerDistanceToLocation(ego, self.goal_location, distance=3.0))
        
        return root

    def _create_test_criteria(self):
        # ... 保持原有的评估标准代码不变 ...
        criteria = []
        ego = self.ego_vehicles[0]
        hazard = self.other_actors[4] if len(self.other_actors) > 4 else None
        criteria.append(BrokenDownVehicleBrakeCriterion(ego, hazard))
        if hazard:
            criteria.append(BrokenDownVehicleBypassCriterion(ego, hazard))
        criteria.append(BrokenDownVehicleResumeCriterion(ego, self.goal_location))
        criteria.append(
            MinTTCAutoCriterion(actor=self.ego_vehicles[0],other_actors=self.other_actors,distance_threshold=40.0,forward_angle_deg=140.0,terminate_on_failure=False)
        )
        return criteria