import math
import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (KeepVelocity, StopVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle, DriveDistance)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (ReverseVehicleBrakeCriterion, 
                                                                    ReverseVehicleBypassCriterion, 
                                                                    ReverseVehicleResumeCriterion,
                                                                    MinTTCAutoCriterion)
from srunner.scenarios.basic_scenario import BasicScenario

class EgoSpeedControl(py_trees.behaviour.Behaviour):
    """
    二合一节点：管理主车速度。若检测到人为接管（刹车/油门 > 0.05），则停止控制并返回 SUCCESS。
    """
    def __init__(self, ego_vehicle, target_velocity=10.0, name="EgoSpeedControl"):
        super(EgoSpeedControl, self).__init__(name)
        self.ego_vehicle = ego_vehicle
        self.target_velocity = target_velocity
        self._taken_over = False

    def update(self):
        if not self.ego_vehicle or not self.ego_vehicle.is_alive:
            return py_trees.common.Status.FAILURE
        
        if self._taken_over:
            return py_trees.common.Status.SUCCESS

        control = self.ego_vehicle.get_control()
        if control.throttle > 0.05 or control.brake > 0.05:
            self._taken_over = True
            return py_trees.common.Status.SUCCESS

        # 保持目标速度
        transform = self.ego_vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        velocity = carla.Vector3D(math.cos(yaw) * self.target_velocity, 
                                  math.sin(yaw) * self.target_velocity, 0)
        self.ego_vehicle.set_target_velocity(velocity)
        return py_trees.common.Status.RUNNING

class ParkVehicle(py_trees.behaviour.Behaviour):
    """彻底锁死车辆：挂空挡、拉手刹，防止溜车"""
    def __init__(self, actor, name="ParkVehicle"):
        super(ParkVehicle, self).__init__(name)
        self.actor = actor

    def update(self):
        if not self.actor or not self.actor.is_alive:
            return py_trees.common.Status.FAILURE
        
        control = carla.VehicleControl(brake=1.0, hand_brake=True, manual_gear_shift=True, gear=0)
        self.actor.apply_control(control)
        return py_trees.common.Status.SUCCESS

class KeepReverseVelocity(py_trees.behaviour.Behaviour):
    """
    让车辆以指定速度倒车（沿车头反方向运动）
    """
    def __init__(self, actor, target_speed, name="KeepReverseVelocity"):
        super(KeepReverseVelocity, self).__init__(name)
        self.actor = actor
        self.target_speed = target_speed
        self.count = 0

    def update(self):
        if self.count == 0:
            if not self.actor or not self.actor.is_alive:
                return py_trees.common.Status.FAILURE

            transform = self.actor.get_transform()
            yaw = math.radians(transform.rotation.yaw)

            # 车辆“前向”单位向量
            forward_x = math.cos(yaw)
            forward_y = math.sin(yaw)

            # 倒车 = 沿前向的反方向给速度
            velocity = carla.Vector3D(
                -forward_x * self.target_speed,
                -forward_y * self.target_speed,
                0.0
            )
            self.actor.set_target_velocity(velocity)
            self.count += 1
            return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.RUNNING

class ReverseVehicle(BasicScenario):
    """
    场景：自车直行，右侧车辆倒车横穿，左侧自行车前行
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):

        self._init_speed = float(
            config.other_parameters.get("init_speed", {}).get("value", 10.0)
        )
        self._reverse_speed = float(
            config.other_parameters.get("reverse_speed", {}).get("value", 3.0)
        )
        self._trigger_distance = float(
            config.other_parameters.get("trigger_distance", {}).get("value", 40)
        )
        super(ReverseVehicle, self).__init__("ReverseVehicle", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

        if self.ego_vehicles:
            ego = self.ego_vehicles[0]
            yaw = math.radians(ego.get_transform().rotation.yaw)
            ego.set_target_velocity(carla.Vector3D(math.cos(yaw) * self._init_speed, math.sin(yaw) * self._init_speed))

    def _initialize_actors(self, config):
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
        root = py_trees.composites.Parallel(
            "ReverseVehicleBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )

        if not self.ego_vehicles or len(self.other_actors) < 3:
            return root

        ego = self.ego_vehicles[0]
        right_car = self.other_actors[0]
        left_bike = self.other_actors[1]
        front_vehicle = self.other_actors[2]

        # 1. 主车控制
        root.add_child(EgoSpeedControl(ego, target_velocity=self._init_speed))
   
        # 距离触发
        trigger_dist = InTriggerDistanceToVehicle(right_car, ego, distance=self._trigger_distance)

        # 2. 右侧车辆逻辑：触发 -> 倒车至指定Y坐标 -> 停止 -> 驻车
        right_car_seq = py_trees.composites.Sequence("RightCarSequence")
        right_move_par = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        right_move_par.add_children([
            KeepReverseVelocity(right_car, self._reverse_speed),
            DriveDistance(right_car, 8.3)
        ])

        right_car_seq.add_children([
            trigger_dist,
            right_move_par,
            StopVehicle(right_car, 1.0),
            ParkVehicle(right_car)
        ])

        # 3. 左侧自行车逻辑：触发 -> 前行10米 -> 停止
        left_bike_seq = py_trees.composites.Sequence("LeftBikeSequence")
        left_move_par = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        left_move_par.add_children([
            KeepVelocity(left_bike, 1.45),
            DriveDistance(left_bike, 10.0)
        ])

        left_bike_seq.add_children([
            trigger_dist,
            left_move_par,
            StopVehicle(left_bike, 1.0)
        ])

        # 4. 新增前方障碍物车辆：触发 -> 匀速前进 -> 停止
        front_vehicle_seq = py_trees.composites.Sequence("FrontVehicleSequence")
        front_vehicle_move_par = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        front_vehicle_move_par.add_children([
            KeepVelocity(front_vehicle, 2.0),
            DriveDistance(front_vehicle, 9.0)
        ])

        front_vehicle_seq.add_children([
            trigger_dist,
            front_vehicle_move_par,
            StopVehicle(front_vehicle, 1.0),
            ParkVehicle(front_vehicle)
        ])

        root.add_children([right_car_seq, left_bike_seq, front_vehicle_seq])
        return root

    def _create_test_criteria(self):
        ego, hazard = self.ego_vehicles[0], self.other_actors[0]
        goal_loc = carla.Location(x=107, y=134.423, z=0.5)
        route_y = 134.423

        

        return [
            ReverseVehicleBrakeCriterion(ego, hazard, trigger_x=60.0, brake_threshold=0.2, min_brake_duration=0.3),
            ReverseVehicleBypassCriterion(ego, hazard, route_center_y=route_y, lateral_threshold=0.8, pass_x_margin=0.5),
            ReverseVehicleResumeCriterion(ego, goal_loc, route_center_y=route_y, goal_dist_threshold=3.0, center_recover_threshold=4.0, min_resume_speed=1.0),
            MinTTCAutoCriterion(actor=self.ego_vehicles[0],other_actors=self.other_actors,distance_threshold=40.0,forward_angle_deg=140.0,terminate_on_failure=False)
        ]

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
