import py_trees
import carla
import math
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import KeepVelocity, StopVehicle
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToLocation
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, Criterion
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.timer import GameTime

# 导入您定义的三个 Criteria
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    EbikeDetectionAndDecelerateCriterion,
    PedestrianDetectionAndStopCriterion,
    ResumeAfterPedestrianCriterion,
    MinTTCAutoCriterion
)


class EbikeAndPedestrianCross(BasicScenario):
    """
    场景描述：主车沿路线行驶，电瓶车和行人从侧面斜穿过马路。
    评估标准：
    1. 识别电瓶车并减速：25分
    2. 识别行人并刹车：50分
    3. 离开风险区并恢复通行：25分
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        # ========== 读取 XML 参数 ==========
        self._init_speed_kmh = float(
            config.other_parameters.get("init_speed", {}).get("value", 40.0)
        )
        self._init_speed_ms = self._init_speed_kmh / 3.6
        
        self._difficulty = config.other_parameters.get("difficulty", {}).get("value", "level2")
        
        self._ebike_speed = float(
            config.other_parameters.get("ebike_speed", {}).get("value", 15.0)
        )
        
        self._pedestrian_speed = float(
            config.other_parameters.get("pedestrian_speed", {}).get("value", 1.5)
        )
        
        trigger_transform = config.trigger_points[0]
        self._trigger_location = trigger_transform.location
        
        # 根据难度设置刹车判定阈值
        if self._difficulty == "level1":
            self._brake_threshold = 0.10
            self._stop_speed_threshold = 2.0
            self._min_stop_duration = 0.2
            self._max_response_time = 15.0
        elif self._difficulty == "level2":
            self._brake_threshold = 0.15
            self._stop_speed_threshold = 1.5
            self._min_stop_duration = 0.3
            self._max_response_time = 12.0
        else:  # level3
            self._brake_threshold = 0.20
            self._stop_speed_threshold = 1.0
            self._min_stop_duration = 0.5
            self._max_response_time = 10.0
        
        self.timeout = timeout
        super(EbikeAndPedestrianCross, self).__init__(
            "EbikeAndPedestrianCross",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )
        
        # ========== 设置自车初始速度 ==========
        if self.ego_vehicles:
            ego = self.ego_vehicles[0]
            yaw = math.radians(ego.get_transform().rotation.yaw)
            ego.set_target_velocity(carla.Vector3D(
                math.cos(yaw) * self._init_speed_ms,
                math.sin(yaw) * self._init_speed_ms
            ))

    def _initialize_actors(self, config):
        """
        根据 XML 配置自动生成 Actor
        """
        for i, actor_conf in enumerate(config.other_actors):
            actor = CarlaDataProvider.request_new_actor(actor_conf.model, actor_conf.transform)
            if actor:
                self.other_actors.append(actor)

    def _create_behavior(self):
        """
        封装控制逻辑
        """
        ego = self.ego_vehicles[0]
        ebike = self.other_actors[0] if len(self.other_actors) > 0 else None
        pedestrian = self.other_actors[1] if len(self.other_actors) > 1 else None
        
        root = py_trees.composites.Parallel(
            "CrossBehavior", 
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )

        if pedestrian:
            ped_sequence = py_trees.composites.Sequence("PedestrianBehavior")
            ped_trigger = InTriggerDistanceToLocation(ego, self._trigger_location, distance=1.5)
            ped_walk = KeepVelocity(pedestrian, target_velocity=self._pedestrian_speed, duration=18.0)
            ped_stop = StopVehicle(pedestrian, 1.0)
            ped_sequence.add_children([ped_trigger, ped_walk, ped_stop])

        if ebike:
            bike_sequence = py_trees.composites.Sequence("EbikeBehavior")
            bike_trigger = InTriggerDistanceToLocation(ego, self._trigger_location, distance=1.5)
            bike_drive = KeepVelocity(ebike, target_velocity=self._ebike_speed, duration=10.0)
            bike_stop = StopVehicle(ebike, 1.0)
            bike_sequence.add_children([bike_trigger, bike_drive, bike_stop])

        if ebike and pedestrian:
            root.add_children([ped_sequence, bike_sequence])
        
        return root

    def _create_test_criteria(self):
        """
        定义场景测试通过的标准
        """
        criteria = []
        ego = self.ego_vehicles[0]
        ebike = self.other_actors[0] if len(self.other_actors) > 0 else None
        pedestrian = self.other_actors[1] if len(self.other_actors) > 1 else None
        
        goal_location = carla.Location(x=-49.379311, y=-34.194073, z=0.5)
        
        # ========== 1. 识别电瓶车并减速（25分） ==========
        if ebike:
            criteria.append(
                EbikeDetectionAndDecelerateCriterion(
                    actor=ego,
                    hazard_actor=ebike,
                    brake_threshold=0.05,
                    throttle_reduction=0.2,
                    min_decelerate_duration=0.3,
                    max_response_time=8.0,
                    terminate_on_failure=False
                )
            )
        
        # ========== 2. 识别行人并刹车（50分） ==========
        if pedestrian:
            criteria.append(
                PedestrianDetectionAndStopCriterion(
                    actor=ego,
                    hazard_actor=pedestrian,
                    brake_threshold=self._brake_threshold,
                    stop_speed_threshold=self._stop_speed_threshold,
                    min_stop_duration=self._min_stop_duration,
                    max_response_time=self._max_response_time,
                    terminate_on_failure=False
                )
            )
        
        # ========== 3. 离开风险区并恢复通行（25分） ==========
        criteria.append(
            ResumeAfterPedestrianCriterion(
                actor=ego,
                hazard_actor=pedestrian,
                goal_location=goal_location,
                goal_dist_threshold=50.0,
                min_resume_speed=0.0,
                terminate_on_failure=False
            )
        )
        criteria.append(
            MinTTCAutoCriterion(actor=self.ego_vehicles[0],other_actors=self.other_actors,distance_threshold=40.0,forward_angle_deg=140.0,terminate_on_failure=False)
        )
        
        # ========== 碰撞检测 ==========
        criteria.append(CollisionTest(ego))
        
        return criteria

    def __del__(self):
        """清理资源"""
        self.remove_all_actors()