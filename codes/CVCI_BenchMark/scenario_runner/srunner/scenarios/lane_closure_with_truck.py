import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    CollisionTest,
    DecelerationForConstructionTest,
    RoutePassCompletionTest,
    MinTTCAutoCriterion
)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario


class LaneClosureWithTruck(BasicScenario):

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        self.timeout = timeout

        # 1. 从 XML 动态读取参数（自带默认值兜底）
        params = getattr(config, 'other_parameters', {})
        self._initial_speed_kph = float(params.get('init_speed', {}).get('value', 130.0))
        self._truck_distance    = float(params.get('truck_distance', {}).get('value', 55.0))
        self._truck_offset      = float(params.get('truck_lateral_offset', {}).get('value', 0.5))
        self._cone_start_dist   = float(params.get('cone_start_distance', {}).get('value', 30.0))
        self._cone_end_dist     = float(params.get('cone_end_distance', {}).get('value', 45.0))
        self._cone_start_off    = float(params.get('cone_start_offset', {}).get('value', -7.0))
        self._cone_end_off      = float(params.get('cone_end_offset', {}).get('value', -3.3))

        super(LaneClosureWithTruck, self).__init__(
            "LaneClosureWithTruck", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

    def _spawn_actor_on_route(self, base_wp, forward_dist, lateral_offset, model, is_vehicle=False):
        """通用辅助函数：沿着弯道前方生成 Actor"""
        wps = base_wp.next(forward_dist)
        if not wps: return None
        
        target_wp = wps[0]
        transform = target_wp.transform
        # 基于当前航向的右侧向量进行横向平移，完美适配弯道
        location = transform.location + transform.get_right_vector() * lateral_offset
        location.z += 0.2
        
        print(location)
        print(transform.rotation)
        actor = CarlaDataProvider.request_new_actor(model, carla.Transform(location, transform.rotation))
        if actor:
            actor.set_simulate_physics(True)
            if is_vehicle:
                actor.set_light_state(carla.VehicleLightState.Special1)
            self.other_actors.append(actor)
        return actor

    def _initialize_actors(self, config):
        carla_map = CarlaDataProvider.get_map()
        # 直接使用 XML 中的 trigger_point 作为基准计算位置，脱离对自车当前位置的依赖
        base_location = config.trigger_points[0].location
        base_wp = carla_map.get_map().get_waypoint(base_location) if hasattr(carla_map, 'get_map') else carla_map.get_waypoint(base_location)

        # 1. 生成大货车
        self._spawn_actor_on_route(base_wp, self._truck_distance, self._truck_offset, "vehicle.carlamotors.carlacola", True)

        # 2. 生成斜向引导锥桶 (8个)
        num_cones = 8
        cone_model = "static.prop.constructioncone"
        for i in range(num_cones):
            fraction = i / float(num_cones - 1)
            forward_dist = self._cone_start_dist + (self._cone_end_dist - self._cone_start_dist) * fraction
            lateral_offset = self._cone_start_off + (self._cone_end_off - self._cone_start_off) * fraction
            self._spawn_actor_on_route(base_wp, forward_dist, lateral_offset, cone_model)

        # 3. 生成直排隔离锥桶 (12个)
        num_straight_cones = 12
        straight_end_dist = self._cone_end_dist + 35.0
        for i in range(1, num_straight_cones):
            fraction = i / float(num_straight_cones - 1)
            forward_dist = self._cone_end_dist + (straight_end_dist - self._cone_end_dist) * fraction
            self._spawn_actor_on_route(base_wp, forward_dist, self._cone_end_off, cone_model)

        # 4. 初始化自车速度
        target_speed_mps = self._initial_speed_kph / 3.6
        ego_transform = self.ego_vehicles[0].get_transform()
        direction = ego_transform.get_forward_vector()
        direction.z = 0.0 
        
        # 归一化后设置速度
        direction = direction / (direction.length() + 1e-6)
        self.ego_vehicles[0].set_target_velocity(
            carla.Vector3D(direction.x * target_speed_mps, direction.y * target_speed_mps, 0.0)
        )

    def _create_behavior(self):
        root = py_trees.composites.Parallel(
            "StaticBarrierBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        root.add_child(DriveDistance(self.ego_vehicles[0], 120))
        return root

    def _create_test_criteria(self):
        criteria = [
            CollisionTest(self.ego_vehicles[0], other_actor_type="miscellaneous", terminate_on_failure=False, name="CollisionTestStatic"),
            MinTTCAutoCriterion(actor=self.ego_vehicles[0],
                                other_actors=self.other_actors,
                                distance_threshold=40.0,
                                forward_angle_deg=140.0,
                                terminate_on_failure=False),
            DecelerationForConstructionTest(
                self.ego_vehicles[0],
                start_distance=self._cone_start_dist,
                end_distance=self._truck_distance,
                initial_speed_kmh=self._initial_speed_kph,
                target_speed_reduction=40.0
            ),
            RoutePassCompletionTest(self.ego_vehicles[0], pass_distance=self._truck_distance + 20.0)
        ]
        
        # 动态判定是否生成了卡车，有的话添加单独的碰撞检测
        if self.other_actors: 
            criteria.append(CollisionTest(
                self.ego_vehicles[0], other_actor=self.other_actors[0], terminate_on_failure=False, name="CollisionTestVehicle"
            ))
            
        return criteria

    def __del__(self):
        self.remove_all_actors()