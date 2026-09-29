import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from DriveTransformer.team_code.pid_controller import DecouplePIDController
try:
    from DriveTransformer.team_code.cvci_auxiliary_system import CVCIAuxiliarySystem
except ModuleNotFoundError:
    from team_code.cvci_auxiliary_system import CVCIAuxiliarySystem
from DriveTransformer.team_code.cvci_scenario_classifier import ScenarioClassifier
from DriveTransformer.team_code.cvci_scenario_context import ScenarioContext
from DriveTransformer.team_code.cvci_scenario_rules_v4 import ScenarioRuleRegistry
try:
    from DriveTransformer.team_code.auxiliary_perception import PointCloudGeometryFallback
except ModuleNotFoundError:
    from team_code.auxiliary_perception import PointCloudGeometryFallback
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,
                        wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from team_code.planner import RoutePlanner
from pyquaternion import Quaternion

SAVE_PATH = None #os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
DISABLE_BEV_SENSOR = os.environ.get('DISABLE_BEV_SENSOR', '0').lower() in ('1', 'true', 'yes')
CVCI_RECORD_MP4_DIR = os.environ.get('CVCI_RECORD_MP4_DIR', '')
CVCI_AUXILIARY_PERCEPTION_ENABLED = os.environ.get('CVCI_AUXILIARY_PERCEPTION_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
CVCI_LIDAR_ENABLED = os.environ.get('CVCI_LIDAR_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
CVCI_LEGACY_DETECTION_RULES_ENABLED = os.environ.get('CVCI_LEGACY_DETECTION_RULES_ENABLED', '0').lower() in ('1', 'true', 'yes', 'on')

DETECTION_CLASSES = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
]
MAP_CLASSES = ['Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign']



def _safe_filename(value):
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(value))[:180]


def get_entry_point():
    return 'DriveTransformerAgent'


class DriveTransformerAgent(autonomous_agent.AutonomousAgent):
    """
    Drive TransformerAgent
    """
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.controller = DecouplePIDController(speed_k_p=2.0, speed_k_i=0.8, speed_k_d=1.5, steer_k_p=1.5, steer_k_i=0.2, steer_k_d=0.2)
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            self.config_path = path_to_conf_file
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.device = os.environ.get("CVCI_AGENT_DEVICE", "cuda")
        if torch.cuda.is_available():
            if self.device.startswith("cuda:"):
                torch.cuda.set_device(int(self.device.split(":", 1)[1]))
            else:
                torch.cuda.set_device(0)
            print(f"CVCI agent CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} CVCI_AGENT_DEVICE={self.device} current_device={torch.cuda.current_device()} device_name={torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
        cfg = Config.fromfile(self.config_path)
        self.cameras = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        #remap path
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)  
        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        if cfg.get('lora_config', None):
            from adzoo.drivetransformer.mmdet3d_plugin.models.utils.lora import apply_lora
            lora_stats = apply_lora(self.model, cfg.lora_config)
            print(f"Applied LoRA in agent: {lora_stats['num_replaced']} Linear modules; trainable={lora_stats['trainable_params']}", flush=True)
        # load checkpoint
        if self.ckpt_path != "None":
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            ckpt = ckpt["state_dict"]
            new_state_dict = OrderedDict()
            for key, value in ckpt.items():
                new_key = key.replace("model.","").replace("._orig_mod", "")
                new_state_dict[new_key] = value
            print(self.model.load_state_dict(new_state_dict, strict = False))
        # MMCV's fp16 wrapper is appropriate for CUDA inference but leaves
        # Half tensors in convolution layers on CPU, where PyTorch's
        # slow_conv2d implementation does not support them.  Keep the CPU
        # fallback numerically valid; CUDA retains the original fp16 path.
        if not str(self.device).startswith("cpu"):
            wrap_fp16_model(self.model)
        self.model.to(self.device)
        self.model.eval()

        self.test_pipeline = []
        self.past_ego_pos_cache = []
        self.cache_lenth = 20
        # pipeline
        for test_pipeline in cfg.test_pipeline:
            if test_pipeline["type"] not in ['LoadMultiViewImageFromFiles','LoadAnnotations3D', "CustomObjectRangeFilter", "CustomObjectNameFilter", "TrajPreprocess"]:
                self.test_pipeline.append(test_pipeline)
            if test_pipeline["type"] == "CustomFormatBundle3D":
                test_pipeline["collect_keys"] = ['lidar2img', 'cam_intrinsic','timestamp', 'ego_pose', 'ego_pose_inv', 'pad_shape']
            if test_pipeline["type"] == "CustomCollect3D":
                test_pipeline["keys"] = ['img', 'ego_his_trajs', 'ego_lcf_feat', 'ego_fut_cmd', 'prev_exists', 'index', 'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'pad_shape']
        self.test_pipeline = Compose(self.test_pipeline)
        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0
        self.pid_metadata = {}
        self.prev_control_cache = []
        self.prev_control_list = []
        self.model_inference_interval = max(
            1, int(os.environ.get('CVCI_MODEL_INFERENCE_INTERVAL', '2') or 2)
        )
        self.cached_model_detection_result = None
        self.cached_model_raw_control = None
        self.step_time_avg = []
        self.rule_state = {
            "frame": 0,
            "low_speed_count": 0,
            "stuck_count": 0,
            "front_clear_count": 0,
            "front_blocked_count": 0,
            "red_light_count": 0,
            "front_vehicle_brake_count": 0,
            "front_obstacle_brake_count": 0,
            "last_rule_action": "none",
            "last_adjusted_control": None,
            "recent_speeds": [],
            "recent_controls": [],
            "last_detection_context": None,
            "last_v4_rule_action": "none",
        }
        self.scenario_classifier = ScenarioClassifier()
        self.scenario_rule_registry = ScenarioRuleRegistry()
        self.cvci_auxiliary_system = CVCIAuxiliarySystem()
        self.scenario_id_log_path = os.environ.get('CVCI_SCENARIO_ID_LOG_PATH', '').strip()
        self.scenario_id_expected_macro = os.environ.get('EXPECTED_MACRO', '').strip()
        self.scenario_id_route_id = os.environ.get('CVCI_ROUTE_ID', '').strip()
        self.scenario_id_last_macro = ''
        self.scenario_id_stable_frames = 0
        self.scenario_id_candidate_macro = 'unknown'
        self.scenario_id_candidate_frames = 0
        self.scenario_id_confirmed_macro = 'unknown'
        self.auxiliary_perception_enabled = CVCI_AUXILIARY_PERCEPTION_ENABLED
        self.lidar_enabled = (CVCI_LIDAR_ENABLED and self.auxiliary_perception_enabled) or self.cvci_auxiliary_system.wants_lidar
        self.lidar_geometry = PointCloudGeometryFallback() if self.lidar_enabled else None
        self.record_mp4_path = None
        self.record_mp4_writer = None
        self.record_mp4_every = int(os.environ.get('CVCI_RECORD_MP4_EVERY', '2'))
        if CVCI_RECORD_MP4_DIR:
            pathlib.Path(CVCI_RECORD_MP4_DIR).mkdir(parents=True, exist_ok=True)
            self.record_mp4_path = pathlib.Path(CVCI_RECORD_MP4_DIR) / (_safe_filename(self.save_name) + '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.record_mp4_writer = cv2.VideoWriter(str(self.record_mp4_path), fourcc, 10.0, (640, 360))
            if not self.record_mp4_writer.isOpened():
                print(f'Failed to open MP4 recorder: {self.record_mp4_path}', flush=True)
                self.record_mp4_writer = None
            else:
                print(f'CVCI MP4 recording to {self.record_mp4_path}', flush=True)
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            print("SAVE Result to ", string)
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb_front').mkdir()
            # (self.save_path / 'rgb_front_right').mkdir()
            # (self.save_path / 'rgb_front_left').mkdir()
            # (self.save_path / 'rgb_back').mkdir()
            # (self.save_path / 'rgb_back_right').mkdir()
            # (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()

        # transform from lidar to image coordinates
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                              [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                              [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                    [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                            [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                            [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        # transform from lidar to camera coordinates
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                              [ 0.  ,  0.  , -1.  , -0.24],
                              [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        
        # camera intrinsics
        self.cam_intrinsics = {
        'CAM_FRONT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                              [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                   [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[560.16603057,   0.        , 800.        ,   0.        ],
                             [  0.        , 560.16603057, 450.        ,   0.        ],
                             [  0.        ,   0.        ,   1.        ,   0.        ],
                             [  0.        ,   0.        ,   0.        ,   1.        ]]),
     
        'CAM_BACK_LEFT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
  
        'CAM_BACK_RIGHT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        }
        
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        topdown_extrinsics =  np.array([[1.0, 0.0, 0.0, 0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 50.0], [0.0, 0.0, 0.0, 1.0]])
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ topdown_extrinsics
        
        self.all_sensors =  {
                # camera rgb
                'CAM_FRONT':{
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                'CAM_FRONT_LEFT':{
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                'CAM_FRONT_RIGHT':{
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                'CAM_BACK':{
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                'CAM_BACK_LEFT':{
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                'CAM_BACK_RIGHT':{
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                'IMU':{
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                'GPS':{
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                'SPEED':{
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                'bev': {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    },
                'LIDAR': {
                        'type': 'sensor.lidar.ray_cast',
                        'x': 0.0, 'y': 0.0, 'z': 2.4,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'channels': 64,
                        'range': 85.0,
                        'points_per_second': 120000,
                        'rotation_frequency': 10.0,
                        'upper_fov': 10.0,
                        'lower_fov': -25.0,
                        'id': 'LIDAR'
                    }
                
        }
   

    def _init(self):
        # get gps reference point
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        # route planner
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._plan_gps_HACK, True)
        self._command_planner = RoutePlanner(7.5, 25.0, 257, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._command_planner.set_route(self._global_plan, True)
        self.initialized = True
  
    def sensors(self):
        sensors = []
        select_sensor_names = self.cameras + ['IMU','GPS','SPEED']
        if IS_BENCH2DRIVE and not DISABLE_BEV_SENSOR:
            select_sensor_names.append('bev')
        if self.lidar_enabled:
            select_sensor_names.append('LIDAR')
        for key in select_sensor_names:
            sensors.append(self.all_sensors[key])
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in self.cameras:
            raw_img = input_data[cam][1][:, :, :3]
            if cam == 'CAM_FRONT' and self.record_mp4_writer is not None and self.step % max(self.record_mp4_every, 1) == 0:
                frame = cv2.resize(raw_img, (640, 360), interpolation=cv2.INTER_AREA)
                self.record_mp4_writer.write(frame)
            img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img

        if 'bev' in input_data:
            bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            bev = np.zeros((512, 512, 3), dtype=np.uint8)
        lidar_points = None
        if self.lidar_enabled and 'LIDAR' in input_data:
            lidar_points = input_data['LIDAR'][1]
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
        far_node, far_command = self._command_planner.run_step(pos)

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node,
                'command_far':far_command,
                'command_far_xy':far_node,
                'lidar_points': lidar_points,
                }
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)

        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['cam_intrinsic'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = np.array(self.step / 20)
        results['box_type_3d'], _ = get_box_type('LiDAR')
        results['index'] = self.step
        results['prev_exists'] = (self.step > 1) 
        for cam in self.cameras: 
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['cam_intrinsic'].append(self.cam_intrinsics[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
  
        raw_theta = tick_data['compass'] if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        # can bus
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        results['aug_config'] = {'resize': 0.66, 'resize_dims': (1056, 594), 'crop': (0, 210, 1056, 594), 'flip': False, 'rotate': 0, 'rotate_3d': 0}
        # ego_lcf_feat
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0] = tick_data['speed']
        ego_lcf_feat[2:4] = can_bus[10:12].copy()
        ego_lcf_feat[4] = can_bus[15]
        ego_lcf_feat[5] = 4.89238167
        ego_lcf_feat[6] = 1.83671331
        ego_lcf_feat[7] = tick_data['speed']
        ego_lcf_feat[8] = 0 if len(self.prev_control_cache) < 2 else self.prev_control_cache[0].steer
        results['ego_lcf_feat'] = ego_lcf_feat
        # command
        command = np.zeros(140)
        command[0:6] = self.command2hot(tick_data['command_far'])
        command[70:76] = self.command2hot(tick_data['command_near'])
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        command_far_xy = np.array([tick_data['command_far_xy'][0]-can_bus[0],-tick_data['command_far_xy'][1]-can_bus[1]])  
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_near_xy = rotation_matrix @ command_near_xy
        local_command_far_xy = rotation_matrix @ command_far_xy
        command[6:70] = self.pos2posemb(local_command_far_xy)
        command[76:140] = self.pos2posemb(local_command_near_xy)
        results['ego_fut_cmd'] = command
        # ego position
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        current_pose = lidar2global
        current_pose_inv = self.invert_pose(current_pose)
        results['ego_pose'] = current_pose
        results['ego_pose_inv'] = current_pose_inv   
        # ego past trajectory
        past_pose_1 = self.past_ego_pos_cache[-10] if len(self.past_ego_pos_cache) >= 10 else lidar2global
        past_pose_2 = self.past_ego_pos_cache[0] if len(self.past_ego_pos_cache) == 20 else lidar2global   
        past2current_1 = current_pose_inv @ past_pose_1
        past2current_2 = current_pose_inv @ past_pose_2
        past2current_1_xy = past2current_1[0:2,3]
        past2current_2_xy = past2current_2[0:2,3]
        ego_his_trajs = np.zeros((2,2))
        ego_his_trajs[0] = past2current_1_xy - past2current_2_xy
        ego_his_trajs[1] = -past2current_1_xy
        results['ego_his_trajs'] = ego_his_trajs
        if len(self.past_ego_pos_cache)==20:
            self.past_ego_pos_cache.pop(0)
        self.past_ego_pos_cache.append(current_pose)
        if (
            self.step % self.model_inference_interval != 0
            and self.cached_model_detection_result is not None
            and self.cached_model_raw_control is not None
        ):
            # Keep the auxiliary controller at its historically validated
            # 10 Hz cadence.  Its state-machine thresholds are measured in
            # controller updates, so running it on the formerly skipped odd
            # frames would halve all physical waiting/recovery durations.
            if self.step % 2 == 1:
                return self.prev_control
            return self._run_cached_model_step(tick_data, timestamp)
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        # data pipeline
        results = self.test_pipeline(results)      
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if isinstance(data,torch.Tensor):
                    input_data_batch[key] = data.to(self.device)
                    if input_data_batch[key].dtype==torch.float64:
                        input_data_batch[key] = input_data_batch[key].to(torch.float32)
                elif isinstance(data,list):
                    if torch.is_tensor(data[0]):
                        input_data_batch[key][0] = input_data_batch[key][0].to(self.device)
                        if input_data_batch[key][0].dtype==torch.float64:
                            input_data_batch[key][0] = input_data_batch[key][0].to(torch.float32)
        step_start_time = time.time()
        # model inference
        output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True)
        detection_result = self.build_current_frame_detection_result(output_data_batch[0], tick_data, timestamp)
        self.handle_current_frame_detection_result(detection_result)
        self.cached_model_detection_result = detection_result
        self.step_time_avg.append(float(time.time()-step_start_time))
        if len(self.step_time_avg)==20:
            # print("Model Avg Step Time:", np.mean(self.step_time_avg))
            self.step_time_avg.pop(0)
        all_out_truck = None
        ego_traj_cls_scores = None
        selected_mode = 0 
        angles = output_data_batch[0]['ego_fut_preds_fix_dist'][0,selected_mode,:,0].float().cpu().numpy()
        # for trajectories with fixed distance, the output is the angle with y-axis in lidar coordinate system. 
        # get the x, y coordinate with the disance and angle.
        ego_traj_fix_dist = np.arange(1,21,dtype=np.float64).reshape(-1,1).repeat(2,1) 
        ego_traj_fix_dist[:,0] *= np.cos(angles)
        ego_traj_fix_dist[:,1] *= np.sin(angles)
        ego_traj_fix_time = output_data_batch[0]['ego_fut_preds_fix_time'][0,selected_mode,:,[1,0]].float().cpu().numpy() 
        if self.step <= 20: # waiting for scenerio initialization (cars are more likely to disappear suddenly in this period)
            steer, throttle, brake = 0.0, 0.0, 1.0
        else:
            steer, throttle, brake = self.controller.step(ego_traj_fix_time, ego_traj_fix_dist, tick_data['speed']) # controller
        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        raw_control = control
        self.cached_model_raw_control = carla.VehicleControl(
            steer=float(raw_control.steer),
            throttle=float(raw_control.throttle),
            brake=float(raw_control.brake),
            reverse=bool(getattr(raw_control, 'reverse', False)),
        )
        if CVCI_LEGACY_DETECTION_RULES_ENABLED:
            control = self._apply_rule_based_control(
                raw_control=raw_control,
                detection_result=detection_result,
                tick_data=tick_data,
                ego_traj_fix_time=ego_traj_fix_time,
                ego_traj_fix_dist=ego_traj_fix_dist,
                timestamp=timestamp,
            )
        else:
            self.rule_state['last_rule_action'] = 'none'
            self.rule_state['last_v4_rule_action'] = 'disabled'
        control = self.cvci_auxiliary_system.process(
            raw_control=control,
            model_detection=detection_result,
            tick_data=tick_data,
            timestamp=float(timestamp),
            legacy_rule_action=self.rule_state.get('last_rule_action', 'none'),
        )
        self.pid_metadata['auxiliary_system'] = getattr(self.cvci_auxiliary_system, 'last_debug', {})
        self._write_scenario_identification_log(float(timestamp))
        self.pid_metadata['raw_steer'] = raw_control.steer
        self.pid_metadata['raw_throttle'] = raw_control.throttle
        self.pid_metadata['raw_brake'] = raw_control.brake
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['speed'] = float(tick_data['speed'])
        self.pid_metadata['rule_action'] = self.rule_state.get('last_rule_action', 'none')
        aux_log_period = int(os.environ.get('CVCI_AUX_LOG_PERIOD', '0') or 0)
        if aux_log_period > 0 and self.step % aux_log_period == 0:
            print(f"CVCI_AUX frame={self.step} {self.pid_metadata.get('auxiliary_system', {})}", flush=True)
        if os.environ.get('CVCI_FIRST_FRAME_RECOGNITION_PROBE', '0').lower() in ('1', 'true', 'yes', 'on'):
            raise RuntimeError('CVCI_FIRST_FRAME_RECOGNITION_PROBE_COMPLETE')
        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data, ego_traj_fix_time.copy(), ego_traj_fix_dist.copy(), draw_traj=True)
        self.prev_control = control
        if len(self.prev_control_cache)==2:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        
        return control

    def _run_cached_model_step(self, tick_data, timestamp):
        """Reuse model output while keeping CVCI arbitration at the validated 10 Hz."""
        cached = self.cached_model_raw_control
        raw_control = carla.VehicleControl(
            steer=float(cached.steer),
            throttle=float(cached.throttle),
            brake=float(cached.brake),
            reverse=bool(getattr(cached, 'reverse', False)),
        )
        self.rule_state['last_rule_action'] = 'none'
        self.rule_state['last_v4_rule_action'] = 'disabled'
        control = self.cvci_auxiliary_system.process(
            raw_control=raw_control,
            model_detection=self.cached_model_detection_result,
            tick_data=tick_data,
            timestamp=float(timestamp),
            legacy_rule_action='none',
        )
        self.pid_metadata['auxiliary_system'] = getattr(self.cvci_auxiliary_system, 'last_debug', {})
        self._write_scenario_identification_log(float(timestamp))
        self.pid_metadata['raw_steer'] = raw_control.steer
        self.pid_metadata['raw_throttle'] = raw_control.throttle
        self.pid_metadata['raw_brake'] = raw_control.brake
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['speed'] = float(tick_data['speed'])
        self.pid_metadata['rule_action'] = 'none'
        aux_log_period = int(os.environ.get('CVCI_AUX_LOG_PERIOD', '0') or 0)
        if aux_log_period > 0 and self.step % aux_log_period == 0:
            print(f"CVCI_AUX frame={self.step} {self.pid_metadata.get('auxiliary_system', {})}", flush=True)
        self.prev_control = control
        if len(self.prev_control_cache) == 2:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control

    def _write_scenario_identification_log(self, timestamp):
        if not self.scenario_id_log_path:
            return
        debug = self.pid_metadata.get('auxiliary_system', {}) or {}
        cls_info = self.scenario_classifier.classify_auxiliary_debug(debug)
        detected = str(cls_info.get('macro_scenario', '') or 'unknown')
        confidence = float(cls_info.get('confidence', 0.0) or 0.0)
        frozen_online_macro = str(debug.get('frozen_route_macro', '') or '')
        if frozen_online_macro and frozen_online_macro != 'unknown':
            # The formal full-144 policy freezes the first legal online frame.
            # Log that runtime owner immediately; the historical 60-frame
            # audit debounce would otherwise conceal the actual first-frame
            # decision that owns rule and controller selection.
            detected = frozen_online_macro
            confidence = max(confidence, 0.80)
            self.scenario_id_confirmed_macro = detected
            self.scenario_id_candidate_macro = detected
            self.scenario_id_candidate_frames = 1
        elif self.scenario_id_confirmed_macro != 'unknown':
            detected = self.scenario_id_confirmed_macro
        elif detected != 'unknown' and confidence >= 0.75:
            if detected == self.scenario_id_candidate_macro:
                self.scenario_id_candidate_frames += 1
            else:
                self.scenario_id_candidate_macro = detected
                self.scenario_id_candidate_frames = 1
            if self.scenario_id_candidate_frames >= 60:
                self.scenario_id_confirmed_macro = detected
            else:
                detected = 'unknown'
                confidence = 0.0
        else:
            self.scenario_id_candidate_macro = 'unknown'
            self.scenario_id_candidate_frames = 0
            detected = 'unknown'
            confidence = 0.0
        if detected == self.scenario_id_last_macro:
            self.scenario_id_stable_frames += 1
        else:
            self.scenario_id_last_macro = detected
            self.scenario_id_stable_frames = 1
        phase = str(cls_info.get('phase', '') or '')
        evidence = str(cls_info.get('evidence', '') or '')
        top3 = str(cls_info.get('top3_macro_candidates', detected) or detected)
        expected = self.scenario_id_expected_macro
        is_match = bool(expected and detected == expected)
        path = pathlib.Path(self.scenario_id_log_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists() or path.stat().st_size == 0
            with path.open('a', encoding='utf-8') as f:
                if new_file:
                    f.write('timestamp,route_id,expected_macro,detected_macro,confidence,top3_macro_candidates,phase,evidence,frames_stable,is_match\n')
                safe_evidence = evidence.replace(',', ';').replace('\n', ' ')
                f.write(
                    f'{timestamp:.3f},{self.scenario_id_route_id or (self.scenario_classifier.route_ids[0] if self.scenario_classifier.route_ids else "")},'
                    f'{expected},{detected},{confidence:.3f},{top3},{phase},{safe_evidence},'
                    f'{self.scenario_id_stable_frames},{int(is_match)}\n'
                )
        except Exception as exc:
            if int(os.environ.get('CVCI_AUX_LOG_PERIOD', '0') or 0) > 0 and self.step % 40 == 0:
                print(f'CVCI_SCENARIO_ID_LOG_ERROR {repr(exc)}', flush=True)
    

    def build_current_frame_detection_result(self, model_output, tick_data, timestamp):
        """Format current-frame detections for lightweight rule logic."""
        objects = []
        boxes_3d = model_output.get('boxes_3d', None)
        if boxes_3d is not None and all(key in model_output for key in ['scores_3d', 'labels_3d']):
            box_tensor = boxes_3d.tensor.detach().cpu().numpy()
            scores = model_output['scores_3d'].detach().cpu().numpy()
            labels = model_output['labels_3d'].detach().cpu().numpy()
            trajs = model_output.get('trajs_3d', None)
            trajs = trajs.detach().cpu().numpy() if trajs is not None else None

            for obj_id, box in enumerate(box_tensor):
                label = int(labels[obj_id])
                obj = {
                    'id': obj_id,
                    'label': label,
                    'class_name': DETECTION_CLASSES[label] if label < len(DETECTION_CLASSES) else str(label),
                    'score': float(scores[obj_id]),
                    'box_lidar': {
                        'x': float(box[0]),
                        'y': float(box[1]),
                        'z': float(box[2]),
                        'dx': float(box[3]),
                        'dy': float(box[4]),
                        'dz': float(box[5]),
                        'yaw': float(box[6]),
                    },
                }
                if box.shape[0] > 8:
                    obj['box_lidar']['vx'] = float(box[7])
                    obj['box_lidar']['vy'] = float(box[8])
                if trajs is not None:
                    obj['traj'] = trajs[obj_id].reshape(-1, 2).astype(float).tolist()
                objects.append(obj)

        map_objects = []
        if all(key in model_output for key in ['map_boxes_3d', 'map_scores_3d', 'map_labels_3d']):
            map_boxes = model_output['map_boxes_3d'].detach().cpu().numpy()
            map_scores = model_output['map_scores_3d'].detach().cpu().numpy()
            map_labels = model_output['map_labels_3d'].detach().cpu().numpy()
            map_pts = model_output.get('map_pts_3d', None)
            map_pts = map_pts.detach().cpu().numpy() if map_pts is not None else None

            for map_id, box in enumerate(map_boxes):
                label = int(map_labels[map_id])
                map_obj = {
                    'id': map_id,
                    'label': label,
                    'class_name': MAP_CLASSES[label] if label < len(MAP_CLASSES) else str(label),
                    'score': float(map_scores[map_id]),
                    'box': box.astype(float).tolist(),
                }
                if map_pts is not None:
                    map_obj['pts'] = map_pts[map_id].astype(float).tolist()
                map_objects.append(map_obj)

        lidar_geometry = None
        if self.lidar_geometry is not None:
            try:
                lidar_geometry = self.lidar_geometry.update(tick_data.get('lidar_points'), timestamp=float(timestamp))
            except Exception as exc:
                lidar_geometry = {'available': False, 'stale': True, 'error': repr(exc)}

        return {
            'frame': int(self.step),
            'timestamp': float(timestamp),
            'objects': objects,
            'map_objects': map_objects,
            'lidar_geometry': lidar_geometry,
            'ego': {
                'speed': float(tick_data['speed']),
                'gps': np.asarray(tick_data['gps']).astype(float).tolist(),
                'pos': np.asarray(tick_data['pos']).astype(float).tolist(),
                'compass': float(tick_data['compass']),
                'command_near': int(tick_data['command_near']),
                'command_far': int(tick_data['command_far']),
            },
        }

    def handle_current_frame_detection_result(self, detection_result):
        self.rule_state['frame'] = int(detection_result.get('frame', self.step))
        self.rule_state['last_detection_context'] = self._analyze_detection_context(detection_result)
        return None

    def _is_object_in_front_corridor(self, obj, x_min, x_max, y_abs_max):
        box = obj.get('box_lidar') or {}
        x = float(box.get('x', 999.0))
        y = float(box.get('y', 999.0))
        return x_min <= x <= x_max and abs(y) <= y_abs_max

    def _map_object_front_distance(self, map_obj):
        candidates = []
        for key in ('box', 'pts'):
            value = map_obj.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                candidates.append((float(arr[0]), float(arr[1])))
        if not candidates:
            return None
        front = [(x, y) for x, y in candidates if x >= -2.0 and abs(y) <= 5.0]
        if not front:
            return None
        return min((float(np.hypot(x, y)) for x, y in front), default=None)

    def _analyze_detection_context(self, detection_result):
        context = {
            'front_clear': True,
            'recovery_clear': True,
            'immediate_hazard': False,
            'side_risk': False,
            'front_vehicle_distance': None,
            'front_pedestrian_distance': None,
            'front_obstacle_distance': None,
            'nearest_front_object': None,
            'left_clear': True,
            'right_clear': True,
            'has_red_light_or_stop': False,
            'red_stop_distance': None,
            'risk_level': 0,
        }
        nearest_dist = None
        for obj in detection_result.get('objects', []):
            if float(obj.get('score', 0.0)) < 0.32:
                continue
            cls = str(obj.get('class_name', '')).lower()
            box = obj.get('box_lidar') or {}
            x = float(box.get('x', 999.0))
            y = float(box.get('y', 999.0))
            dist = max(x, 0.0)
            if 0.0 <= x <= 18.0 and abs(y) <= 4.2 and float(obj.get('score', 0.0)) >= 0.38:
                if y < -2.2:
                    context['left_clear'] = False
                if y > 2.2:
                    context['right_clear'] = False
            if 0.0 <= x <= 8.0 and 1.6 < abs(y) <= 3.4 and float(obj.get('score', 0.0)) >= 0.42:
                context['side_risk'] = True
            if cls in ('car', 'van', 'truck', 'bicycle', 'pedestrian', 'traffic_cone', 'others'):
                y_limit = 2.6 if cls in ('pedestrian', 'bicycle') else 2.2
                x_limit = 20.0 if cls in ('pedestrian', 'bicycle') else 15.0
                if self._is_object_in_front_corridor(obj, 0.0, x_limit, y_limit):
                    context['front_clear'] = False
                    if nearest_dist is None or dist < nearest_dist:
                        nearest_dist = dist
                        context['nearest_front_object'] = obj
                    if cls in ('car', 'van', 'truck'):
                        context['front_vehicle_distance'] = dist if context['front_vehicle_distance'] is None else min(context['front_vehicle_distance'], dist)
                    elif cls in ('pedestrian', 'bicycle'):
                        context['front_pedestrian_distance'] = dist if context['front_pedestrian_distance'] is None else min(context['front_pedestrian_distance'], dist)
                    else:
                        context['front_obstacle_distance'] = dist if context['front_obstacle_distance'] is None else min(context['front_obstacle_distance'], dist)
            if cls == 'traffic_light' and self._is_object_in_front_corridor(obj, 0.0, 18.0, 4.0) and float(obj.get('score', 0.0)) >= 0.45:
                context['has_red_light_or_stop'] = True
                context['red_stop_distance'] = dist if context['red_stop_distance'] is None else min(context['red_stop_distance'], dist)

        for map_obj in detection_result.get('map_objects', []):
            if float(map_obj.get('score', 0.0)) < 0.35:
                continue
            cls = str(map_obj.get('class_name', '')).lower()
            if cls not in ('trafficlight', 'stopsign'):
                continue
            dist = self._map_object_front_distance(map_obj)
            if dist is not None and dist <= 16.0:
                context['has_red_light_or_stop'] = True
                context['red_stop_distance'] = dist if context['red_stop_distance'] is None else min(context['red_stop_distance'], dist)

        lidar_geometry = detection_result.get('lidar_geometry') or {}
        if lidar_geometry.get('available') and not lidar_geometry.get('stale', False):
            lidar_dist = lidar_geometry.get('front_distance')
            if lidar_dist is not None and (lidar_geometry.get('front_blocked') or lidar_geometry.get('near_blocked')):
                context['front_clear'] = False
                context['front_obstacle_distance'] = lidar_dist if context['front_obstacle_distance'] is None else min(context['front_obstacle_distance'], float(lidar_dist))
                context['risk_level'] = max(context['risk_level'], 2)

        if context['front_pedestrian_distance'] is not None and context['front_pedestrian_distance'] < 9.0:
            context['risk_level'] = max(context['risk_level'], 3)
        if context['front_vehicle_distance'] is not None and context['front_vehicle_distance'] < 6.0:
            context['risk_level'] = max(context['risk_level'], 3)
        if context['front_obstacle_distance'] is not None and context['front_obstacle_distance'] < 5.0:
            context['risk_level'] = max(context['risk_level'], 3)
        close_ped = context['front_pedestrian_distance'] is not None and context['front_pedestrian_distance'] < 6.0
        close_vehicle = context['front_vehicle_distance'] is not None and context['front_vehicle_distance'] < 4.0
        close_obstacle = context['front_obstacle_distance'] is not None and context['front_obstacle_distance'] < 4.4
        close_red_stop = context['has_red_light_or_stop'] and context['red_stop_distance'] is not None and context['red_stop_distance'] < 6.5
        context['immediate_hazard'] = bool(close_ped or close_vehicle or close_obstacle or close_red_stop)
        context['recovery_clear'] = not context['immediate_hazard']
        if not context['front_clear']:
            context['risk_level'] = max(context['risk_level'], 1)
        if context['has_red_light_or_stop'] and (context['red_stop_distance'] is None or context['red_stop_distance'] < 9.0):
            context['risk_level'] = max(context['risk_level'], 2)
        return context

    def _update_rule_state(self, context, tick_data, raw_control):
        state = self.rule_state
        speed = float(tick_data.get('speed', 0.0))
        state['recent_speeds'].append(speed)
        state['recent_speeds'] = state['recent_speeds'][-30:]
        state['recent_controls'].append((float(raw_control.steer), float(raw_control.throttle), float(raw_control.brake)))
        state['recent_controls'] = state['recent_controls'][-30:]
        if speed < 0.3:
            state['low_speed_count'] += 1
        else:
            state['low_speed_count'] = 0
        if context.get('recovery_clear', context['front_clear']):
            state['front_clear_count'] += 1
            state['front_blocked_count'] = 0
        else:
            state['front_blocked_count'] += 1
            state['front_clear_count'] = 0
            state['stuck_count'] = 0
        if context['has_red_light_or_stop']:
            state['red_light_count'] += 1
        else:
            state['red_light_count'] = 0

    def _trajectory_has_anomaly(self, ego_traj_fix_time, ego_traj_fix_dist):
        for traj in (ego_traj_fix_time, ego_traj_fix_dist):
            arr = np.asarray(traj, dtype=np.float32)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return True
            if arr.ndim >= 2 and arr.shape[0] >= 4:
                lateral = arr[:8, 1]
                forward = arr[:8, 0]
                if np.nanmax(np.abs(lateral)) > 7.0:
                    return True
                if np.nanmin(forward) < -2.0:
                    return True
                if np.nanmax(np.abs(np.diff(lateral))) > 4.5:
                    return True
        return False

    def _trajectory_recovery_ok(self, ego_traj_fix_time, ego_traj_fix_dist, raw_steer):
        if abs(float(raw_steer)) > 0.45:
            return False
        for traj in (ego_traj_fix_time, ego_traj_fix_dist):
            arr = np.asarray(traj, dtype=np.float32)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return False
            if arr.ndim >= 2 and arr.shape[0] >= 6:
                lateral = arr[:8, 1]
                forward = arr[:8, 0]
                if np.nanmax(np.abs(lateral)) > 4.2:
                    return False
                if np.nanmax(np.abs(np.diff(lateral))) > 2.8:
                    return False
                if np.nanmin(forward) < -0.8:
                    return False
        return True

    def _make_control(self, steer, throttle, brake):
        return carla.VehicleControl(
            steer=float(np.clip(steer, -1.0, 1.0)),
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
        )

    def _apply_rule_based_control(self, raw_control, detection_result, tick_data, ego_traj_fix_time, ego_traj_fix_dist, timestamp):
        context = self.rule_state.get('last_detection_context') or self._analyze_detection_context(detection_result)
        self._update_rule_state(context, tick_data, raw_control)
        state = self.rule_state
        speed = float(tick_data.get('speed', 0.0))
        steer = float(raw_control.steer)
        throttle = float(raw_control.throttle)
        brake = float(raw_control.brake)
        action = 'none'
        emergency = False

        ped_dist = context.get('front_pedestrian_distance')
        veh_dist = context.get('front_vehicle_distance')
        obs_dist = context.get('front_obstacle_distance')
        red_dist = context.get('red_stop_distance')
        immediate_hazard = bool(context.get('immediate_hazard'))

        if context.get('has_red_light_or_stop') and state.get('red_light_count', 0) >= 3 and red_dist is not None and red_dist < 6.5:
            throttle = 0.0
            brake = max(brake, 0.70 if speed > 1.0 else 0.40)
            action = 'red_stop_hold'
        if ped_dist is not None and ped_dist < 6.0:
            throttle = 0.0
            brake = 1.0 if ped_dist < 4.0 else max(brake, 0.75)
            action = 'front_pedestrian_brake'
            emergency = ped_dist < 4.0
        elif veh_dist is not None and veh_dist < 4.0:
            throttle = 0.0
            brake = 1.0 if veh_dist < 2.6 else max(brake, 0.60)
            action = 'front_vehicle_brake'
            emergency = veh_dist < 2.6
        elif obs_dist is not None and obs_dist < 4.8:
            throttle = min(throttle, 0.05)
            brake = max(brake, 0.65 if obs_dist < 3.4 else 0.45)
            action = 'front_obstacle_brake'

        front_vehicle_conflict = (
            veh_dist is not None
            and float(veh_dist) < 4.5
            and ped_dist is None
            and obs_dist is None
            and not (red_dist is not None and red_dist < 8.0)
        )
        if front_vehicle_conflict:
            state['front_vehicle_brake_count'] = min(state.get('front_vehicle_brake_count', 0) + 1, 400)
        else:
            state['front_vehicle_brake_count'] = 0
        if action == 'front_obstacle_brake':
            state['front_obstacle_brake_count'] = min(state.get('front_obstacle_brake_count', 0) + 1, 400)
        else:
            state['front_obstacle_brake_count'] = 0

        front_vehicle_wait = int(state.get('front_vehicle_brake_count', 0))
        bounded_front_vehicle_release = (
            action == 'front_vehicle_brake'
            and speed < 0.8
            and front_vehicle_wait >= 70
            and veh_dist is not None
            and float(veh_dist) >= 2.20
            and (not emergency or front_vehicle_wait >= 160)
            and ped_dist is None
            and obs_dist is None
            and not (red_dist is not None and red_dist < 8.0)
            and not context.get('side_risk', False)
        )
        if bounded_front_vehicle_release:
            # Avoid turning a conservative front-vehicle yield into a permanent stop.
            # The release is distance-gated and only creeps forward after a sustained
            # low-speed wait; very close vehicles and pedestrian/red-light hazards keep
            # their hard brake behavior until the bounded deadlock escape below.
            creep = 0.16
            steer_limit = 0.55
            if front_vehicle_wait >= 140:
                creep = 0.24
            if front_vehicle_wait >= 180:
                creep = 0.45
                steer_limit = 0.75
            if front_vehicle_wait >= 260:
                creep = 0.55
                steer_limit = 0.85
            throttle = max(throttle, creep)
            brake = 0.0
            steer = float(np.clip(steer, -steer_limit, steer_limit))
            action = 'front_vehicle_deadlock_escape' if front_vehicle_wait >= 180 else 'front_vehicle_wait_release'

        front_vehicle_deadlock_escape = (
            action == 'front_vehicle_brake'
            and front_vehicle_conflict
            and speed < 0.5
            and front_vehicle_wait >= 180
            and ped_dist is None
            and obs_dist is None
            and not (red_dist is not None and red_dist < 8.0)
            and not context.get('side_risk', False)
        )
        if front_vehicle_deadlock_escape:
            throttle = max(throttle, 0.45 if front_vehicle_wait < 260 else 0.55)
            brake = 0.0
            steer = float(np.clip(steer, -0.85, 0.85))
            action = 'front_vehicle_deadlock_escape'

        front_obstacle_wait = int(state.get('front_obstacle_brake_count', 0))
        bounded_front_obstacle_release = (
            action == 'front_obstacle_brake'
            and speed < 0.6
            and front_obstacle_wait >= 80
            and obs_dist is not None
            and float(obs_dist) >= 3.20
            and ped_dist is None
            and veh_dist is None
            and not (red_dist is not None and red_dist < 8.0)
            and not context.get('side_risk', False)
        )
        if bounded_front_obstacle_release:
            creep = 0.14
            if front_obstacle_wait >= 140:
                creep = 0.22
            if front_obstacle_wait >= 220:
                creep = 0.30
            throttle = max(throttle, creep)
            brake = 0.0
            steer = float(np.clip(steer, -0.22, 0.22))
            action = 'front_obstacle_wait_release'

        traj_anomaly = self._trajectory_has_anomaly(ego_traj_fix_time, ego_traj_fix_dist) or abs(steer) > 0.95
        if traj_anomaly and action == 'none':
            steer = float(np.clip(steer, -0.55, 0.55))
            throttle = min(throttle, 0.22)
            brake = max(brake, 0.15 if speed > 1.5 else brake)
            action = 'trajectory_guard'

        raw_is_holding = float(raw_control.brake) > 0.18 or float(raw_control.throttle) < 0.18
        traj_recovery_ok = self._trajectory_recovery_ok(ego_traj_fix_time, ego_traj_fix_dist, raw_control.steer)
        steer_side_clear = (
            abs(float(raw_control.steer)) < 0.28
            or (float(raw_control.steer) < 0.0 and context.get('left_clear', True))
            or (float(raw_control.steer) > 0.0 and context.get('right_clear', True))
        )
        can_recover = (
            context.get('recovery_clear', context.get('front_clear'))
            and speed < 0.55
            and state.get('low_speed_count', 0) >= 5
            and state.get('front_clear_count', 0) >= 3
            and raw_is_holding
            and traj_recovery_ok
            and steer_side_clear
            and not context.get('side_risk', False)
        )
        if can_recover and action not in ('red_stop_hold', 'front_pedestrian_brake', 'front_vehicle_brake', 'front_obstacle_brake'):
            state['stuck_count'] = min(state.get('stuck_count', 0) + 1, 40)
            if state['stuck_count'] <= 12:
                target_throttle = 0.24 + 0.012 * state['stuck_count']
            else:
                target_throttle = 0.40 if abs(float(raw_control.steer)) < 0.22 else 0.32
            throttle = max(throttle, min(target_throttle, 0.42))
            brake = 0.0
            steer = float(np.clip(steer, -0.32, 0.32))
            action = 'clear_stuck_recovery'
        elif (
            context.get('recovery_clear', False)
            and speed < 1.2
            and not immediate_hazard
            and action == 'none'
            and float(raw_control.throttle) < 0.22
            and traj_recovery_ok
            and steer_side_clear
            and not context.get('side_risk', False)
        ):
            throttle = max(throttle, 0.18)
            brake = min(brake, 0.05)
            action = 'clear_crawl_release'

        last = state.get('last_adjusted_control')
        if last is not None and not emergency:
            max_throttle_rise = 0.12
            if action == 'clear_stuck_recovery':
                max_throttle_rise = 0.18
            elif action == 'clear_crawl_release':
                max_throttle_rise = 0.12
            throttle = min(throttle, float(last.throttle) + max_throttle_rise)
            if brake < float(last.brake) and action in ('none', 'trajectory_guard', 'clear_crawl_release'):
                brake = max(brake, float(last.brake) - 0.18)
            if action == 'clear_stuck_recovery':
                brake = min(brake, 0.02)
            steer = 0.72 * steer + 0.28 * float(last.steer)

        adjusted = self._make_control(steer, throttle, brake)
        adjusted, v4_action = self._apply_scenario_specific_override(
            adjusted, context, tick_data, detection_result
        )
        state['last_adjusted_control'] = adjusted
        state['last_rule_action'] = action
        state['last_v4_rule_action'] = v4_action
        return adjusted

    def _apply_scenario_specific_override(self, control, detection_context, tick_data, detection_result):
        cls_info = self.scenario_classifier.classify(detection_context, tick_data, self.step)
        scenario_context = ScenarioContext(
            route_id=cls_info.get('route_id') or None,
            macro_scenario=cls_info.get('macro_scenario', 'unknown'),
            scenario_name=cls_info.get('scenario_name', ''),
            ego_speed=float(tick_data.get('speed', 0.0)),
            route_command=str(tick_data.get('command_near', '')),
            detections=detection_result.get('objects', []),
            phase=cls_info.get('phase', 'unknown'),
            risk_flags=cls_info.get('risk_flags', {}),
            confidence=float(cls_info.get('confidence', 0.0)),
            frame_idx=int(self.step),
        )
        action = self.scenario_rule_registry.apply(scenario_context)
        if not action.active_rule:
            return control, 'none'

        steer = float(control.steer)
        throttle = float(control.throttle)
        brake = float(control.brake)
        speed = float(tick_data.get('speed', 0.0))

        if action.target_speed is not None and speed > float(action.target_speed):
            throttle = 0.0
            excess_speed = speed - float(action.target_speed)
            if action.active_rule == 'trucks_encountered_during_construction':
                speed_brake = min(0.65, 0.18 + 0.025 * excess_speed)
            else:
                speed_brake = min(0.18, 0.04 + 0.018 * excess_speed)
            if not scenario_context.risk_flags.get('immediate_hazard', False):
                brake = max(brake, speed_brake)
            else:
                brake = max(brake, speed_brake)
        elif action.target_speed is not None and not scenario_context.risk_flags.get('immediate_hazard', False):
            brake = min(brake, 0.03)
            if speed < float(action.target_speed) - 2.0:
                throttle = max(throttle, 0.36)
        throttle = min(throttle * float(action.throttle_scale), 0.45)
        if action.reason == 'construction_clear_recovery_release':
            throttle = max(throttle, 0.30)
            brake = min(brake, 0.02)
            low_speed_count = int(self.rule_state.get('low_speed_count', 0))
            if speed < 0.8 and low_speed_count > 45:
                steer = float(np.clip(steer, -0.16, 0.16))
                throttle = max(throttle, 0.38)
                brake = 0.0
            if speed < 0.45 and low_speed_count > 110:
                # Bounded deadlock escape for construction lane closure: slow creep with
                # a small deterministic sweep, not a route-id action sequence.
                sweep = -0.22 if ((low_speed_count // 90) % 2 == 0) else 0.22
                steer = float(np.clip(sweep, -0.24, 0.24))
                throttle = max(throttle, 0.52)
                brake = 0.0
        if action.brake is not None:
            throttle = 0.0 if float(action.brake) >= 0.30 else throttle
            brake = max(brake, float(action.brake))
        if abs(float(action.steer_bias)) > 1e-4:
            steer += float(action.steer_bias)
        steer = steer * float(action.steer_scale)
        if action.steer_limit is not None:
            steer_limit = abs(float(action.steer_limit))
            steer = float(np.clip(steer, -steer_limit, steer_limit))
        else:
            steer = float(np.clip(steer, -1.0, 1.0))
        if action.steer_smoothing is not None:
            last = self.rule_state.get('last_adjusted_control')
            if last is not None:
                alpha = float(np.clip(action.steer_smoothing, 0.0, 0.9))
                steer = (1.0 - alpha) * steer + alpha * float(last.steer)

        adjusted = self._make_control(steer, throttle, brake)
        log_period = int(os.environ.get('CVCI_V4_RULE_LOG_PERIOD', '20'))
        if log_period > 0 and self.step % log_period == 0:
            print(
                f"CVCI_V4_RULE frame={self.step} macro={scenario_context.macro_scenario} "
                f"phase={scenario_context.phase} action={action.active_rule} reason={action.reason} "
                f"speed={speed:.2f} target={action.target_speed} bias={action.steer_bias:.3f} control="
                f"{adjusted.steer:.3f},{adjusted.throttle:.3f},{adjusted.brake:.3f}",
                flush=True,
            )
        return adjusted, action.active_rule + ':' + action.reason

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose
    
    def command2hot(self,command,max_dim=6):
        if command < 0:
            command = 4
        command -= 1
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return cmd_one_hot
    
    def pos2posemb(self,pos, num_pos_feats=32, temperature=10000):
        scale = 2 * np.pi
        pos = pos * scale
        dim_t = np.arange(num_pos_feats, dtype=np.float32)
        dim_t = temperature ** (2 * (dim_t//2) / num_pos_feats)
        pos_tmp = pos[..., None] / dim_t
        posemb = np.stack((np.sin(pos_tmp[..., 0::2]), np.cos(pos_tmp[..., 1::2])), axis=-1)
        return posemb.reshape(-1)
    
    def save(self, tick_data, ego_fut_preds_fix_time, ego_fut_preds_fix_dist, draw_traj=False):
        frame = self.step //10
        Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        
        if draw_traj: # draw predict ego trajectories in bev image
            ego_fut_preds_fix_time = ego_fut_preds_fix_time[:,[1,0]]
            ego_fut_preds_fix_time = np.concatenate([ego_fut_preds_fix_time[:,], np.zeros((ego_fut_preds_fix_time.shape[0], 1)), np.ones((ego_fut_preds_fix_time.shape[0], 1))], axis=-1)
            ego_fut_preds_fix_time = np.dot(self.coor2topdown, ego_fut_preds_fix_time.T).T
            ego_fut_preds_fix_time[:, :2] /= ego_fut_preds_fix_time[:, 2:3]
            ego_fut_preds_fix_time = np.nan_to_num(ego_fut_preds_fix_time)
            for k in range(ego_fut_preds_fix_time.shape[0]):
                cv2.circle(tick_data['bev'], (int(ego_fut_preds_fix_time[k, 0]), int(ego_fut_preds_fix_time[k, 1])), 0, (0, 0, 255), 5)
            
            ego_fut_preds_fix_dist = ego_fut_preds_fix_dist[:,[1,0]]
            ego_fut_preds_fix_dist = np.concatenate([ego_fut_preds_fix_dist, np.zeros((ego_fut_preds_fix_dist.shape[0], 1)), np.ones((ego_fut_preds_fix_dist.shape[0], 1))], axis=-1)
            ego_fut_preds_fix_dist = np.dot(self.coor2topdown, ego_fut_preds_fix_dist.T).T
            ego_fut_preds_fix_dist[:, :2] /= ego_fut_preds_fix_dist[:, 2:3]
            ego_fut_preds_fix_dist = np.nan_to_num(ego_fut_preds_fix_dist)
            for k in range(ego_fut_preds_fix_dist.shape[0]):
                cv2.circle(tick_data['bev'], (int(ego_fut_preds_fix_dist[k, 0]), int(ego_fut_preds_fix_dist[k, 1])), 0, (255, 0, 0), 5)
        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    def destroy(self):
        if getattr(self, 'record_mp4_writer', None) is not None:
            try:
                self.record_mp4_writer.release()
                print(f'CVCI MP4 recording finalized: {self.record_mp4_path}', flush=True)
            except Exception as e:
                print(f'Failed to finalize MP4 recording: {e}', flush=True)
            self.record_mp4_writer = None
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
    
    
    
    
    
