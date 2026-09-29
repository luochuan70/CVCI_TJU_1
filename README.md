# CVCI Bench2InterActDrive 项目说明

本仓库是在 `0824` 项目包基础上整理的 CARLA/Bench2Drive 闭环评测项目。项目提供端到端自动驾驶 agent、路线文件、场景规则、评测器调用脚本以及官方 evaluator 生成的结果。运行前请从项目提供的网盘地址下载所需模型权重，并按下文说明放入指定位置。

项目支持两类官方评测路线：

| 路线集合 | 路线数量 | 主要能力 |
|---|---:|---|
| 极端场景 | 144 | 对特殊风险场景进行识别、避险、交互和恢复 |
| 普通场景 | 220 | 路线跟随、交通规则遵守、动态障碍物处理和稳定控制 |

两类路线共用基础感知、规划和控制框架。启动时通过 `CVCI_ROUTE_SET` 选择官方路线集合和对应运行栈；普通场景在运行过程中使用在线雷达观测选择普通场景规则。整个运行过程不需要人工干预。

## 1. 项目内容

```text
0929/
├── README.md                         # 项目说明和运行方法
├── codes/
│   ├── run_all_routes.sh             # 官方路线启动脚本
│   ├── code/                         # agent 入口、路线分流和运行时调度
│   ├── DriveTransformer/             # 基础模型、规划、控制和场景规则
│   ├── CVCI_BenchMark/               # evaluator、scenario_runner 和事实生成
│   ├── scene_variants/extreme_144/   # 极端场景专用规则代码
│   ├── historical_controllers/       # 场景控制器及其来源材料
│   ├── code/OpenPCDet/               # 点云检测依赖
│   ├── runtime_path/                 # Python 运行时兼容设置
│   ├── provenance/                   # 运行来源记录
│   ├── ordinary_rules/               # 普通场景在线规则与局部规划
│   ├── harness/                      # 雷达特征与普通场景分类模块
│   ├── data/classifier_model.json    # 普通场景分类器参数
│   ├── tools/verify_package.py       # 发布包完整性检查
│   └── tests/                        # 运行时和规则单元测试
├── route/
│   ├── CVCI_BenchMark.xml            # 144 条极端场景路线
│   └── bench2drive220.xml             # 220 条普通场景路线
├── weights/                          # 从网盘下载并放置运行所需权重
│   ├── iter_25000.pth
│   └── drivetransformer_large.pth
└── results/                          # 每条路线的 evaluator 结果
    ├── 144/routes/route_<route_id>/result.json
    └── 220/routes/route_<route_id>/result.json
```

CARLA Server 和 CARLA Python API 属于官方评测环境，不包含在本仓库中。模型所需的 DriveTransformer、MMCV 源码以及 Python 3.7 编译扩展均放在 `codes/DriveTransformer` 中，不依赖仓库外的 `B2D_ZOO` 目录。除 CARLA、Python、CUDA 等环境配置外，项目代码和资源均通过仓库内相对路径访问。

## 2. 环境要求

### 2.1 软件和硬件

| 组件 | 要求 |
|---|---|
| 操作系统 | Linux，Bash 环境 |
| CARLA | CARLA 0.9.15，要求 `CARLA_ROOT/CarlaUE4.sh` 存在且可执行 |
| CARLA Python API | 与 CARLA Server 匹配的 Python API；当前启动脚本按 `carla-0.9.15-py3.7-linux-x86_64.egg` 配置 |
| Python | 当前包内编译扩展和 CARLA 0.9.15 Python egg 按 Python 3.7 构建；建议使用 Python 3.7.16 |
| GPU | NVIDIA GPU，支持 CUDA 的 PyTorch 环境 |
| 深度学习环境 | 与模型匹配的 PyTorch、CUDA、xformers、MMCV 和 OpenPCDet |
| MMCV 扩展 | 默认使用 `codes/DriveTransformer/mmcv` 中的包内扩展；目标环境 ABI 不匹配时运行 `codes/build_mmcv_extensions.sh` |

依赖清单和基础安装说明位于：

```text
codes/DriveTransformer/requirements.txt
codes/CVCI_BenchMark/scenario_runner/requirements.txt
codes/CVCI_BenchMark/leaderboard/requirements.txt
```

由于 PyTorch、CUDA、MMCV 和 CARLA Python API 存在 ABI 依赖，建议使用与评测服务器一致的环境，不要直接假定在任意 Python 环境中安装清单即可运行。

### 2.2 必需环境变量

先进入克隆后的仓库根目录，再设置环境：

```bash
cd <0929 仓库根目录>

export REPO_ROOT="$(pwd)"
export PYTHON="$(command -v python)"
export CARLA_ROOT=<CARLA_0.9.15 根目录>
export CUDA_VISIBLE_DEVICES=0
```

若评测环境使用指定的 Python 3.7 解释器，应将 `PYTHON` 显式设置为该解释器。启动脚本会自动设置项目内的 OpenPCDet、DriveTransformer、MMCV、Bench2Drive evaluator、CARLA Python API 和相关 `PYTHONPATH`。`CARLA_ROOT` 必须是 CARLA 根目录，不是 `PythonAPI` 子目录。

### 2.3 模型权重下载与放置

运行前请从下面的百度网盘下载项目所需模型权重，并将下载内容解压或复制到仓库根目录下的 `weights/` 文件夹：

- 百度网盘：[下载运行文件](https://pan.baidu.com/s/1x7SJgz9VWWaavOLNX09UBg?pwd=4a65)
- 提取码：`4a65`

下载并解压后，请保持以下目录结构和文件名：

```text
0929/
└── weights/
    ├── iter_25000.pth
    └── drivetransformer_large.pth
```

放置完成后，可在仓库根目录检查：

```bash
ls -lh weights/
```

## 3. 正确运行方法

以下命令与 `0824` 项目的正式运行入口保持一致，均从仓库根目录执行。

### 3.1 运行前预检

先检查发布包的路线、结果、权重位置和关键代码文件：

```bash
python3 codes/tools/verify_package.py
```

再运行正式启动器的预检。预检不会启动 CARLA：

```bash
CVCI_PREFLIGHT_ONLY=1 CVCI_ROUTE_SET=220 \
  bash codes/run_all_routes.sh 2100 8100
```

如需检查极端场景入口：

```bash
CVCI_PREFLIGHT_ONLY=1 CVCI_ROUTE_SET=144 \
  bash codes/run_all_routes.sh 2100 8100
```

启动器预检包括 Python、CARLA 启动器、路线 XML、agent、运行配置、依赖文件、参考路线清单和关键代码文件。

### 3.2 运行全部普通场景路线

```bash
CVCI_ROUTE_SET=220 \
  bash codes/run_all_routes.sh 2100 8100
```

默认读取 `route/bench2drive220.xml`，结果写入 `results/220`。
如果该目录已经包含完整结果，启动器会按断点续跑规则跳过对应路线；从头复现时请使用 3.6 节所示的独立输出目录。

### 3.3 运行全部极端场景路线

```bash
CVCI_ROUTE_SET=144 \
  bash codes/run_all_routes.sh 2100 8100
```

默认读取 `route/CVCI_BenchMark.xml`，结果写入 `results/144`。
如果该目录已经包含完整结果，启动器会按断点续跑规则跳过对应路线；从头复现时请使用 3.6 节所示的独立输出目录。

`CVCI_ROUTE_SET` 支持以下取值：

```text
220 或 ordinary_220
144 或 extreme_144
```

### 3.4 运行指定路线

`CVCI_ONLY_ROUTE` 使用路线 XML 中的 `route_id`，不是文件行号：

```bash
CVCI_ROUTE_SET=220 \
CVCI_ONLY_ROUTE=<route_id> \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/single_route" \
  bash codes/run_all_routes.sh 2100 8100
```

例如，若 XML 中存在路线 ID `1825`，可使用：

```bash
CVCI_ROUTE_SET=220 CVCI_ONLY_ROUTE=1825 \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/smoke_1825" \
  bash codes/run_all_routes.sh 2100 8100
```

运行一条极端路线时，将 `CVCI_ROUTE_SET` 改为 `144`，并填入对应的极端路线 ID。

### 3.5 自定义路线文件

本地测试可以通过 `CVCI_ROUTES_FILE` 指定 XML：

```bash
CVCI_ROUTES_FILE="${REPO_ROOT}/route/custom_routes.xml" \
CVCI_ONLY_ROUTE=<route_id> \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/custom" \
  bash codes/run_all_routes.sh 2100 8100
```

官方评测必须使用组织方规定的路线、地图、天气、场景触发器和传感器接口，不得通过自定义 XML 修改官方评测内容。

### 3.6 端口、续跑和独立复现目录

启动脚本参数为：

```text
bash codes/run_all_routes.sh <CARLA_RPC_PORT> <TRAFFIC_MANAGER_PORT>
```

默认端口为 `2100` 和 `8100`。evaluator 会自动启动 CARLA Server 和 Traffic Manager；运行正式命令时不要在相同端口手工启动其他 CARLA 实例。

如果输出目录中已经存在完整路线结果，脚本会自动跳过该路线。可以使用同一输出目录继续运行未完成路线：

```bash
CVCI_ROUTE_SET=220 \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/220" \
CVCI_MAX_ATTEMPTS=4 \
  bash codes/run_all_routes.sh 2100 8100
```

本仓库已经附带结果文件。若希望保留这些结果并从空目录开始独立复现，可指定新的输出目录：

```bash
CVCI_ROUTE_SET=220 \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/reproduction_220" \
  bash codes/run_all_routes.sh 2100 8100
```

极端场景从空目录复现时使用：

```bash
CVCI_ROUTE_SET=144 \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/reproduction_144" \
  bash codes/run_all_routes.sh 2100 8100
```

## 4. 路线和运行时分流

### 4.1 路线文件

| 文件 | 内容 |
|---|---|
| `route/CVCI_BenchMark.xml` | 144 条极端场景路线 |
| `route/bench2drive220.xml` | 220 条普通场景路线 |

普通路线 XML 包含 44 类场景、每类 5 条路线。路线身份仅由启动器用于选择和执行官方路线；普通场景运行时分类不读取路线文件名、路线 ID、测试顺序或 XML 场景标签。

### 4.2 运行配置

| 用途 | 配置/辅助文件 |
|---|---|
| 144 极端路线 | 极端场景专用规则和辅助检测配置 |
| 220 普通路线 | `codes/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py` |
| 220 在线分类 | `codes/data/classifier_model.json` |
| 辅助点云检测 | `codes/code/pointpillar_kitti.pth`、`codes/code/pointpillar.yaml` |

### 4.3 自动分流流程

`CVCI_ROUTE_SET` 选择对应的官方路线清单和 agent 入口：

- `144`：使用冻结的极端场景运行栈和 `weights/iter_25000.pth`；
- `220`：使用普通场景运行栈和 `weights/drivetransformer_large.pth`。

在普通场景运行栈中，第一帧有效 `RADAR_FRONT` 数据到达后，系统根据连续在线雷达观测提取特征并选择普通场景规则。分类证据不足时使用安全 fallback；分类和规则路由不依赖路线 ID、文件名或官方 XML 中的场景真值。144 极端场景路径不会进入普通场景规则路由器。

当前 agent 声明了 `RADAR_FRONT` 传感器。由于比赛规则要求使用组织方固定的传感器接口，正式评测前须确认该传感器及其配置被赛方平台允许；不得在比赛平台擅自增加或修改传感器。

## 5. 运行时执行顺序

一次路线运行由以下步骤组成：

1. 启动脚本检查环境、路线 ID、配置和运行所需文件。
2. 启动器根据 `CVCI_ROUTE_SET` 选择官方路线清单、agent 和模型权重。
3. evaluator 启动 CARLA，加载路线地图，并设置同步仿真和固定步长。
4. Traffic Manager 设置同步模式和 hybrid physics；地图加载后初始化交通灯并执行首个仿真 tick。
5. evaluator 创建 agent，agent 初始化对应的基础模型和运行时模块。
6. 普通场景 agent 在有效 `RADAR_FRONT` 数据到达后执行在线特征提取、分类和规则路由；极端场景保持原有冻结运行路径。
7. agent 执行感知、路线规划、避险和车辆控制。
8. evaluator 生成路线 JSON 和评测事实，并清理本次运行创建的进程。

CARLA Server 和 Traffic Manager 由 evaluator 统一管理。推荐使用 `run_all_routes.sh` 作为唯一正式入口，以保证地图加载、首 tick、Traffic Manager 和 agent 的初始化顺序一致。

## 6. 主要代码职责

| 文件或目录 | 作用 |
|---|---|
| `codes/run_all_routes.sh` | 正式启动器；解析路线、设置环境、逐路线调用 evaluator、处理续跑和清理 |
| `codes/code/radar_model_switch_agent.py` | 极端场景 agent 入口；声明传感器并进入冻结的极端场景运行栈 |
| `codes/code/ordinary_online_model_agent.py` | 普通场景 agent 入口；接收在线雷达观测并连接普通场景分类与规则路由 |
| `codes/code/radar_scene_gate.py` | 路线集合校验和运行证据记录 |
| `codes/code/scene_runtime_dispatch.py` | 选择极端或普通场景运行时模块 |
| `codes/harness/radar_features.py` | 从连续雷达帧和自车状态提取普通场景在线特征 |
| `codes/harness/ordinary_classifier.py` | 普通场景分类、置信度、时间平滑和安全 fallback |
| `codes/data/classifier_model.json` | 普通场景分类器参数和类别配置 |
| `codes/ordinary_rules/router.py` | 普通场景规则路由、状态隔离和生命周期管理 |
| `codes/ordinary_rules/class_01.py` 至 `class_44.py` | 44 个普通场景类别的独立规则入口 |
| `codes/ordinary_rules/local_planner.py` | 普通场景避障局部轨迹规划和轨迹跟踪辅助 |
| `codes/DriveTransformer/team_code/drivetransformer_b2d_agent.py` | 基础 agent、传感器预处理、模型推理和控制主流程 |
| `codes/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py` | DriveTransformer 模型结构和推理配置 |
| `codes/DriveTransformer/team_code/planner.py` | 路线规划和目标点生成 |
| `codes/DriveTransformer/team_code/pid_controller.py` | 车辆纵向、横向 PID 控制 |
| `codes/DriveTransformer/team_code/cvci_auxiliary_system.py` | 辅助感知、安全判断和风险处理 |
| `codes/DriveTransformer/team_code/cvci_historical_dedicated_dispatch.py` | 场景专用控制器分派 |
| `codes/DriveTransformer/team_code/cvci_scenario_classifier.py` | 根据可观测状态进行场景类别判断和运行审计 |
| `codes/DriveTransformer/team_code/cvci_scenario_context.py` | 场景上下文、状态和动作数据结构 |
| `codes/DriveTransformer/team_code/cvci_macro_rule_registry.py` | 场景规则与控制模块的归属登记 |
| `codes/DriveTransformer/team_code/cvci_scenario_rules_v4.py` | 场景规则注册和行为规则实现 |
| `codes/scene_variants/extreme_144/` | 极端场景专用规则和控制代码 |
| `codes/historical_controllers/` | 场景控制器及来源材料 |
| `codes/CVCI_BenchMark/leaderboard/leaderboard/leaderboard_evaluator.py` | 启动 CARLA、设置仿真、调用场景执行器并生成路线结果 |
| `codes/CVCI_BenchMark/leaderboard/leaderboard/carla_load_world_helper.py` | 地图加载、同步设置、交通灯初始化和首 tick |
| `codes/CVCI_BenchMark/leaderboard/leaderboard/carla_traffic_manager_helper.py` | Traffic Manager 连接、同步模式和交通物理设置 |
| `codes/CVCI_BenchMark/scenario_runner/` | 场景触发、交通参与者和路线场景执行 |
| `codes/CVCI_BenchMark/leaderboard/leaderboard/utils/facts_creator.py` | 生成路线评分所需的事实数据 |
| `codes/CVCI_BenchMark/leaderboard/leaderboard/utils/facts_to_score.py` | 将极端场景事实转换为场景分数 |

## 7. 结果和评分

### 7.1 结果目录

运行生成的结果 JSON 文件均位于仓库根目录的 `results/` 文件夹下，并按路线集合分目录保存：

```text
results/144/routes/route_<route_id>/result.json
results/220/routes/route_<route_id>/result.json
```

`results/144` 保留与 `0824` 发布包一致的 144 条极端场景 evaluator 结果。`results/220` 共包含 220 条普通路线结果：存在更新且覆盖完整五条路线类别的使用更新后的 evaluator 终态结果，其余路线保留 `0824` 的原始结果。每个路线目录保存 evaluator 生成的 `result.json`，不使用人工填写或推测分数。

### 7.2 官方评分汇总

按当前 `results/` 目录中的文件汇总，144 路线使用 `_checkpoint.global_record.scores_mean.score_challenge`，220 路线使用 `_checkpoint.global_record.scores_mean.score_composed`，结果如下：

| 路线集合 | 评分字段 | 当前分数 |
|---|---|---:|
| 144 极端场景 | `score_challenge`（B2IAD） | **83.531678** |
| 220 普通场景 | `score_composed`（B2D） | **70.237356** |

两个结果目录分别包含 144 和 220 个可解析的 evaluator 结果文件，且对应评分字段均非空。上表按各自完整路线数作为分母计算；路线自身的失败终态仍按 evaluator 给出的正式分数计入，不被删除或人工替换。

最终成绩应使用组织方公布的 evaluator 和评分脚本计算。两个组合权重必须使用组织方公布的值，且权重之和为 1；本项目不自行猜测最终组合权重。

根据比赛规则，普通路线和极端场景的计算形式为：

```text
S_B2D   = mean_j(RouteCompletion_j × InfractionPenalty_j)
S_B2IAD = mean_i(BaseScore_i × CollisionPenalty_i)
S_Final = w_B2D × S_B2D + w_B2IAD × S_B2IAD
```

极端场景碰撞惩罚按规则使用：行人碰撞乘以 `0.5`，车辆碰撞乘以 `0.6`，静态物体碰撞乘以 `0.65`。系统目标是在完成路线的同时遵守交通规则、响应危险、与交通参与者交互并恢复稳定行驶，而不是通过始终停车获得高分。

## 8. 与比赛规则对应的提交内容

本项目提交包应包含或明确提供：

- 可执行的端到端 agent 代码；
- 所需模型权重，或可复现的下载和放置方式；
- 配置文件和环境依赖说明；
- 用于官方评测的 shell script；
- 官方脚本生成的 JSON 结果（若该赛段要求提交结果）；
- 模型加载、推理、路线跟随、控制和安全处理的简要方法说明。

运行必须满足以下规则：

- 全程闭环、自动运行，不允许人工干预；
- 使用赛方固定的路线、地图、天气、交通参与者、触发器和评分逻辑；
- 不通过路线文件、场景触发器或外部真值提前获取场景答案；
- 遵守交通灯、停止标志和其他交通规则，处理车辆、行人、障碍物及危险交互；
- 在组织方评测平台上可重复运行。

极端场景覆盖的任务类别包括：

1. Lead Vehicle Disappearance
2. High-Speed Temporary Construction Barrier
3. High-Speed Reckless Lane Cutting
4. Highway Accident Vehicle
5. Construction Lane Closure with Truck
6. Roundabout Merge Conflict
7. Ghost Probe — Students Crossing Road
8. Disabled Vehicle on Road
9. E-Bike and Pedestrian Crossing Obliquely
10. Reverse Vehicle Intrusion
11. Crazy Motorbike Crossing
12. Left Turn with Blind Spot Hidden Car

## 9. GitHub 发布注意事项

两个模型权重体积较大，不作为普通 Git 文件上传。本项目通过上述百度网盘链接提供下载；使用者下载并放置到 `weights/` 后，再按照本 README 的预检和运行命令启动。下载后必须保持 README 中约定的文件名和相对路径。

MMCV/CUDA 编译扩展与操作系统、Python、PyTorch 和 CUDA 版本相关；包内软链接均指向包内相对路径。请使用 `git clone` 获取仓库，并在匹配环境中运行包完整性检查、启动器预检；ABI 不匹配时重新编译扩展。

赛方在新环境中运行前，应先完成预检，再用一条普通路线和一条极端路线进行 smoke test，确认地图加载、Traffic Manager、首 tick、传感器、运行时分流、路线结果生成和进程清理均正常，最后再运行完整路线集合。
