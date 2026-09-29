from setuptools import setup, find_packages

setup(
    name='scenario_runner',  # 包的名称
    version='0.9.15',        # 版本号（随便写一个即可，这里用常用的 CARLA 版本号）
    packages=find_packages(), # 自动寻找当前目录下的所有 Python 包（会找到 srunner）
)
