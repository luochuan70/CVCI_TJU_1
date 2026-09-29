<div align="center">   

# DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving
</div>

![teaser](docs/overall.png)

>Official implementation of paper [DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving](https://arxiv.org/abs/2503.07656). *Xiaosong Jia, Junqi You, Zhiyuan Zhang, Junchi Yan*. **ICLR 2025**


**DriveTransformer** offers a unified, parallel, and synergistic approach to end-to-end autonomous driving, facilitating easier training and scalability. The framework is composed of three unified operations: **task self-attention, sensor cross-attention, temporal cross-attention** and has three key features:
* **Task Parallelism:** All agent, map, and planning queries direct interact with each other at each block.
* **Sparse Representation:** Task queries direct interact with raw sensor features.
* **Streaming Processing:** Task queries are stored and passed as history information. 


## Getting Started

- [Download and installation](docs/INSTALL.md)
- [Data preprocessing](docs/DATA_PREP.md)
- [Training and Evaluation](docs/TRAIN_EVAL.md)

## Model and Result

| Model | Driving Score | Success Rate(%) | Efficiency | Comfortness | Latency | Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: |:---: |:---: |
| DriveTransformer-Large | 63.46 | 35.01 | 100.64 | 20.78 | 211.7ms | [config](adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py) | [Google Drive](https://drive.google.com/file/d/1wAXFWfjJm0cmP_pmgTkwxTUEs6Zu5j6i/view?usp=sharing)/[Baidu Cloud](https://pan.baidu.com/s/1ZunlLWRJXIblEG_L8rxRew?pwd=1234) |
 
## Citation 

```bibtex
@inproceedings{jia2025drivetransformer,
  title={DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving},
  author={Xiaosong Jia and Junqi You and Zhiyuan Zhang and Junchi Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```