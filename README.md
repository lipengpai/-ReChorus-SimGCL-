![logo](./docs/_static/logo2.0.png)
---

![PyPI - Python Version](https://img.shields.io/badge/pyhton-3.10-blue) 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/THUwangcy/ReChorus) 
[![arXiv](https://img.shields.io/badge/arXiv-ReChorus-%23B21B1B)](https://arxiv.org/abs/2405.18058)

## 本课程实验：在 ReChorus 中复现 SimGCL

本仓库在官方 ReChorus 2.0 基础上增加了 SIGIR 2022 模型
`SimGCL`，并提供 BPRMF、LightGCN、SimGCL 在两个 ReChorus 数据集上的
统一对比流程。仓库保留可复现源码、自动测试、实际运行日志、最终汇总结果和
实验报告；体积较大的数据副本与模型检查点不纳入版本控制，可按下述命令生成。

主要新增内容：

- `src/models/general/SimGCL.py`：LightGCN 编码、逐层随机噪声增强、BPR + InfoNCE 联合目标；
- `src/models/general/LightGCN.py`：修复上游代码强制调用 CUDA 的问题，支持 CPU/GPU 自动迁移；
- `experiments/prepare_movielens.py`：确定性的 MovieLens-1M 5-core、时间划分和 99 负例处理；
- `experiments/make_pilot_dataset.py`：从完整数据构造资源受限但可重复的 CPU 子集；
- `experiments/run_experiments.py`：运行两数据集、三模型的统一对比；
- `experiments/run_full_cpu.py`：在完整数据上分段运行、跳过已完成项目并执行累计时间限制；
- `experiments/tune_simgcl.py`：仅依据验证集选择 SimGCL 超参数；
- `experiments/results/final_results.csv` 与 `comparison.png`：实测结果和结果图；
- `experiments/results_full_cpu/results.csv`：两个完整数据集上的最终 CPU 结果；
- `tests/`：模型前向/反向传播、CPU 兼容性和数据处理测试。
- `report/`：最终实验报告 PDF、Overleaf 源文件及同目录插图。

### 项目结构说明

```text
ReChorus/
├─ data/                       # 数据集、数据格式示例及预处理说明
├─ docs/                       # ReChorus 原始文档、教程和示意图
├─ experiments/                # 本实验的数据处理、训练、调参、汇总和绘图脚本
│  ├─ results/                 # CPU 子集实验结果、调参记录、日志和对比图
│  └─ results_full_cpu/        # 完整数据 CPU 实验结果、状态与审计日志
├─ report/                     # LaTeX 报告、最终 PDF、插图及绘图脚本
├─ src/                        # ReChorus 框架与推荐模型源码
│  ├─ helpers/                 # 数据读取、训练、评估和任务 Runner
│  ├─ models/                  # 各类推荐模型及本实验实现的 SimGCL
│  └─ utils/                   # 通用工具函数和神经网络层
├─ tests/                      # 本实验的自动化测试
├─ requirements.txt            # ReChorus 原始依赖
├─ requirements-experiment.txt # 本实验建议安装的精简依赖
├─ LICENSE                     # 项目许可证
└─ README.md                   # 项目入口、复现命令与结构说明
```

各目录的主要内容如下。

| 路径 | 主要内容与用途 |
| --- | --- |
| `data/` | 保存框架数据及数据说明。`Grocery_and_Gourmet_Food/` 包含 Amazon Grocery 的训练集、验证集、测试集、物品元数据及 Reader 缓存；`MovieLens_1M/` 保存 MovieLens-1M 的原始数据处理入口；`MIND_Large/` 是上游框架保留的数据示例。完整实验使用的 `ML_1MTOPK/` 由 `experiments/prepare_movielens.py` 生成，因体积原因不提交仓库。 |
| `docs/` | ReChorus 2.0 的使用文档。根目录文档介绍安装、参数和支持模型；`tutorials/` 提供不同推荐任务的 Notebook 教程；`demo_scripts_results/` 保存官方示例脚本与结果说明；`_static/` 保存文档图片。 |
| `experiments/` | 本次课程实验的主工作区。`prepare_movielens.py` 处理 MovieLens-1M，`make_pilot_dataset.py` 构造 CPU 子集，`run_experiments.py` 运行子集对比，`tune_simgcl.py` 进行 SimGCL 验证集调参，`run_full_cpu.py` 和 `run_full_gpu.ps1` 分别执行完整数据 CPU/GPU 实验，`summarize_results.py` 与 `plot_results.py` 汇总并绘制结果。 |
| `experiments/results/` | 保存资源受限子集实验的 `final_results.csv`、`simgcl_tuning.csv`、训练日志和 `comparison.png`。这些结果用于实现验证和超参数行为分析。 |
| `experiments/results_full_cpu/` | 保存完整数据实验的 `results.csv`、`status.json` 和逐模型日志；`interrupted/`、`retry/` 记录中断实验、重新运行及独立评估过程，用于检查点语义和可复现性审计。模型检查点体积较大，不纳入版本控制。 |
| `report/` | 保存可直接上传 Overleaf 的 `main.tex`、最终报告 PDF、四张报告插图以及 `generate_report_figures.py`。该脚本可依据实验结果重新生成报告中的流程图、早停曲线和完整数据对比图。 |
| `src/` | ReChorus 主程序源码。`main.py` 是命令行训练入口，`exp.py` 用于批量实验；`helpers/` 实现 Reader 与 Runner；`models/` 按 general、sequential、context、context_seq、reranker 等任务分类保存模型；`utils/` 提供公共层与工具函数。 |
| `src/models/general/` | 本实验重点模型目录。`BPRMF.py` 与 `LightGCN.py` 是对比基线，`SimGCL.py` 是新增实现；其中 LightGCN 同时完成了 CPU/GPU 设备兼容性修复。其余文件为 ReChorus 原有的通用推荐模型。 |
| `tests/` | 自动化回归测试。分别检查 SimGCL 的前向传播、损失与梯度，MovieLens 数据预处理，完整 CPU 运行控制，以及中断续跑时模型、优化器和随机状态的恢复。可通过 `python -m unittest discover -s tests -v` 一次运行。 |

根目录中的 `.gitignore` 用于排除原始大数据、缓存、临时文件和模型检查点；
`requirements-experiment.txt` 是复现实验的推荐依赖入口，而
`requirements.txt` 保留上游 ReChorus 的完整环境定义。

### 快速复现实测 CPU 实验

```powershell
pip install -r requirements-experiment.txt
python -m unittest discover -s tests -v
python experiments/run_experiments.py --epochs 10 --seed 2026
python experiments/tune_simgcl.py --epochs 10 --seed 2026
python experiments/summarize_results.py
python experiments/plot_results.py
```

实测使用嵌入维度 32、2 层图传播、batch size 1024、Adam 学习率 0.001、
L2=1e-4、最多 10 轮、early stop=3。每个评估样本含 1 个正例和 99 个负例，
因此 ReChorus 的 HR@K 等价于 Recall@K。随机种子固定为 2026。

### 完整数据 CPU 实验

```powershell
python experiments/run_full_cpu.py --time-limit-hours 10 --epochs 200 `
  --patience 5 --runs-per-invocation 1 --seed 2026
```

每次默认只完成一组“数据集—模型”组合；再次执行同一命令会跳过已经完成的
组合。时间限制会计入结果目录已有日志中的训练时间，避免分段启动后重新获得
完整的 10 小时预算。每个完整轮次还会保存独立的续跑状态，其中包含最后模型、
Adam 优化器、验证历史及随机数状态；若进程异常中断，可从上一完整轮严格继续。

### 完整数据 GPU 实验

Amazon Grocery 数据沿用框架目录 `data/Grocery_and_Gourmet_Food`。MovieLens
原始数据与处理后的 `data/ML_1MTOPK` 不提交到 GitHub；下载并解压 MovieLens-1M
后，可运行以下命令重新生成确定性数据：

```powershell
python experiments/prepare_movielens.py `
  --ratings data/MovieLens_1M/raw/ml-1m/ratings.dat `
  --movies data/MovieLens_1M/raw/ml-1m/movies.dat `
  --output-dir data/ML_1MTOPK --seed 2026
```

准备好两个完整数据集后，在 CUDA 环境中运行：

```powershell
powershell -ExecutionPolicy Bypass -File experiments/run_full_gpu.ps1
```

完整配置采用论文式参考设置：64 维、3 层、batch size 2048、最多 200 轮、
early stop=5、SimGCL 的 temperature=0.2、eps=0.1、cl_rate=0.2。
其中 `cl_rate` 不是适用于所有数据集的固定最优值；论文也针对不同数据集调参。
CPU 子集结果仅用于验证实现和分析超参数行为，不能与原论文全量数据结果直接横向比较。


ReChorus2.0 is a modular and task-flexible PyTorch library for recommendation, especially for research purpose. It aims to provide researchers a flexible framework to implement various recommendation tasks, compare different algorithms, and adapt to diverse and highly-customized data inputs. We hope ReChorus2.0 can serve as a more convinient and user-friendly tool for researchers, so as to form a "Chorus" of recommendation tasks and algorithms.

The previous version of ReChorus can be found at [ReChorus1.0](https://github.com/THUwangcy/ReChorus/tree/ReChorus1.0)

## What's New in ReChorus2.0:

- **New Tasks**: Newly supporting the context-aware top-k recommendation and CTR prediction task. Newly supporting the Impression-based re-ranking task.
- **New Models**: Adding Context-aware Recommenders and Impression-based Re-ranking Models. Listed below.
- **New dataset format**: Supporting various contextual feature input. Customizing candidate item lists in training and evaluation. Supporting variable length positive and negative samples.
- **Task Flexible**: Each model can serve for different tasks, and task switching is conveniently achieved by altering *model mode*.
  

This framework is especially suitable for researchers to choose or implement desired experimental settings, and compare algorithms under the same setting. The characteristics of our framework can be summarized as follows:

- **Modular**: primary functions modularized into distinct components: runner, model, and reader, facilitating code comprehension and integration of new features.
  
- **Swift**: concentrate on your model design ***in a single file*** and implement new models quickly.

- **Efficient**: multi-thread batch preparation, special implementations for the evaluation, and around 90% GPU utilization during training for deep models.

- **Flexible**: implement new readers or runners for different datasets and experimental settings, and each model can be assigned with specific helpers.

## Structure

Generally, ReChorus decomposes the whole process into three modules:

- [Reader](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseReader.py): read dataset into DataFrame and append necessary information to each instance
- [Runner](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseRunner.py): control the training process and model evaluation, including evaluation metrics.
- [Model](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BaseModel.py): define how to generate output (predicted labels or ranking scores) and prepare batches.

![logo](./docs/_static/module_new.png)

## Requirements & Getting Started
See in the doc for [Requirements & Getting Started](https://github.com/THUwangcy/ReChorus/tree/master/docs/Getting_Started.md).

## Tasks & Settings

The tasks & settings are listed below

<table>
<tr><th> Tasks </th><th> Runner </th><th> Metrics </th><th> Loss Functions</th><th> Reader </th><th> BaseModel </th><th> Models</th><th> Model Modes </th></tr>
<tr><td rowspan="3"> Top-k Recommendation </td><td rowspan="3"> BaseRunner </td><td rowspan="3"> HitRate NDCG </td><td rowspan="3"> BPR </td><td> BaseReader </td><td> BaseModel.GeneralModel </td><td> general </td><td> '' </td></tr>
<tr><td> SeqReader </td><td> BaseModel.SequentialModel </td><td> sequential </td><td> '' </td></tr>
<tr><td> ContextReader </td><td> BaseContextModel.ContextModel </td><td> context </td><td> 'TopK' </td></tr>
<tr><td> CTR Prediction </td><td> CTRRunner </td><td> AUC Logloss </td><td> BPR, BCE </td><td> ContextReader </td><td> BaseContextModel.ContextCTRModel </td><td> context </td><td> 'CTR' </td></tr>
<tr><td rowspan="4"> Impression-based Ranking </td><td rowspan="4"> ImpressionRunner </td><td rowspan="4"> HitRate NDCG MAP </td><td rowspan="4"> List-level BPR, Listnet loss, Softmax cross entropy loss, Attention rank </td><td> ImpressionReader </td><td> BaseImpressionModel.ImpressionModel </td><td> general </td><td> 'Impression' </td></tr>
<tr><td> ImpressionSeqReader </td><td> BaseImpressionModel.ImpressionSeqModel </td><td> sequential </td><td> 'Impression' </td></tr>
<tr><td> ImpressionReader </td><td> BaseRerankerModel.RerankModel </td><td> reranker </td><td> 'General' </td></tr>
<tr><td> ImpressionSeqReader </td><td> BaseRerankerModel.RerankSeqModel </td><td> reranker </td><td> 'Sequential' </td></tr>
</table>


## Arguments
See in the doc for [Main Arguments](https://github.com/THUwangcy/ReChorus/tree/master/docs/Main_Arguments.md).

## Models
See in the doc for [Supported Models](https://github.com/THUwangcy/ReChorus/tree/master/docs/Supported_Models.md).

Experimental results and corresponding configurations are shown in [Demo Script Results](https://github.com/THUwangcy/ReChorus/tree/master/docs/demo_scripts_results/README.md).


## Citation

**If you find ReChorus is helpful to your research, please cite either of the following papers. Thanks!**

```
@inproceedings{li2024rechorus2,
  title={ReChorus2. 0: A Modular and Task-Flexible Recommendation Library},
  author={Li, Jiayu and Li, Hanyu and He, Zhiyu and Ma, Weizhi and Sun, Peijie and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
  pages={454--464},
  year={2024}
}
```
```
@inproceedings{wang2020make,
  title={Make it a chorus: knowledge-and time-aware item modeling for sequential recommendation},
  author={Wang, Chenyang and Zhang, Min and Ma, Weizhi and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={109--118},
  year={2020}
}
```
```
@article{王晨阳2021rechorus,
  title={ReChorus: 一个综合, 高效, 易扩展的轻量级推荐算法框架},
  author={王晨阳 and 任一 and 马为之 and 张敏 and 刘奕群 and 马少平},
  journal={软件学报},
  volume={33},
  number={4},
  pages={0--0},
  year={2021}
}
```

This is also our public implementation for the following papers (codes and datasets to reproduce the results can be found at corresponding branch):


- *Chenyang Wang, Min Zhang, Weizhi Ma, Yiqun Liu, and Shaoping Ma. [Make It a Chorus: Knowledge- and Time-aware Item Modeling for Sequential Recommendation](http://www.thuir.cn/group/~mzhang/publications/SIGIR2020Wangcy.pdf). In SIGIR'20.*

```bash
git clone -b SIGIR20 https://github.com/THUwangcy/ReChorus.git
```

- *Chenyang Wang, Weizhi Ma, Min Zhang, Chong Chen, Yiqun Liu, and Shaoping Ma. [Towards Dynamic User Intention: Temporal Evolutionary Effects of Item Relations in Sequential Recommendation](https://chenchongthu.github.io/files/TOIS-KDA-wcy.pdf). In TOIS'21.*

```bash
git clone -b TOIS21 https://github.com/THUwangcy/ReChorus.git
```

- *Chenyang Wang, Weizhi Ma, Chong, Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. [Sequential Recommendation with Multiple Contrast Signals](https://dl.acm.org/doi/pdf/10.1145/3522673). In TOIS'22.*

```bash
git clone -b TOIS22 https://github.com/THUwangcy/ReChorus.git
```

- *Chenyang Wang, Zhefan Wang, Yankai Liu, Yang Ge, Weizhi Ma, Min Zhang, Yiqun Liu, Junlan Feng, Chao Deng, and Shaoping Ma. [Target Interest Distillation for Multi-Interest Recommendation](). In CIKM'22.*

```bash
git clone -b CIKM22 https://github.com/THUwangcy/ReChorus.git
```

## Contact

**ReChorus 1.0**: Chenyang Wang (THUwangcy@gmail.com)

**ReChorus 2.0**: Jiayu Li (lijiayu997@gmail.com), Hanyu Li (l-hy12@outlook.com)

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
