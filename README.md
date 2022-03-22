# Deep Reinforcement Learning for Entity Alignment [pdf](https://openreview.net/pdf?id=Bi4BpAAqx0)

## Introduction

Embedding-based methods have attracted increasing attention in recent entity alignment (EA) studies. Although great promise they can offer, there are still several limitations. The most notable is that they identify the aligned entities based on cosine similarity, ignoring the semantics underlying the embeddings themselves. Furthermore, these methods are shortsighted, heuristically selecting the closest entity as the target and allowing multiple entities to match the same candidate. To address these limitations, we model entity alignment as a sequential decision-making task, in which an agent sequentially decides whether two entities are matched or mismatched based on their representation vectors. The proposed reinforcement learning (RL)-based entity alignment framework can be flexibly adapted to most embedding-based EA methods. The experimental results demonstrate that it consistently advances the performance of several state-of-the-art methods, with a maximum improvement of 31.1% on Hits@1.

## Dependencies

Please first download the dataset from [OpenEA](https://github.com/nju-websoft/OpenEA), and then install gym and required packages of OpenEA:

```bash
conda create -n openea python=3.6
conda activate openea
conda install tensorflow-gpu==1.8
conda install -c conda-forge graph-tool==2.29
conda install -c conda-forge python-igraph
pip install -r requirement.txt
pip install gym
```

## Quick Start

Use the following scripts to run RLEA with RDGCN as basic EEA model on D-Y:

```bash
cd run
python runRLEA.py --model_name rdgcn --dataset D_Y
```

If run with the stored embeddings:

```bash
python runRLEA.py --model_name rdgcn --dataset D_Y --restore_embeddings True
```

For SEA which has projection matrices:

```bash
python runRLEA.py --model_name sea --dataset D_Y --mapping True
```


Currently available EEA models: JAPE, SEA, RSN, RDGCN, AlignE, BootEA. 

The corresponding model names: jape, sea, rsn, rdgcn, aligne, bootea.

Currently available datasets: EN_FR, EN_DE, D_W, D_Y