# HybridResources: Augmenting Online RL with Offline Data

This repository contains the implementation of the paper **"Augmenting Online RL with Offline Data is All You Need: A Unified Hybrid RL Algorithm Design and Analysis"**. The repository is written in Python and provides a unified framework for combining offline and online in reinforcement learning.

## Requirements

To install the required dependencies, use the following command:
```bash
pip install numpy matplotlib gymnasium tqdm pillow pickle5
```

# Tasks

This repository includes implementations for four tasks: Linear Contextual Bandits, Tabular MDPs, MovieLens Environment, and MountainCar Environment. Each task involves generating sub-optimality gap and regret experiments, followed by visualization of the results.

## Linear Contextual Bandits

 - Generate Sub-optimality Gap Experiments:

```
python Contextual_Hybrid_UCB.py
```

 - Generate Regret Experiments:

```
python Contextual_Hybrid_UCB-Regret.py
```

## Tabular MDPs

 - Generate Sub-optimality Gap Experiments:

```
python TabularMDP_Hybrid_UCB.py
```

 - Generate Regret Experiments:

```
python TabularMDP_Hybrid_UCB-Regret.py
```

## Main Paper Plots

To generate the plots of experiments in the main paper, linear contextual bandits and tabular MDPs, run:

```
python draw.py
```

## MovieLens Environment

 - Generate Sub-optimality Gap and Regret Experiments:

```
python MovieLens_Hybrid_UCB.py
python MovieLens_Hybrid_UCB-Regret.py
```

 - Plot Results:

```
python MovieLens_draw.py
```

## MountainCar Environment

Since the offline datasets are generated a little different from previous tasks, we need to first generate the offline datasets with the following command.

 - Generate Offline Datasets:

```
python MountCar_Generate_OfflineData.py
```

 - Generate Sub-optimality Gap and Regret Experiments:

```
python MountCar_Hybrid_UCB.py
python MountCar_Hybrid_UCB-Regret.py
```

 - Plot Results:

```
python MountCar_draw.py
```