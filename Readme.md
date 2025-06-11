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

To run the experiments for Linear Contextual Bandits, use the following bash script:

```bash
bash LinearContextual.sh
```

## Tabular MDPs

To run the experiments for Tabular MDPs, use the following bash script:

```bash
bash TabularMDP.sh
```


## MovieLens Environment

To run the experiments for the MovieLens Environment, use the following bash script:

```bash
bash MovieLens.sh
```

## MountainCar Environment

To run the experiments for the MountainCar Environment, use the following bash script:

```bash
bash MountCar.sh
```