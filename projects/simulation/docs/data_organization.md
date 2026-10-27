# Data Organization Guide

This guide explains how to organize data for SimEngine. The platform uses a modular data structure where each subsystem (PolicyEngine, SimEngine) has its own data requirements while sharing common formats.

## Quick Overview

```
SimEngine/
├── data/                          # Main data directory
│   ├── raw/                       # Raw datasets (nuPlan, OpenScene)
│   ├── policy_engine/                # PolicyEngine-specific data
│   └── sim_engine/                # SimEngine-specific data
├── projects/
│   ├── PolicyEngine/                 # Algorithm training & evaluation
│   └── SimEngine/                 # Closed-loop photorealistic simulation
├── experiments/                   # Experiment outputs
└── data_output/                   # Default data outputs dir
```

---

## Data Download

We provide pre-processed datasets and model checkpoints via **[ModelScope](https://www.modelscope.cn/datasets/OpenDriveLab/SimEngine)** and **Hugging Face[https://huggingface.co/datasets/OpenDriveLab/SimEngine]**.

### Option 1: Download from ModelScope (Recommended for Users in China)

```bash
pip install modelscope
modelscope download --dataset OpenDriveLab/SimEngine
```

### Option 2: Download from Hugging Face

```bash
# Install Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Download the dataset
hf download OpenDriveLab/SimEngine --repo-type dataset --local-dir /path/to/your/SimEngine_repo
```

> **Note:** Stay tuned to our [News section](../README.md#-news).

---

## Directory Structure

### 1. Raw Data (`data/raw/`)

Raw datasets from nuPlan and OpenScene.

```bash
data/raw/
├── nuplan/                        # nuPlan raw dataset
│   └── dataset/
│      ├── maps/                  # HD maps (required for all modules)
│      │   ├── nuplan-maps-v1.0.json
│      │   ├── us-nv-las-vegas-strip/
│      │   ├── us-ma-boston/
│      │   ├── us-pa-pittsburgh-hazelwood/
│      │   └── sg-one-north/
│      └── nuplan-v1.1/
│          ├── sensor_blobs/      # Camera images and LiDAR
│          └── splits/            # Train/val/test splits
│   
│
└── openscene-v1.1/                # OpenScene dataset (nuPlan-based)
    ├── sensor_blobs/
    │   ├── trainval/              # Training sensor data
    │   └── test/                  # Test sensor data
    └── meta_datas/
        ├── trainval/              # Training metadata
        └── test/                  # Test metadata
```

**Setup commands:**

```bash
cd SimEngine/data/raw

# Create symlink to nuPlan dataset
ln -s /path/to/nuplan nuplan

# Create symlink to OpenScene dataset
ln -s /path/to/openscene-v1.1 openscene-v1.1
```

---

### 2. PolicyEngine Data (`data/policy_engine/`)

Data for end-to-end model training and evaluation.

```bash
data/policy_engine/
├── openscene-synthetic/           # Synthetic data from SimEngine
│   ├── sensor_blobs/
│   ├── meta_datas/
│   └── pdms_pkl/
│
├── ckpts/                         # Pre-trained model checkpoints
│
├── pdms_cache/                    # Pre-computed PDM metrics cache
│   ├── pdm_8192_gt_cache_navtrain.pkl
│   └── pdm_8192_gt_cache_navtest.pkl
│
├── merged_infos_navformer/
│   ├── nuplan_openscene_navtrain.pkl
│   └── nuplan_openscene_navtest.pkl
│
│
└── test_8192_kmeans.npy          # K-means clustering for PDM
```

---

### 3. SimEngine Data (`data/sim_engine/`)

Data for closed-loop simulation.

```bash
data/sim_engine/
├── assets/                        # Scene assets for simulation
│   ├── navtest
│   ├── navtrain
│   └── navtest_failures
│
└── scenarios/                     # Scenario configurations
    ├── original/                  # Original logged scenarios
    │   ├── navtest_failures/
    │   ├── navtrain_50pct_collision/
    │   ├── navtrain_ep_per1/
    │   ├── navtrain_failures_per1/
    │   └── navtrain_hydramdp_failures/
    │
    └── augmented/                 # Augmented scenarios (from BWM)
        ├── navtrain_50pct_collision/
        ├── navtrain_50pct_ep_1pct/
        └── navtrain_50pct_offroad/

```

---

### 4. Experiments Output (`experiments/`)

Generated experiment results and logs.

```bash
experiments/
└── closed_loop_exps/              # Closed-loop simulation results
    ├── exp_vadv2_50pct_ep20/
    │   ├── navtest_failures_NR/   # Non-reactive results
    │   │   ├── split_0/ ... split_7/
    │   │   ├── plan_traj/
    │   │   └── sim_output/
    │   │       └── openscene_format/
    │   │           ├── meta_datas/
    │   │           ├── pdms_pkl/
    │   │           ├── sensor_blobs/
    │   │           └── all_scenes_pdm_averages_NR.csv
    │   │
    │   └── navtest_failures_R/    # Reactive results
    │       └── (same structure as NR)
    │
    └── exp_vadv2_100pct_ep20/
        └── (similar structure)
```

This directory is automatically created during simulation runs.

---

## Environment Variables

Set these environment variables for proper data access:

```bash
# Add to ~/.bashrc or ~/.zshrc
export SIMREPO_ROOT="/path/to/SimEngine"
export NUPLAN_MAPS_ROOT="${SIMREPO_ROOT}/data/raw/nuplan/maps"
export PYTHONPATH=$SIMREPO_ROOT:$PYTHONPATH
```

Apply changes:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

---