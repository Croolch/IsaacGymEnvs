# How to generate motion dataset?

This document describes how to generate a motion dataset using the `poselib` package. To generate a motion dataset, you must prepare the following:
1. A mujoco model file (MJCF) that describes the skeleton of the character.
2. A motion dataset in the form of a `.fbx` file.
3. A configuration file that describes the mapping between the skeleton joints in mocap file and the MJCF file.

There will be several steps to generate a final motion dataset. Let's take cmu mocap and amp_humanoid as an example.

## Step 1: Genarate character T-pose file
First, we need to generate a T-pose file for the character. This file will be used as a reference pose for retargeting the motion data. To generate the T-pose file, check the `generate_amp_humanoid_tpose.py` script in the `poselib` package. You should adjust the pose as T-pose for your character and some other parameters in the script.

```bash
python generate_amp_humanoid_tpose.py
```

## Step 2: Generate motion dataset T-pose file
Next, we need to generate a T-pose file for the motion dataset. This file will be used as a reference pose for retargeting the motion data. To generate the T-pose file, check the `generate_cmu_tpose.py` script in the `poselib` package. 

```bash
python generate_cmu_tpose.py
```

## Step 3: Prepare the motion dataset
Prepare the motion dataset in the form of a `.fbx` file. Then run `retarget_motion_from_fbx.py` script to retarget the motion data to the character skeleton.

```bash
python retarget_motion_from_fbx.py
```

## Step 4: Generate dataset config yaml
You should have a directory contains a batch of retargeted motion data. Then run `generate_dataset_config_yaml.py` script to generate the dataset config yaml file. The default weights for every motion data is 1.0. You can adjust the weights in the generated yaml file.

```bash
python generate_dataset_config_yaml.py
```