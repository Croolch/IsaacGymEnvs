import torch

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_motion, plot_skeleton_state

motion = SkeletonMotion.from_fbx("data/amass/fbx/amass-male-pickup-tpose.fbx", root_joint="pelvis", fps=60, is_local=True)
skeleton = motion.skeleton_tree
# generate zero rotation pose
local_rotations = motion.local_rotation[0]
local_rotations[skeleton.index("pelvis")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
    local_rotations[skeleton.index("pelvis")]
)
local_rotations[skeleton.index("pelvis")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([180.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), 
    local_rotations[skeleton.index("pelvis")]
) 

global_translations = motion.global_translation[0]
min_height = torch.min(global_translations[:, 2])
root_translation = motion.root_translation[0]
y_offset = 1.125
root_translation += torch.tensor([0.0, -y_offset, -min_height])
zero_pose = SkeletonState.from_rotation_and_root_translation(skeleton, local_rotations, root_translation, is_local=True)

# save and visualize T-pose
zero_pose.to_file("data/amass/amass_tpose_local.json")
plot_skeleton_state(zero_pose)