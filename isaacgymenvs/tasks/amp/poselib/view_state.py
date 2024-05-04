from poselib.visualization.common import plot_skeleton_state
from poselib.skeleton.skeleton3d import SkeletonState


state = SkeletonState.from_file("data/amp_humanoid_tpose.npy")
plot_skeleton_state(state)