from poselib.skeleton.skeleton3d import SkeletonState
from poselib.visualization.common import plot_skeleton_state


# motion = SkeletonMotion.from_fbx("data/dog/fbx/D1_004_KAN01_001.fbx", fps=120, root_joint="Hips")
pose = SkeletonState.from_file("data/dog/mjcf_dog_tpose.npy")



plot_skeleton_state(pose)