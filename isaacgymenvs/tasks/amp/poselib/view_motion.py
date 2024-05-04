from poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive


# motion = SkeletonMotion.from_fbx("data/dog/fbx/D1_010_KAN01_001.fbx", fps=120, root_joint="Hips")
motion = SkeletonMotion.from_file("data/amp_humanoid_dance.npy")



plot_skeleton_motion_interactive(motion)