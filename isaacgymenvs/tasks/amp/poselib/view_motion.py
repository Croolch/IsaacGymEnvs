from poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive


# motion = SkeletonMotion.from_fbx("data/dog/fbx/D1_010_KAN01_001.fbx", fps=120, root_joint="Hips")
# motion = SkeletonMotion.from_file("data/dog/npy/D1_071_KAN02_002.npy")
motion = SkeletonMotion.from_file("../../../../assets/amp/motions/amp_humanoid_walk.npy")



plot_skeleton_motion_interactive(motion)