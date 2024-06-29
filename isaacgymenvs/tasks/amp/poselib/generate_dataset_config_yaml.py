
import os
import yaml


def generate_config_by_dir(directory):
    motion_files = []
    motion_weights = []

    for file in os.listdir(directory):
        if file.endswith(".npy"):
            motion_files.append(file)
            motion_weights.append(1.0)
    
    # normalize weights
    total_weight = sum(motion_weights)
    motion_weights = [w / total_weight for w in motion_weights]

    # write yaml
    motion_list = []
    for i in range(len(motion_files)):
        motion_dict = {}
        motion_dict["file"] = motion_files[i]
        motion_dict["weight"] = motion_weights[i]
        motion_list.append(motion_dict)
    
    with open(os.path.join(directory, "dataset_config.yaml"), "w") as f:
        yaml.dump({"motions": motion_list}, f)

if __name__ == "__main__":
    generate_config_by_dir("data/dog/retarget")