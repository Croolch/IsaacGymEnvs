# python: bvh to fbx

import bpy
import os
import sys

def get_bvh_files(directory):
    bvh_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.bvh'):
                bvh_files.append(os.path.join(root, file))
    return bvh_files

def add_tpose():
    # copy tpose to armature's current animation
    # pose mode
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')

    curr_anim = bpy.context.object.animation_data.action
    tpose_anim = bpy.data.actions["tpose"]

    first_frame = curr_anim.frame_range[0]
    for fcurve in tpose_anim.fcurves:
        curr_anim.fcurves.find(fcurve.data_path, index=fcurve.array_index).keyframe_points.insert(first_frame-1, fcurve.keyframe_points[0].co.y)

    # change root location
    root_bone = bpy.context.object.pose.bones[0]
    root_fcurve_loc_x = curr_anim.fcurves.find(f'pose.bones["{root_bone.name}"].location', index=0)
    root_fcurve_loc_y = curr_anim.fcurves.find(f'pose.bones["{root_bone.name}"].location', index=1)
    root_fcurve_loc_z = curr_anim.fcurves.find(f'pose.bones["{root_bone.name}"].location', index=2)

    root_fcurve_loc_x.keyframe_points.insert(frame=first_frame-1, value=root_fcurve_loc_x.keyframe_points[1].co[1])
    root_fcurve_loc_y.keyframe_points.insert(frame=first_frame-1, value=root_fcurve_loc_y.keyframe_points[1].co[1])
    root_fcurve_loc_z.keyframe_points.insert(frame=first_frame-1, value=root_fcurve_loc_z.keyframe_points[1].co[1])


    bpy.ops.object.mode_set(mode='OBJECT')
        

def bvh_to_fbx(bvh_file, fbx_file=None):
    clear_scene()
    
    bpy.ops.import_anim.bvh(filepath=bvh_file, update_scene_fps=True)
    if fbx_file is None:
        output_dir = os.path.join(os.getcwd(), 'fbx')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fbx_file = os.path.join(output_dir, os.path.basename(bvh_file).replace('.bvh', '.fbx'))
    
    # edit animation
    add_tpose()
    
    # push to nla strip
    add_nla_strip()

    bpy.ops.export_scene.fbx(filepath=fbx_file, use_selection=True, bake_anim_use_all_actions=False)

def add_nla_strip():
    action = bpy.context.object.animation_data.action
    if action is not None:
        track = bpy.context.object.animation_data.nla_tracks.new()
        track.name = action.name
        track.strips.new(action.name, int(action.frame_range[0]), action)
        bpy.context.object.animation_data.action = None

def clear_scene():
    # clear all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    # clear animation
    for anim in bpy.data.actions:
        # reserve tpose animation
        if anim.name[-5:] == "tpose":
            continue
        bpy.data.actions.remove(anim)

if __name__ == '__main__':

    dir_path = sys.argv[-1]
    if dir_path is None or not os.path.isdir(dir_path) or len(dir_path) == 0:
        dir_path = os.getcwd()
        print('No directory path provided, using current working directory')
    
    bvh_files = get_bvh_files(dir_path)
    for bvh_file in bvh_files:
        bvh_to_fbx(bvh_file)
        print(f'{bvh_file} converted to fbx')
