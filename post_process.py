import numpy as np
import os
from collections import abc
import shutil


def is_empty(data):
    """Safely check if data is empty"""
    if data is None:
        return True
    if isinstance(data, abc.Iterable) and not isinstance(data, (str, bytes)):
        try:
            return len(data) == 0
        except (TypeError, ValueError):
            return True
    return False


def inverse_rotation_matrix(rotation_matrix):
    """Calculate the inverse of a rotation matrix (inverse equals transpose)"""
    return rotation_matrix.T


def rotate_vertices(vertices, rotation_matrix):
    """Rotate vertex coordinates using a rotation matrix"""
    vertices = np.array(vertices, dtype=np.float32)
    rotation_matrix = np.array(rotation_matrix, dtype=np.float32)
    if rotation_matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")
    return vertices @ rotation_matrix.T


def rotate_joints(joints, rotation_matrix):
    """Rotate joint coordinates, handling empty cases"""
    if is_empty(joints):
        return joints
    return rotate_vertices(joints, rotation_matrix)


def get_rotation_matrix(model_id):
    """Get the inverse rotation matrix corresponding to the model ID"""
    original_id = model_id[:-1] if model_id.endswith('0') else model_id

    rotation_type1_models = [
        '13172', '13187', '13204',
        '12023', '12025', '12040', '12049', '12053', '12058'
    ]
    rotation_type2_models = [
        '13181', '13211',
        '12014', '12015', '12021', '12034', '12036', '12045', '12057'
    ]

    if original_id in rotation_type1_models:
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        print(f"Model {model_id} uses rotation matrix 1, inverse applied")
    elif original_id in rotation_type2_models:
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        print(f"Model {model_id} uses rotation matrix 2, inverse applied")
    else:
        raise ValueError(f"No rotation matrix configuration found for model {original_id}")

    return inverse_rotation_matrix(rotation_matrix)


def process_model_folder(model_dir, output_dir):
    """Process a single model folder, select rotation matrix based on model ID and remove trailing 0"""
    model_id = os.path.basename(model_dir)
    if not model_id.endswith('0'):
        print(f"Warning: Model {model_id} has no trailing 0, may not need processing")

    original_id = model_id[:-1]
    output_model_dir = os.path.join(output_dir, original_id)
    os.makedirs(output_model_dir, exist_ok=True)

    try:
        rotation_matrix = get_rotation_matrix(model_id)
    except ValueError as e:
        print(f"Error: {e}, skipping model {model_id}")
        return

    vertices_path = os.path.join(model_dir, 'transformed_vertices.npy')
    if os.path.exists(vertices_path):
        try:
            vertices = np.load(vertices_path)
            rotated_vertices = rotate_vertices(vertices, rotation_matrix)
            np.save(os.path.join(output_model_dir, 'transformed_vertices.npy'), rotated_vertices)
            print(f"[Vertices processed] {model_id} -> {original_id}")
        except Exception as e:
            print(f"Vertex processing error {model_id}: {e}")

    skeleton_path = os.path.join(model_dir, 'predict_skeleton.npy')
    if os.path.exists(skeleton_path):
        try:
            joints = np.load(skeleton_path)
            rotated_joints = rotate_joints(joints, rotation_matrix)
            np.save(os.path.join(output_model_dir, 'predict_skeleton.npy'), rotated_joints)
            print(f"[Skeleton processed] {model_id} -> {original_id}")
        except Exception as e:
            print(f"Skeleton processing error {model_id}: {e}")


def batch_process_rotations(input_root, output_root):
    """Batch process mixamo and vroid folders, apply corresponding rotation matrices by model ID"""
    target_dirs = ['mixamo', 'vroid']
    vt = ["120140", "120150", "120210", "120340", "120360", "120450", "120570"]
    mt = ["131720", "131870", "132040", "132110"]


    for dir_name in target_dirs:
        input_dir = os.path.join(input_root, dir_name)
        output_dir = os.path.join(output_root, dir_name)

        if not os.path.exists(input_dir):
            print(f"Warning: Input directory {input_dir} does not exist, skipping")
            continue

        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing {dir_name} folder...")

        for model_id in os.listdir(input_dir):
            model_dir = os.path.join(input_dir, model_id)

            if model_id in vt or model_id in mt:

                process_model_folder(model_dir, output_dir)
                shutil.rmtree(model_dir)
        print(f"{dir_name} processing completed\n")


def main():
    input_root = r"predict_test/predict"
    output_root = input_root

    if not os.path.exists(input_root):
        print(f"Error: Input directory {input_root} does not exist")
        return

    batch_process_rotations(input_root, output_root)
    print("All model rotations completed, trailing 0s removed from IDs")


if __name__ == "__main__":
    main()