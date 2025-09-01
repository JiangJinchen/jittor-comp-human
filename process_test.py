import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
from math import cos, sin, radians
from collections.abc import Iterable
import time


def is_empty(data):
    """Check if data is empty safely"""
    if data is None:
        return True
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        try:
            return len(data) == 0
        except (TypeError, ValueError):
            return True
    return False


def rotate_vertices(vertices, rotation_matrix):
    """Rotate vertex coordinates using rotation matrix"""
    if vertices is None:
        return None
    vertices = np.array(vertices, dtype=np.float32)
    return vertices @ rotation_matrix.T


def rotate_joints(joints, rotation_matrix):
    """Rotate joint coordinates with empty value handling"""
    if is_empty(joints):
        return joints
    return rotate_vertices(joints, rotation_matrix)


def ensure_directory_exists(directory):
    """Ensure the directory exists"""
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"Failed to create directory {directory}: {e}")
            return False
    return True


def visualize_point_cloud_comparison(original, rotated, output_img_dir, model_id):
    """Visualize original and rotated point clouds for comparison"""
    if not ensure_directory_exists(output_img_dir):
        print("Failed to create image directory, skipping image saving")
        return None

    output_path = os.path.join(output_img_dir, f"{model_id}.png")
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=1, alpha=0.6)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], s=1, alpha=0.6, c='r')
    ax2.set_title('Rotated Point Cloud')
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')

    for ax in [ax1, ax2]:
        ax.view_init(elev=30, azim=45)
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        max_range = max([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
        mid = [sum(lim) / 2 for lim in [xlim, ylim, zlim]]
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    plt.suptitle(f"Model {model_id} Orientation Adjustment", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def process_rotated_npz(data, rotation_matrix, output_path, model_id, output_img_dir=None, export_image=True, processed_files=None, dataset_root=None):
    """Process rotated NPZ file, modify ID field and save directly"""
    fields_to_rotate = ['vertices', 'joints']
    rotated_data = {}

    for key in data.keys():
        if key in fields_to_rotate:
            if key == 'vertices':
                rotated_data[key] = rotate_vertices(data[key], rotation_matrix)
            elif key == 'joints':
                rotated_data[key] = rotate_joints(data[key], rotation_matrix)
        else:
            rotated_data[key] = data[key]

    if 'id' in rotated_data:
        original_id = rotated_data['id']
        try:
            if isinstance(original_id, (int, float, np.number)):
                new_id = int(original_id) * 10
            elif isinstance(original_id, str):
                new_id = original_id + '0'
            else:
                new_id = str(original_id) + '0'

            rotated_data['id'] = new_id
            print(f"ID modified: {original_id} -> {new_id}")
        except Exception as e:
            print(f"Error processing ID: {e}")
            return False
    else:
        print("Warning: 'id' field not found in file")
        return False

    try:
        np.savez(output_path, **rotated_data)
        print(f"  - Rotated NPZ saved: {output_path}")
        
        if processed_files is not None and dataset_root is not None:
            
            rel_path = os.path.relpath(output_path, dataset_root)
            processed_files.append(rel_path)
        
        if export_image and 'vertices' in data and not is_empty(data['vertices']) and output_img_dir:
            output_img = visualize_point_cloud_comparison(
                data['vertices'], rotated_data['vertices'], output_img_dir, model_id
            )
            if output_img:
                print(f"  - Image saved: {output_img}")
        return True
    except Exception as e:
        print(f"  - Failed to save rotated NPZ: {e}")
        return False


def process_single_npz(npz_path, model_id, subdir, output_root_dir, output_img_dir=None, export_image=True, processed_files=None, dataset_root=None):
    """Process a single NPZ file directly without temp file"""
    start_time = time.time()
    print(f"Processing model: {model_id} (from {subdir} folder)")

    final_output_npz = os.path.join(os.path.dirname(npz_path), f"{model_id}0.npz")
    
    success = False

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  - Error loading {npz_path}: {e}")
        return False

    rotation_matrix = None
    if subdir == 'mixamo':
        if model_id in ['13172', '13187', '13204']:
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])
        elif model_id in [ '13211']:
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
        else:
            print(f"  - Not in specified rotation list, skipping\n")
            return False
    elif subdir == 'vroid':
        if model_id in ['12014', '12015', '12021', '12034', '12036', '12045', '12057']:
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
        else:
            print(f"  - Not in specified rotation list, skipping\n")
            return False
    else:
        print(f"  - Warning: Unknown folder {subdir}, skipping\n")
        return False

    if rotation_matrix is not None:
        if process_rotated_npz(data, rotation_matrix, final_output_npz, model_id, output_img_dir, export_image, processed_files, dataset_root):
            print(f"  - Directly saved: {final_output_npz}")
            success = True
    
    print(f"  - Processing completed, time taken: {time.time() - start_time:.2f} seconds\n")
    return success


def batch_process_directories(input_root, output_img_dir=None, export_image=True, dataset_root=None):
    """Batch process NPZ files and generate list file with relative paths"""
    if export_image and output_img_dir:
        ensure_directory_exists(output_img_dir)
    
    target_subdirs = ['mixamo', 'vroid']
    success_count = 0
    total_count = 0
    processed_count = 0
    processed_files = []

    for subdir in target_subdirs:
        subdir_path = os.path.join(input_root, subdir)
        
        if not os.path.exists(subdir_path) or not os.path.isdir(subdir_path):
            print(f"Warning: Subfolder {subdir_path} does not exist, skipping")
            continue
        
        print(f"Starting to process subfolder: {subdir_path}")
        
        npz_files = [
            os.path.join(subdir_path, f) for f in os.listdir(subdir_path)
            if f.endswith('.npz') and os.path.isfile(os.path.join(subdir_path, f))
        ]
        
        if not npz_files:
            print(f"  - Warning: No NPZ files found in {subdir_path}")
            continue
        
        total_count += len(npz_files)
        print(f"  - Found {len(npz_files)} NPZ files, starting processing...\n")
        
        for npz_file in npz_files:
            try:
                model_id = os.path.basename(npz_file).split('.')[0]
                processed_count += 1
                if process_single_npz(npz_file, model_id, subdir, input_root, output_img_dir, export_image, processed_files, dataset_root):
                    success_count += 1
            except Exception as e:
                print(f"  - Error processing file: {e}")
                continue
        
        print(f"  - Subfolder {subdir_path} processing completed, {success_count}/{processed_count} files succeeded\n")
    
    if processed_files and dataset_root:
        list_file_path = os.path.join(dataset_root, "post_process_test_list.txt")
        with open(list_file_path, 'w') as f:
            for file_path in processed_files:
                f.write(f"{file_path}\n")
        print(f"Post process list generated: {list_file_path}, total {len(processed_files)} files")
    
    print(f"Batch processing completed: Checked {total_count} files, processed {processed_count} eligible files, {success_count} succeeded")
    return processed_files


def main():
    """Main function"""
    input_root = "data/test"
    output_image_directory = "test_reshaped_png"
    export_images = False
    dataset_root = "data"  

    if not os.path.exists(input_root) or not os.path.isdir(input_root):
        print(f"Error: Input directory {input_root} does not exist")
        return

    batch_process_directories(input_root, output_image_directory, export_images, dataset_root)


if __name__ == "__main__":
    main()
