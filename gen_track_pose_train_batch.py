import os
import numpy as np
import argparse
from dataset.asset import Asset

def main():
    # Add command line argument parsing (for multi-terminal sharding)
    parser = argparse.ArgumentParser(description='Data augmentation parallel processing')
    parser.add_argument('--cls', required=True, choices=['mixamo', 'vroid'], 
                        help='Specify the category to process (mixamo or vroid)')
    parser.add_argument('--total_workers', type=int, required=True, 
                        help='Total number of parallel terminals')
    parser.add_argument('--worker_id', type=int, required=True, 
                        help='Current terminal ID (starting from 0)')
    args = parser.parse_args()

    # Configure paths
    track_dir = "data/track"
    train_dir = "/mnt/jittorjc/data/train"
    output_root = "/mnt/jittorjc/data/train"
    mixamo_output_prefix = "mixamo-"
    vroid_output_prefix = "vroid-"
    start_idx = 1001
    end_idx = 1512  # 1001+512-1=1512
    assert end_idx - start_idx + 1 == 512, "Need to generate 512 data groups"

    # 1. Process track data: Merge all sequences and sample every 2 frames
    all_matrix_basis = []
    for track_file in os.listdir(track_dir):
        if track_file.endswith(".npz"):
            track_path = os.path.join(track_dir, track_file)
            data = np.load(track_path)
            matrix_basis = data["matrix_basis"]  # shape: (frame, J, 4, 4)
            sampled = matrix_basis[::2]  # Sample every 2 frames
            all_matrix_basis.append(sampled)
    
    # Merge and ensure total count is 512
    all_matrix_basis = np.concatenate(all_matrix_basis, axis=0)
    if len(all_matrix_basis) < 512:
        repeat = (512 // len(all_matrix_basis)) + 1
        all_matrix_basis = np.tile(all_matrix_basis, (repeat, 1, 1, 1))[:512]
    else:
        all_matrix_basis = all_matrix_basis[:512]
    print(f"Generated {len(all_matrix_basis)} matrix_basis for augmentation")

    # 2. Process models of specified category (only process the category specified by args.cls)
    cls = args.cls
    cls_dir = os.path.join(train_dir, cls)
    if not os.path.exists(cls_dir):
        print(f"Category directory does not exist: {cls_dir}")
        return
    
    # Get all model files under this category and sort them (to ensure consistency in multi-terminal sharding)
    model_files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".npz")])
    if not model_files:
        print(f"No model files found in category {cls}")
        return

    # 3. Task sharding: Current terminal only processes model files assigned to it
    # Distribute files evenly across workers (e.g., 1323 files to 4 terminals: 331,331,331,330)
    total = len(model_files)
    per_worker = total // args.total_workers
    remainder = total % args.total_workers
    # Calculate the file index range for current terminal
    start = args.worker_id * per_worker + min(args.worker_id, remainder)
    end = start + per_worker + (1 if args.worker_id < remainder else 0)
    assigned_files = model_files[start:end]
    print(f"Current terminal (ID: {args.worker_id}) assigned {len(assigned_files)} files ({start}-{end-1})")

    # 4. Process assigned files and skip completed tasks
    output_prefix = mixamo_output_prefix if cls == "mixamo" else vroid_output_prefix
    for model_file in assigned_files:
        model_path = os.path.join(cls_dir, model_file)
        # Check if this model has completed all 512 transformations (process if any is missing)
        all_done = True
        for i in range(512):
            output_idx = start_idx + i
            output_cls_dir = os.path.join(output_root, f"{output_prefix}{output_idx}")
            save_path = os.path.join(output_cls_dir, model_file)
            if not os.path.exists(save_path):
                all_done = False
                break
        if all_done:
            print(f"Model {model_file} has completed all transformations, skipping")
            continue

        # Process incomplete transformations
        print(f"Start processing model: {model_file}")
        try:
            asset = Asset.load(model_path)
            for i in range(512):
                output_idx = start_idx + i
                output_cls_dir = os.path.join(output_root, f"{output_prefix}{output_idx}")
                save_path = os.path.join(output_cls_dir, model_file)
                # Skip already completed single transformations
                if os.path.exists(save_path):
                    continue
                # Create output directory
                os.makedirs(output_cls_dir, exist_ok=True)
                # Apply transformation and save
                asset_copy = Asset(** asset.__dict__)  # Deep copy to avoid cumulative transformations
                asset_copy.apply_matrix_basis(all_matrix_basis[i])
                save_data = {
                    "cls": "",
                    "id": -1,
                    "vertices": asset_copy.vertices,
                    "vertex_normals": np.array([], dtype=np.float32),
                    "faces": np.array([], dtype=np.int64),
                    "face_normals": np.array([], dtype=np.float32),
                    "joints": asset_copy.joints,
                    "skin": np.array([], dtype=np.float32),
                    "parents": [],
                    "names": [],
                    "matrix_local": np.array([], dtype=np.float32)
                }
                np.savez(save_path, **save_data)
        except Exception as e:
            print(f"Error processing {model_path}: {e}")

    print(f"Terminal {args.worker_id} processing completed")

if __name__ == "__main__":
    main()
