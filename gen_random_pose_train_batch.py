import os
import numpy as np
import argparse
from dataset.asset import Asset
from dataset.format import HARD, parents
import tempfile  
from concurrent.futures import ThreadPoolExecutor, as_completed

def copy_asset(asset: Asset) -> Asset:
    """Deep copy asset to avoid modifying original data"""
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        np.savez(
            tmp_path,
            cls=asset.cls,
            id=asset.id,
            vertices=asset.vertices,
            vertex_normals=asset.vertex_normals,
            faces=asset.faces,
            face_normals=asset.face_normals,
            joints=asset.joints,
            skin=asset.skin,
            parents=asset.parents,
            names=asset.names,
            matrix_local=asset.matrix_local
        )
    copied_asset = Asset.load(tmp_path)
    os.remove(tmp_path)
    return copied_asset

def is_task_complete(model_file, output_root, output_prefix, num_variations, start_idx):
    """Check if all variations of the model have been generated"""
    for i in range(num_variations):
        output_idx = start_idx + i
        var_dir = os.path.join(output_root, f"{output_prefix}{output_idx}")
        output_file = os.path.join(var_dir, model_file)
        if not os.path.exists(output_file):
            return False
    return True

def process_asset(model_path, model_file, output_root, output_prefix, 
                 num_variations, start_idx, max_angle):
    """Process a single asset to generate variations with random poses"""
    try:
        original_asset = Asset.load(model_path)
        print(f"Loaded asset: {model_path} (Joints: {original_asset.J})")
    except Exception as e:
        print(f"Failed to load {model_path}: {str(e)}")
        return
    
    for i in range(num_variations):
        output_idx = start_idx + i
        var_dir = os.path.join(output_root, f"{output_prefix}{output_idx}")
        os.makedirs(var_dir, exist_ok=True)
        
        npz_output = os.path.join(var_dir, model_file)
        if os.path.exists(npz_output):
            continue
        
        try:
            asset = copy_asset(original_asset)
            matrix_basis = asset.get_random_matrix_basis(random_pose_angle=max_angle)
            asset.apply_matrix_basis(matrix_basis)
            
            np.savez(
                npz_output,
                vertices=asset.vertices,
                joints=asset.joints
            )
        except Exception as e:
            print(f"Error processing {model_file} variation {i + 1}: {str(e)}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"Asset {model_file} generated {i + 1}/{num_variations} variations")

def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Asset variation generation with parallel processing')
    parser.add_argument('--cls', required=True, choices=['mixamo', 'vroid'], 
                        help='Specify the category to process (mixamo or vroid)')
    parser.add_argument('--total_workers', type=int, required=True, 
                        help='Total number of parallel workers')
    parser.add_argument('--worker_id', type=int, required=True, 
                        help='Current worker ID (starting from 0)')
    parser.add_argument('--max_angle', type=float, default=30.0,
                        help='Maximum rotation angle for random poses')
    parser.add_argument('--num_variations', type=int, default=1000,
                        help='Number of variations to generate per asset')
    parser.add_argument('--start_idx', type=int, default=1,
                        help='Starting index for output directories')
    args = parser.parse_args()

    # Configuration paths
    input_dir = f"data/train/{args.cls}"
    output_root = "data/train"
    output_prefix = f"{args.cls}-"
    
    print(f"Using {'HARD' if HARD else 'EASY'} mode (Joints: {len(parents)})")
    os.makedirs(output_root, exist_ok=True)

    # Get all model files and sort them for consistent sharding
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
        
    model_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npz")])
    if not model_files:
        print(f"No NPZ files found in {input_dir}")
        return

    # Task sharding: distribute files evenly across workers
    total = len(model_files)
    per_worker = total // args.total_workers
    remainder = total % args.total_workers
    
    # Calculate file index range for current worker
    start = args.worker_id * per_worker + min(args.worker_id, remainder)
    end = start + per_worker + (1 if args.worker_id < remainder else 0)
    assigned_files = model_files[start:end]
    
    print(f"Worker {args.worker_id} assigned {len(assigned_files)} files ({start}-{end-1}) out of {total} total")

    # Process assigned files
    for model_file in assigned_files:
        model_path = os.path.join(input_dir, model_file)
        
        # Check if all variations are already generated
        if is_task_complete(model_file, output_root, output_prefix, 
                           args.num_variations, args.start_idx):
            print(f"Skipping completed asset: {model_file}")
            continue
        
        # Process the asset
        try:
            process_asset(
                model_path=model_path,
                model_file=model_file,
                output_root=output_root,
                output_prefix=output_prefix,
                num_variations=args.num_variations,
                start_idx=args.start_idx,
                max_angle=args.max_angle
            )
            print(f"Completed processing: {model_file}")
        except Exception as e:
            print(f"Failed to process {model_file}: {str(e)}")

    print(f"Worker {args.worker_id} processing completed")

if __name__ == "__main__":
    main()
