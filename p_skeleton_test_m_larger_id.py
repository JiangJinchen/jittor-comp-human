import jittor as jt
import numpy as np
import os
import argparse
import random
from tqdm import tqdm
import shutil
from collections import abc

from dataset.dataset_skeleton import get_dataloader, transform
from dataset.sampler import SamplerMix
from models.skeleton import create_model

# Set Jittor flags
jt.flags.use_cuda = 1

def predict(args):
    """Main prediction function, including original prediction and post-processing replacement"""
    print("First, making original predictions...")
    original_predict_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.predict_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=SamplerMix(num_samples=args.num_ver, vertex_samples=args.num_ver, Val=True),
        transform=transform,
        return_origin_vertices=True,
        data_name=args.data_name,
        random_pose=False,
    )
        
    print("start predicting original test...")
    for batch_idx, data in tqdm(enumerate(original_predict_loader)):
        vertices, cls, id, origin_vertices, N = data['vertices'], data['cls'], data['id'], data['origin_vertices'], data['N']
        
        # Reshape input if needed
        if vertices.ndim == 3:  # [B, N, 3]
            vertices = vertices.permute(0, 2, 1)  # [B, 3, N]
            
        # Forward pass
        outputs = model(vertices)
        outputs = outputs.reshape(outputs.shape[0], -1, 3)
            
        # Save results
        for i in range(len(cls)):
            path = os.path.join(args.predict_output_dir, cls[i], str(id[i].item()))
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, "predict_skeleton"), outputs[i])
            np.save(os.path.join(path, "transformed_vertices"), origin_vertices[i, :N[i]].numpy())
        
        print("Original prediction finished")
    

def main():
    """Parse arguments and start prediction"""
    predict_output_path = 'predict'
    data_name = 'mixamo'  # 'vroid', 'mixamo', or empty for both datasets
    
    parser = argparse.ArgumentParser(description='Predict using a trained skeleton model with post processing')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    parser.add_argument('--predict_data_list', type=str, default='data/test_list.txt',
                        help='Path to the original prediction data list file')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct2',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default="checkpoint/skeleton_m_larger_id/best_model_715.pkl",
                        help='Path to pretrained model')
    
    # Prediction parameters
    parser.add_argument('--num_ver', type=int, default=4096,
                        help='Number of vertices to sample')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for prediction')
    parser.add_argument('--predict_output_dir', type=str, default="predict_test/"+predict_output_path,
                        help='Directory to save prediction results')
    parser.add_argument('--data_name', type=str, default=data_name,
                        help='Dataset name: mixamo, vroid, or empty for both')
    
    args = parser.parse_args()
    
    # Create model once for both predictions
    global model  
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type
    )
    
    # Load pretrained model
    if not args.pretrained_model:
        raise ValueError("Please specify a pretrained model using --pretrained_model")
    
    print(f"Loading pretrained model from {args.pretrained_model}")
    model.load(args.pretrained_model)
    model.eval()  # Set model to evaluation mode
    
    predict(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()
