import jittor as jt
import numpy as np
import os
import argparse
from jittor import nn
from dataset.dataset_smallid import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from models.skeleton import create_model

from tqdm import tqdm

import random
from models.metrics import J2J
# Set Jittor flags
jt.flags.use_cuda = 1

def predict(args):
    # Create model
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type
    )
    num_ver = args.num_ver
    sampler = SamplerMix(num_samples=num_ver, vertex_samples=num_ver, Val=True)
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        model.load(args.pretrained_model)
    
    predict_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.predict_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=True,
        data_name=args.data_name,
        random_pose=args.random_pose,
    )
    print("start predicting...")
    print("predict_loader path :", len(predict_loader.paths))
    model.eval()
    val_loss = 0.0
    J2J_loss = 0.0
    
    criterion = nn.MSELoss()
    for batch_idx, data in enumerate(predict_loader):
        # Get data and labels
        vertices, joints, cls, id, origin_vertices, N = data['vertices'], data['joints'], data['cls'], data['id'], data['origin_vertices'], data['N']
        joints = joints.reshape(joints.shape[0], -1, 3)

        # Reshape input if needed
        if vertices.ndim == 3:  # [B, N, 3]
            vertices = vertices.permute(0, 2, 1)  # [B, 3, N]

        # Forward pass
        outputs = model(vertices)
        outputs = outputs.reshape(outputs.shape[0], -1, 3)

        loss = criterion(outputs, joints)

        val_loss += loss.item()
        for i in range(outputs.shape[0]):
            J2J_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item()
            # path = os.path.join(args.predict_output_dir, cls[i], str(id[i].item()))
            # os.makedirs(path, exist_ok=True)
            # np.save(os.path.join(path, "predict_skeleton"), outputs[i])
            # o_vertices = origin_vertices[i, :N[i]].numpy()
            # np.save(os.path.join(path, "transformed_vertices"), o_vertices)
            #
            # np.save(os.path.join(path, "predict_skeleton_gt"), joints[i])
            # np.save(os.path.join(path, "J2J_loss"), J2J_loss)


    # Calculate validation statistics
    val_loss /= len(predict_loader.paths)
    J2J_loss /= len(predict_loader.paths)

    print(f"Validation Loss: {val_loss:.5f} J2J Loss: {J2J_loss:.5f}")

def main():
    output = 'skeleton_4096_vm1e-5_12000'
    data_name = 'mixamo' # 'vroid' , 'mixamo' , ''  ,   ''  is two dataset
    # data_name = 'mixamo'  # 'vroid' , 'mixamo' , ''  ,   ''  is two dataset
    # random_pose = 0
    random_pose = 0

    parser = argparse.ArgumentParser(description='Train a point cloud model')
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    parser.add_argument('--predict_data_list', type=str, default='data/val_list.txt',
                        help='Path to the prediction data list file')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct2',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='checkpoint/skeleton_m_small_id/best_model_4876.pkl',
                        help='Path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_ver', type=int, default=4096,
                        help='Batch size for training')
    parser.add_argument('--random_pose', type=int, default=random_pose,
                        help='Apply random pose to asset')
    # Predict parameters
    parser.add_argument('--predict_output_dir', type=str, default='predict_val/'+output,
                        help='Path to store prediction results')
    parser.add_argument('--data_name', type=str, default=data_name,
                        help='mixamo, vroid')
    args = parser.parse_args()
    
    predict(args)
    print(args.pretrained_model)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()