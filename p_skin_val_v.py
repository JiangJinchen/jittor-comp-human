import jittor as jt
import numpy as np
import os
import argparse

from dataset.asset import Asset
from dataset.dataset_smallid import get_dataloader, transform
from dataset.sampler import SamplerMix
from models.sss import create_model

import numpy as np
from scipy.spatial import cKDTree
import random

from tqdm import tqdm

# Set Jittor flags
jt.flags.use_cuda = 1
from models.metrics import J2J


def log_message(message):
    """Helper function to log messages to file and print to console"""
    # with open(log_file, 'a') as f:
    #     f.write(f"{message}\n")
    print(message)
def predict(args):
    # Create model
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type
    )
    num_ver = args.num_ver
    sampler = SamplerMix(num_samples=num_ver, vertex_samples=num_ver, Val=True)
    predict_output_dir = args.predict_output_dir
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
        data_name = args.data_name,
        random_pose=args.random_pose,

    )
    print("start predicting...")
    from jittor import nn
    criterion_l1 = nn.L1Loss()
    val_loss_l1 = 0.0
    val_loss_j2j = 0.0
    model.eval()
    for batch_idx, data in enumerate(predict_loader):
        # Get data and labels
        vertices, joints_val, skins, origin_vertices, N, cls, id = data['vertices'], data['joints'], data['skin'], data['origin_vertices'],  data['N'], data['cls'], data['id'],
        # Forward pass
        res_joints, res_skins = model(vertices, real_joints=None)

        for i in range(vertices.shape[0]):
            # resample
            skin = res_skins[i].numpy()
            o_vertices = origin_vertices[i, :N[i]].numpy()

            tree = cKDTree(vertices[i].numpy())
            distances, indices = tree.query(o_vertices, k=3)

            weights = 1 / (distances + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)  # normalize

            # weighted average of skin weights from the 3 nearest joints
            skin_resampled = np.zeros((o_vertices.shape[0], skin.shape[1]))
            for v in range(o_vertices.shape[0]):
                skin_resampled[v] = weights[v] @ skin[indices[v]]

            path = os.path.join(predict_output_dir, cls[i], str(id[i].item()))
            os.makedirs(path, exist_ok=True)

            np.save(os.path.join(path, "predict_skin"), skin_resampled)
            np.save(os.path.join(path, "transformed_vertices"), o_vertices)


        joints_val = joints_val.reshape(res_joints.shape[0], -1)
        for i in range(res_joints.shape[0]):

            loss_j2j = J2J(res_joints[i].reshape(-1, 3), joints_val[i].reshape(-1, 3))
            loss_l1 = criterion_l1(res_skins[i], skins[i])

            val_loss_j2j += loss_j2j.item()
            val_loss_l1 += loss_l1.item()

    # Calculate validation statistics
    val_loss_l1 /= len(predict_loader.paths)
    val_loss_j2j /= len(predict_loader.paths)
    log_message(f"Validation Loss: l1: {val_loss_l1:.5f}  Loss: joints: {val_loss_j2j:.5f} ")

    print("finished")

def main():


    output = "skin_v_4096_1e-5_12000+test"
    data_name = 'vroid' # 'vroid' , 'mixamo' , ''  ,   ''  is two dataset
    # data_name = 'mixamo' # 'vroid' , 'mixamo' , ''  ,   ''  is two dataset
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
    parser.add_argument('--pretrained_model', type=str, default='checkpoint/skin_v/best_model_76.pkl',
                        help='Path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_ver', type=int, default=4096,
                        help='Batch size for training')
    # Predict parameters
    parser.add_argument('--predict_output_dir', type=str, default='predict_val/'+output,
                        help='Path to store prediction results')
    parser.add_argument('--random_pose', type=int, default=random_pose,
                        help='Apply random pose to asset')
    parser.add_argument('--data_name', type=str, default=data_name,
                        help='mixamo, vroid')
    args = parser.parse_args()

    predict(args)


def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    seed_all(123)
    main()