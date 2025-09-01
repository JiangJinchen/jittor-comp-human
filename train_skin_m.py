import jittor as jt
import numpy as np
import os
import argparse
import time
import random
from tqdm import tqdm
from jittor import nn
from jittor import optim
from scipy.spatial import cKDTree
from dataset.dataset import get_dataloader, transform
from dataset.format import id_to_name
from dataset.sampler import SamplerMix
from models.sss import create_model

from dataset.exporter import Exporter
from models.metrics import J2J
# Set Jittor flags
jt.flags.use_cuda = 1
from jittor.contrib import concat
from math import sqrt
def train(args):
    """
    Main training function

    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')

    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)

    # Log training parameters
    log_message(f"Starting training with parameters: {args}")

    # Create model
    model = create_model(
        model_name=args.model_name,
    )

    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)

    # Create optimizer
    # if args.optimizer == 'sgd':
    #     optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
    #                           weight_decay=args.weight_decay)
    # elif args.optimizer == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # else:
    #     raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # wan cheng wang luo de fen kai chuan ru


    # ske_params = model.mlp.parameters() + model.pct.parameters()
    # skin_params = model.joint_mlp.parameters() + model.vertex_mlp.parameters() + model.pct.parameters()
    # optimizer_skin = optim.Adam(skin_params, lr=0.0001, weight_decay=1e-4)
    # optimizer_ske = optim.Adam(ske_params, lr=0.00001, weight_decay=1e-4)

    # optimizer_skin = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    # optimizer_ske = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    # Create loss function
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    num_ver = args.num_ver
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=num_ver, vertex_samples=num_ver, Val=False),
        transform=transform,
        data_name=args.data_name,
        random_pose=args.random_pose,
    )

    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SamplerMix(num_samples=num_ver, vertex_samples=num_ver, Val=True),
            transform=transform,
            data_name=args.data_name,
            random_pose=args.random_pose,
        )
    else:
        val_loader = None

    predict_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.predict_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=SamplerMix(num_samples=num_ver, vertex_samples=num_ver, Val=True),
        transform=transform,
        return_origin_vertices=True,
        data_name=args.data_name,
        random_pose=False,
    )
    predict_output_dir = args.predict_output_dir
    predict_skin_path = args.predict_skin_path
    # Training loop
    best_loss1 = 99999999
    best_loss2 = 99999999
    for epoch in range( args.epochs):
        # Training phase
        model.train()
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        train_lose_joint = 0.0
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, real_joints, skin = data['vertices'], data['joints'], data['skin']

            res_joint, res_skin = model(vertices, real_joints=None)

            # real_joints = real_joints.reshape(real_joints.shape[0], -1)
            # loss_joint = criterion_mse(res_joint, real_joints)

            loss_mse = criterion_mse(res_skin, skin)
            loss_l1 = criterion_l1(res_skin, skin)
            # loss = loss_mse + loss_l1 + loss_joint
            loss = loss_mse + loss_l1
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()

            # Calculate statistics
            train_loss_mse += loss_mse.item()
            train_loss_l1 += loss_l1.item()
            # train_lose_joint += loss_joint.item()

            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                # log_message(f"Epoch [{epoch + 1}/{args.epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                #             f"Loss mse: {loss_mse.item():.5f} Loss l1: {loss_l1.item():.5f} Loss joint: {loss_joint.item():.5f}")
                log_message(f"Epoch [{epoch + 1}/{args.epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                            f"Loss mse: {loss_mse.item():.5f} Loss l1: {loss_l1.item():.5f} ")

        # Calculate epoch statistics
        train_loss_mse /= len(train_loader)
        train_loss_l1 /= len(train_loader)
        train_lose_joint /= len(train_loader)
        epoch_time = time.time() - start_time

        log_message(f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Train Loss mse: {train_loss_mse:.5f} "
                    f"Train Loss l1: {train_loss_l1:.5f} "
                    f"Train Loss joint: {train_lose_joint:.5f}  "
                    f"Time: {epoch_time:.2f}s "
                    f"LR: {optimizer.lr:.6f}  "
                    )

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss_l1 = 0.0
            val_loss_j2j = 0.0
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints_val, skins = data['vertices'], data['joints'], data['skin']
                #############################################################################################################
                # vertices, skin, cls, id = data['vertices'], data['skin'], data['cls'], data['id']
                # joints_val = []
                # for i in range(len(cls)):
                #     path = os.path.join(predict_output_dir, 'val', cls[i], str(id[i].item()), "predict_skeleton.npy")
                #     data = np.load(path)
                #     joints_val.append(data)
                # joints_val = jt.array(joints_val)

                #############################################################################################################
                # Forward pass
                res_joints, res_skins = model(vertices, real_joints=None)
                joints_val = joints_val.reshape(res_joints.shape[0], -1)
                for i in range(res_joints.shape[0]):

                    loss_j2j = J2J(res_joints[i].reshape(-1, 3), joints_val[i].reshape(-1, 3))
                    loss_l1 = criterion_l1(res_skins[i], skins[i])

                    val_loss_j2j += loss_j2j.item()
                    val_loss_l1 += loss_l1.item()

            # Calculate validation statistics
            val_loss_l1 /= len(val_loader.paths)
            val_loss_j2j /= len(val_loader.paths)
            log_message(f"Validation Loss: l1: {val_loss_l1:.5f}  Loss: joints: {val_loss_j2j:.5f} ")
        #
        #     # Save best model
            if val_loss_l1 < best_loss1:  # best 0.0134         0.016           0.020
                print("start predicting...")

                for batch_idx, data in tqdm(enumerate(predict_loader)):
                    # currently only support batch_size==1 because origin_vertices is not padded
                    vertices, cls, id, origin_vertices, N = data['vertices'], data['cls'], data['id'], data['origin_vertices'], data['N']

                    # load predicted skeleton
                    B = vertices.shape[0]
                    res_joints, res_skins = model(vertices, real_joints=None)
                    res_joints = res_joints.reshape(B, -1, 3)
                    for i in range(B):
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

                        path = os.path.join(predict_output_dir, predict_skin_path, 'epoch_{}'.format(epoch + 1), cls[i], str(id[i].item()))
                        os.makedirs(path, exist_ok=True)

                        # np.save(os.path.join(path, "predict_skeleton"), res_joints[i])
                        np.save(os.path.join(path, "predict_skin"), skin_resampled)
                        np.save(os.path.join(path, "transformed_vertices"), o_vertices)
                print("finished")
            if val_loss_l1 < best_loss1:

                best_loss1 = val_loss_l1
                log_message(f"Saved best model with best l1 loss {val_loss_l1:.5f}  best joint loss {val_loss_j2j:.5f}  ")
                model_path = os.path.join(args.output_dir, 'best_model_{}.pkl'.format(epoch + 1))
                model.save(model_path)
                best_model_path = os.path.join(args.output_dir, 'best_model.pkl'.format(epoch + 1))
                model.save(best_model_path)
        #
        #         # if val_loss_l1 < 0.02:
        #         #     model_path = os.path.join(args.output_dir, 'best_model_{}.pkl'.format(epoch+1))
        #         #     model.save(model_path)
        #         # log_message(f"Saved best model with l1 loss {best_loss:.4f} to {model_path}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    # final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    # model.save(final_model_path)
    # log_message(f"Training completed. Saved final model to {final_model_path}")
    # return model


def main():
    """Parse arguments and start training"""
    output_path = "skin_m_4096_1e-5_12000"
    data_name = 'mixamo' # 'vroid' , 'mixamo' , ''  ,   ''  is two dataset
    # data_name = 'mixamo' # 'vroid' , 'mixamo' , ''  ,   ''  is two dataset
    random_pose = 0
    # random_pose = 1




    parser = argparse.ArgumentParser(description='Train a point cloud model')

    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, default='data/train_list.txt',
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='data/val_list.txt',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    parser.add_argument('--predict_data_list', type=str, default='data/test_list.txt',
                        help='Path to the training data list file')

    parser.add_argument('--predict_skin_path', type=str, default=output_path,
                        help='Path to the training data list file')
    parser.add_argument('--predict_output_dir', type=str, default=output_path,
                        help='Path to the training data list file')
    parser.add_argument('--output_dir', type=str, default=output_path,
                        help='Directory to save output files')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct2',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default="",
                        help='Path to pretrained model')

    # Training parameters
    parser.add_argument('--num_ver', type=int, default=4096,
                        help='Batch size for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=12000,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--random_pose', type=int, default=random_pose,
                        help='Apply random pose to asset')
    # Output parameters

    parser.add_argument('--print_freq', type=int, default=30,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    parser.add_argument('--data_name', type=str, default=data_name,
                        help='mixamo, vroid')
    args = parser.parse_args()

    # Start training
    train(args)


def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    seed_all(123)
    main()