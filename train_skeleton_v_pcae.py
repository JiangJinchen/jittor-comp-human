import jittor as jt
import numpy as np
import os
import argparse
import time
import random
import tqdm
from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from dataset.format import num_joints
from models.skeleton_pcae import create_model

from models.metrics import J2J

# Set Jittor flags
jt.flags.use_cuda = 1


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
        model_type=args.model_type,
        output_channels=num_joints * 3,
    )

    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    # print(model)
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Create loss function
    criterion = nn.MSELoss()

    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024, vertex_samples=1024, Val=False),
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
            sampler=SamplerMix(num_samples=1024, vertex_samples=1024, Val=True),
            transform=transform,
            return_origin_vertices=True,
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
        sampler=SamplerMix(num_samples=1024, vertex_samples=1024, Val=True),
        transform=transform,
        return_origin_vertices=True,
        data_name=args.data_name,
        random_pose=False,
    )
    predict_output_dir = args.predict_output_dir
    best_loss = 99999999
    # Training loop
    for epoch in range(480, args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints = data['vertices'], data['joints']
            # print(vertices.shape)
            # print(joints.shape)
            vertices = vertices.permute(0, 2, 1)  # [B, 3, N]

            outputs = model(vertices)
            # joints = joints.reshape(outputs.shape[0], -1)
            #  print(outputs.shape)
            #  print(joints.shape)
            # print(joints.item())
            # print(outputs.item())
            loss = criterion(outputs, joints)

            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()

            # Calculate statistics
            train_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch + 1}/{args.epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                            f"Loss: {loss.item():.4f}")

        # Calculate epoch statistics
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time

        log_message(f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Train Loss: {train_loss:.4f} "
                    f"Time: {epoch_time:.2f}s "
                    f"LR: {optimizer.lr:.6f}")

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            J2J_loss = 0.0

            outputs_val = []
            outputs_val_gt = []
            outputs_val_ver = []
            paths = []
            for batch_idx, data in enumerate(val_loader):
                model.eval()
                # Get data and labels
                vertices, joints, cls, id, origin_vertices, N = data['vertices'], data['joints'], data['cls'], data[
                    'id'], data['origin_vertices'], data['N']
                joints = joints.reshape(joints.shape[0], -1)

                # Reshape input if needed
                if vertices.ndim == 3:  # [B, N, 3]
                    vertices = vertices.permute(0, 2, 1)  # [B, 3, N]

                # Forward pass
                outputs = model(vertices)
                # val_loss += loss.item()
                for i in range(outputs.shape[0]):
                    J2J_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item()

            # Calculate validation statistics
            val_loss /= len(val_loader.paths)
            J2J_loss /= len(val_loader.paths)

            log_message(f"Validation Loss: {val_loss:.5f} J2J Loss: {J2J_loss:.5f}")

            # Save best model
            # if J2J_loss < best_loss and epoch +1 >= 300:
            if J2J_loss < best_loss:  # best 0.0176
                # save test data
                print("start predicting test...")
                for batch_idx, data in enumerate(predict_loader):
                    vertices, cls, id, origin_vertices, N = data['vertices'], data['cls'], data['id'], data[
                        'origin_vertices'], data['N']
                    # Reshape input if needed
                    if vertices.ndim == 3:  # [B, N, 3]
                        vertices = vertices.permute(0, 2, 1)  # [B, 3, N]
                    B = vertices.shape[0]
                    outputs = model(vertices)

                    outputs = outputs.reshape(B, -1, 3)
                    for i in range(len(cls)):
                        path = os.path.join(predict_output_dir, "epoch_{}".format(epoch + 1), 'test', cls[i],
                                            str(id[i].item()))
                        os.makedirs(path, exist_ok=True)
                        np.save(os.path.join(path, "predict_skeleton"), outputs[i])
                        np.save(os.path.join(path, "transformed_vertices"), origin_vertices[i, :N[i]].numpy())
                print("finished")
                ###################################################
                # save best model
            if J2J_loss < best_loss:
                # if J2J_loss < best_loss:
                best_loss = J2J_loss
                log_message(f"Saved best model with loss {best_loss:.5f}")
                model_path = os.path.join(args.output_dir, 'best_model_{}.pkl'.format(epoch + 1))
                model.save(model_path)
                best_model_path = os.path.join(args.output_dir, 'best_model.pkl'.format(epoch + 1))
                model.save(best_model_path)
                # if J2J_loss < 0.02:
                #     model_path = os.path.join(args.output_dir, 'best_model_{}.pkl'.format(epoch+1))
                #     model.save(model_path)
                # log_message(f"Saved best model with loss {best_loss:.5f} to {model_path}")
                ########################################################################################

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")

    return model, best_loss


def main():
    """Parse arguments and start training"""

    output_path = 'skeleton_pcae_track_1024_m_1e-5_10000'

    # data_name = 'mixamo' # 'vroid' , 'mixamo' , ''  ,   ''  is two dataset
    data_name = 'mixamo'
    random_pose = 1

    parser = argparse.ArgumentParser(description='Train a point cloud model')

    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, required=False, default='data/train_list.txt',
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='data/val_list.txt',
                        help='Path to the validation data list file')
    parser.add_argument('--predict_data_list', type=str, default='data/test_list.txt',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    parser.add_argument('--predict_output_dir', type=str, default=output_path,
                        help='Path to the validation data list file')
    # Model parameters
    parser.add_argument('--model_name', type=str, default='PCAE',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton', 'PCAE'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--random_pose', type=int, default=random_pose,
                        help='Apply random pose to asset')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=output_path,
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=30,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
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