import jittor as jt
import numpy as np
import os
from jittor.dataset import Dataset
from typing import List, Dict, Callable, Union

from .asset import Asset
from .sampler import Sampler


def transform(asset: Asset):
    """
    Transform the asset data into [-1, 1]^3.
    """
    # Find min and max values for each dimension of points
    min_vals = np.min(asset.vertices, axis=0)
    max_vals = np.max(asset.vertices, axis=0)
    
    # Calculate the center of the bounding box
    center = (min_vals + max_vals) / 2
    
    # Calculate the scale factor to normalize to [-1, 1]
    # We take the maximum range across all dimensions to preserve aspect ratio
    scale = np.max(max_vals - min_vals) / 2
    
    # Normalize points to [-1, 1]^3
    normalized_vertices = (asset.vertices - center) / scale
    
    # Apply the same transformation to joints
    if asset.joints is not None:
        normalized_joints = (asset.joints - center) / scale
    else:
        normalized_joints = None
    
    asset.vertices  = normalized_vertices
    asset.joints    = normalized_joints
    # remember to change matrix_local !
    # asset.matrix_local[:, :3, 3] = normalized_joints

class RigDataset(Dataset):
    '''
    A simple dataset class.
    '''
    def __init__(
        self,
        data_root: str,
        paths: List[str],
        train: bool,
        batch_size: int,
        shuffle: bool,
        sampler: Sampler,
        transform: Union[Callable, None] = None,
        return_origin_vertices: bool = False,
        random_pose=False,
        data_name: str = ''  
    ):
        super().__init__()
        self.data_root  = data_root
        self.paths      = paths.copy()
        self.batch_size = batch_size
        self.train      = train
        self.shuffle    = shuffle
        self._sampler   = sampler  
        self.transform  = transform
        self.return_origin_vertices = return_origin_vertices
        self.random_pose = random_pose
        self.data_name = data_name  
        
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.paths),
            shuffle=self.shuffle,
        )
    
    def __getitem__(self, index) -> Dict:
        """
        Get a sample from the dataset
        """
        path = self.paths[index]
        base_path = os.path.join(self.data_root, path)
        original_asset = Asset.load(base_path)

        if self.random_pose:
            if self.train:
                total_paths = len(self.paths)
                
                transform_id = 1 + int(index / total_paths * 1512)

                if 'vroid/' in path:
                    transformed_path  = path.replace('vroid/', f'vroid-{transform_id}/')
                else:
                    transformed_path  = path.replace('mixamo/', f'mixamo-{transform_id}/')

                transformed_full_path = os.path.join(self.data_root, transformed_path)
            else:
                transformed_path = path.replace('train/', 'val/')
                transformed_full_path = os.path.join(self.data_root, transformed_path)

            transform_data = np.load(transformed_full_path, allow_pickle=True)
            asset = original_asset
            asset.vertices = transform_data['vertices']
            asset.joints = transform_data['joints']
        else:
            asset = original_asset

        if self.transform is not None:
            self.transform(asset)
        origin_vertices = jt.array(asset.vertices.copy()).float32()
        
        sampled_asset = asset.sample(sampler=self._sampler)

        vertices    = jt.array(sampled_asset.vertices).float32()
        normals     = jt.array(sampled_asset.normals).float32()

        if sampled_asset.joints is not None:
            joints      = jt.array(sampled_asset.joints).float32()
        else:
            joints      = None

        if sampled_asset.skin is not None:
            skin        = jt.array(sampled_asset.skin).float32()
        else:
            skin        = None

        res = {
            'vertices': vertices,
            'normals': normals,
            'cls': asset.cls,
            'id': asset.id,
        }
        if joints is not None:
            res['joints'] = joints
        if skin is not None:
            res['skin'] = skin
        if self.return_origin_vertices:
            res['origin_vertices'] = origin_vertices
        return res
    
    def collate_batch(self, batch):
        if self.return_origin_vertices:
            max_N = 0
            for b in batch:
                max_N = max(max_N, b['origin_vertices'].shape[0])
            for b in batch:
                N = b['origin_vertices'].shape[0]
                b['origin_vertices'] = np.pad(b['origin_vertices'], ((0, max_N-N), (0, 0)), 'constant', constant_values=0.)
                b['N'] = N
        return super().collate_batch(batch)

def get_dataloader(
    data_root: str,
    data_list: str,
    train: bool,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler,
    transform: Union[Callable, None] = None,
    return_origin_vertices: bool = False,
    data_name: str = '',
    random_pose=0
):
    """
    Create a dataloader for point cloud data
    """
    with open(data_list, 'r') as f:
        if os.path.basename(data_list) == 'test_list.txt':
            paths = [line.strip() for line in f.readlines() if line.__contains__(data_name) and len(line.strip().split('/')[-1].split('.')[0]) > 4]
        else:
            paths = [line.strip() for line in f.readlines() if line.__contains__(data_name)]

    dataset = RigDataset(
        data_root=data_root,
        paths=paths,
        train=train,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=return_origin_vertices,
        random_pose=random_pose,
        data_name=data_name  
    )
    
    return dataset
