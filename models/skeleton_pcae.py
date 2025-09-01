import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
from models.PCAE_model import PCAE
# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group

class SimpleSkeletonModel(nn.Module):
    
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)
class PCAE_model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = PCAE( 
            N = 1024,
            input_normal=False,
            input_attention=False,
            num_latents=512,
            deterministic=True,
            hierarchical_ratio=0.0,
            output_dim=52,
            output_actvn='softmax',
            output_log="l1" == "kl",
            kinematic_tree=None,
            tune_decoder_self_attn=True,
            tune_decoder_cross_attn=True,
            predict_bw=False,
            predict_joints=True,
            predict_joints_tail=False,
            joints_attn=False,
            joints_attn_masked=True,
            joints_attn_causal=False,
            predict_global_trans=False,
            predict_pose_trans=False,
            pose_mode='ortho6d',
            pose_input_joints=True,
            pose_attn=False,
            pose_attn_masked=True,
            pose_attn_causal=False,
            )
    def execute(self, vertices: jt.Var):
        vertices = vertices.permute(0,2,1)
        x = self.model(vertices)
        return x.joints

# Factory function to create models
def create_model(model_name='pct', output_channels=52*3, **kwargs):
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256, output_channels=output_channels)
    elif model_name == 'PCAE':
        return PCAE_model()
    raise NotImplementedError()
