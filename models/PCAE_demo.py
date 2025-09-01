from jittor import init
from models.PCAE_model import PCAE
import jittor as jt



if __name__ == '__main__':
    jt.flags.use_cuda = 1
    input_points = init.gauss((16, 2048, 3), dtype='float32')  # B, D, N

    model = PCAE(
        N=2048,
        input_normal=False,
        input_attention=False,
        num_latents=512,
        deterministic=True,
        hierarchical_ratio=0.0,
        output_dim=22,
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
    joints = model(input_points).joints
    print(joints.shape)
    print(joints)