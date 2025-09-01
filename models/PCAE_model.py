from jittor import nn
import jittor as jt
from models.utils import Attention, create_autoencoder,find_ckpt
import warnings
import numpy as np
from typing import NamedTuple


from PCT.misc.ops import FurthestPointSampler as fps

Output = NamedTuple(
    "Output",
    [("bw", jt.Var), ("joints", jt.Var), ("global_trans", jt.Var), ("pose_trans", jt.Var)],
)
class TransformMLP(nn.Module):
    def __init__(self, in_dim: int, transl_dim=3, rotation_dim=4, scaling_dim=3):
        super().__init__()
        self.transl_dim = transl_dim
        self.rotation_dim = rotation_dim
        self.scaling_dim = scaling_dim

        # 平移分支
        if transl_dim > 0:
            self.transl_mlp = nn.Linear(in_dim, transl_dim)

        # 旋转分支
        if rotation_dim > 0:
            if rotation_dim == 4:  # 四元数表示
                self.rotation_mlp_scalar = nn.Linear(in_dim, 1)
                self.rotation_mlp_vector = nn.Linear(in_dim, rotation_dim - 1)
            else:
                self.rotation_mlp = nn.Linear(in_dim, rotation_dim)

        # 缩放分支
        if scaling_dim > 0:
            self.scaling_mlp = nn.Linear(in_dim, scaling_dim)

    @staticmethod
    def normalize(tensor, dim=-1, eps=1e-8):
        """自定义归一化函数，替代 PyTorch 的 F.normalize"""
        norm = jt.sqrt(jt.sum(tensor.square(), dim=dim, keepdims=True))
        return tensor / jt.maximum(norm, eps)

    def execute(self, feat: jt.Var):
        # 创建空张量作为占位符
        empty = jt.empty(feat.shape[0], feat.shape[1], 0, dtype=feat.dtype)

        # 处理平移分支
        transl = self.transl_mlp(feat) if self.transl_dim > 0 else empty

        # 处理旋转分支
        if self.rotation_dim > 0:
            if self.rotation_dim == 4:
                # 使用指数函数确保标量部分为正
                rotation_scalar = jt.exp(self.rotation_mlp_scalar(feat))
                rotation_vector = self.rotation_mlp_vector(feat)
                rotation = jt.concat([rotation_scalar, rotation_vector], dim=-1)
                # 对四元数进行归一化
                rotation = self.normalize(rotation, dim=-1)
            else:
                rotation = self.rotation_mlp(feat)
                # 如果是旋转向量 (3维) 也需要归一化
                if self.rotation_dim == 3:
                    rotation = self.normalize(rotation, dim=-1)
        else:
            rotation = empty

        # 处理缩放分支
        scaling = jt.exp(self.scaling_mlp(feat)) if self.scaling_dim > 0 else empty

        # 合并所有变换参数
        return jt.concat([transl, rotation, scaling], dim=-1)


class JointsAttention(nn.Module):
    def __init__(self, feat_dim: int, heads=8, dim_head=64, masked=True, kinematic_tree=None, *args, **kwargs):
        super().__init__()

        self.norm_pre = nn.LayerNorm(feat_dim)
        self.attn = Attention(query_dim=feat_dim, heads=heads, dim_head=dim_head, *args, **kwargs)
        self.norm_after = nn.LayerNorm(feat_dim)

        self.mask = None
        if masked:
            assert kinematic_tree is not None, "需要 kinematic_tree 来创建掩码"

            # 创建布尔掩码张量 [N, N]
            mask = jt.zeros((len(kinematic_tree), len(kinematic_tree)), dtype=jt.bool)

            # 根据关节层级关系填充掩码
            for joint in kinematic_tree:
                mask[joint.index, joint.index] = True
                for parent in joint.parent_recursive:
                    mask[joint.index, parent.index] = True

            # 在 Jittor 中作为不可训练状态存储
            self.mask = mask
            self.mask.stop_grad()  # 防止梯度传播
            self.mask.require_grad = False

    def execute(self, feat: jt.Var):
        """
        Args:
            feat: [B, N, D]
        Returns:
            [B, N, D]
        """
        # 准备掩码（如果需要）
        attn_mask = None
        if self.mask is not None:
            # 扩展掩码到批处理维度 [B, N, N]
            attn_mask = self.mask.broadcast([feat.shape[0], 1, 1])

        # 前向传播流程
        norm_feat = self.norm_pre(feat)
        attn_output = self.attn(norm_feat, mask=attn_mask)

        # 残差连接和归一化
        out = attn_output + feat
        out = self.norm_after(out)
        return out

class Embedder3D(nn.Module):
    def __init__(self, dim=48, concat_input=True):
        super().__init__()

        assert dim % 6 == 0, "dim must be divisible by 6"
        self.embedding_dim = dim

        # 创建基础频率向量 [dim//6]
        e = jt.float32(2).pow(jt.arange(self.embedding_dim // 6, dtype=jt.float32)) * jt.float32(jt.pi)

        # 构建三维基础矩阵 [3, dim]
        self.basis = jt.concat([
            jt.concat([e, jt.zeros(self.embedding_dim // 6), jt.zeros(self.embedding_dim // 6)]),
            jt.concat([jt.zeros(self.embedding_dim // 6), e, jt.zeros(self.embedding_dim // 6)]),
            jt.concat([jt.zeros(self.embedding_dim // 6), jt.zeros(self.embedding_dim // 6), e])
        ]).reshape(3, -1)

        # 注册为不可训练的缓冲区
        self.basis.stop_grad()  # 阻止梯度计算
        self.concat_input = concat_input

    def embed(self, xyz: jt.Var):
        """
        Args:
            xyz: [B, N, 3]
        Returns:
            [B, N, dim]
        """
        # 使用 matmul 替代 einsum: bne = bnd @ de
        projections = jt.matmul(xyz, self.basis)  # [B, N, dim]
        return jt.concat([projections.sin(), projections.cos()], dim=-1)

    def execute(self, xyz: jt.Var):
        """
        Args:
            xyz: [B, N, 3]
        Returns:
            [B, N, dim (+3)]
        """
        embeddings = self.embed(xyz)
        if self.concat_input:
            embeddings = jt.concat([embeddings, xyz], dim=-1)
        return embeddings


class JointsEmbedder(nn.Module):
    def __init__(self, include_tail=False, embed_dim=48, out_dim=512, concat_input=True, out_mlp=True):
        super().__init__()
        self.embed = Embedder3D(embed_dim, concat_input=concat_input)
        self.point_num = 2 if include_tail else 1
        self.embedding_dim = self.point_num * (embed_dim + (3 if concat_input else 0))
        self.out_mlp = out_mlp
        if self.out_mlp:
            self.mlp = nn.Linear(self.embedding_dim, out_dim)

    def forward(self, joints):
        """
        Args:
            joints: [B, N, D]
        Returns:
            [B, N, `out_dim` if `out_mlp` else `embedding_dim`]
        """
        B, N, D = joints.shape
        assert D == self.point_num * 3
        out = self.embed(joints.view(B, N * self.point_num, 3)).view(B, N, self.embedding_dim)
        return self.mlp(out) if self.out_mlp else out


class PCAE(nn.Module):
    def __init__(
        self,
        N=1024,
        input_normal=False,
        input_attention=False,
        num_latents=512,
        deterministic=True,
        hierarchical_ratio=0.0,
        output_dim=52,
        output_actvn="softmax",
        output_log=False,
        kinematic_tree=None,
        tune_decoder_self_attn=True,
        tune_decoder_cross_attn=True,
        predict_bw=True,
        predict_joints=False,
        predict_joints_tail=False,
        joints_attn=False,
        joints_attn_masked=True,
        joints_attn_causal=False,
        predict_global_trans=False,
        predict_pose_trans=False,
        pose_mode="dual_quat",
        pose_input_joints=False,
        pose_attn=False,
        pose_attn_masked=True,
        pose_attn_causal=False,
        grid_density=128,
    ):
        super().__init__()

        self.N = N
        self.base = create_autoencoder(dim=512, M=num_latents, N=self.N, latent_dim=8, deterministic=deterministic)
        embed_dim = self.base.point_embed.mlp.out_features
        feat_dim = self.base.decoder_cross_attn.fn.to_out.out_features

        self.input_dims = [3]
        self.input_normal = input_normal
        if self.input_normal:
            self.input_dims.append(3)
            self.normal_embed = JointsEmbedder(out_dim=embed_dim)
            nn.init.zeros_(self.normal_embed.mlp.weight)
            nn.init.zeros_(self.normal_embed.mlp.bias)
        else:
            self.input_dims.append(0)
        self.input_dim = sum(self.input_dims)
        if self.input_dim == self.input_dims[0]:
            self.input_attention = False
        else:
            self.input_attention = input_attention


        self.hierarchical_ratio = float(hierarchical_ratio)
        assert 0 <= hierarchical_ratio < 1.0, f"{hierarchical_ratio=} must be in [0, 1)"

        self.output_dim = output_dim

        self.predict_bw = predict_bw
        if self.predict_bw:
            self.bw_head = nn.Linear(feat_dim, self.output_dim)
            output_actvn = output_actvn.lower()
            if output_actvn == "softmax":
                # self.actvn = nn.LogSoftmax(dim=-1) if self.output_actvn_log else nn.Sigmoid(dim=-1)
                self.actvn = nn.Softmax(dim=-1)
            elif output_actvn == "sigmoid":
                # self.actvn = nn.LogSigmoid() if self.output_actvn_log else nn.Sigmoid()
                self.actvn = nn.Sigmoid()
            elif output_actvn == "relu":
                self.actvn = nn.ReLU()
            elif output_actvn == "softplus":
                self.actvn = nn.Softplus()
            else:
                raise ValueError(f"Invalid activation: {output_actvn}")
            self.output_actvn_log = output_log

        self.tune_decoder_self_attn = tune_decoder_self_attn
        self.tune_decoder_cross_attn = tune_decoder_cross_attn

        self.predict_joints = predict_joints
        self.predict_joints_tail = predict_joints_tail
        if self.predict_joints:
            self.joints_embed = nn.Parameter(jt.randn(1, self.output_dim, embed_dim))
            joints_dim = 6 if self.predict_joints_tail else 3
            self.joints_head = nn.Linear(feat_dim, joints_dim)
            self.joints_attn_causal = joints_attn_causal
        self.predict_global_trans = predict_global_trans
        self.predict_pose_trans = predict_pose_trans
        assert pose_mode in (
            "transl_quat",  # 3 + 4
            "dual_quat",  # 4 + 4
            "transl_ortho6d",  # 3 + 6
            "target_quat",  # 3 + 4
            "target_ortho6d",  # 3 + 6
            "quat",  # 4
            "ortho6d",  # 6
            "local_quat",  # 4
            "local_ortho6d",  # 6
        ), f"Invalid {pose_mode=}"
        self.pose_mode = pose_mode
        self.pose_input_joints = pose_input_joints
        self.grid_density = grid_density
        self.grid = None

    def adapt_ckpt(self, ckpt: dict[str, jt.Var]):
        def access_attr(obj, attr: str):
            for k in attr.split("."):
                obj = getattr(obj, k)
            return obj

        params2replace = []
        if self.predict_bw:
            params2replace.extend(["bw_head.weight", "bw_head.bias"])
        for k in params2replace:
            if k in ckpt:
                ckpt_param = ckpt[k]
                model_param = access_attr(self, k)
                if ckpt_param.shape != model_param.shape:
                    print(
                        f"Size mismatch for {k}: {ckpt_param.shape} from checkpoint vs {model_param.shape} from model. Ignoring it."
                    )
                    ckpt[k] = model_param.to(ckpt_param)

        params2remove = [
            "normal_embed.embed.basis",
            "joints_embedder.embed.basis",
            "joints_head.encoder.embed.basis",
            "joints_head.0.mask",
            "pose_head.0.mask",
        ]
        params2remove.extend(
            [
                f"{l}.{m}"
                for l in ("joints_head", "pose_head")
                for m in ("mask_attn", "mask_parent", "tree_levels_mask", "embed.basis")
            ]
        )
        for k in params2remove:
            if k in ckpt:
                print(f"Removing deprecated params {k} from checkpoint.")
                del ckpt[k]

        params2partial = []
        if self.predict_joints:
            params2partial.append("joints_embed")
        if self.predict_pose_trans:
            params2partial.append("pose_embed")
        for k in params2partial:
            if k in ckpt:
                ckpt_param = ckpt[k]
                model_param = access_attr(self, k)
                if ckpt_param.shape != model_param.shape:
                    assert ckpt_param.shape[0] == model_param.shape[0] and ckpt_param.shape[-1] == model_param.shape[-1]
                    print(
                        f"Size mismatch for {k}: {ckpt_param.shape} from checkpoint vs {model_param.shape} from model. Partially loading it."
                    )
                    if ckpt_param.shape[1] < model_param.shape[1]:
                        ckpt_param_new = model_param.clone().detach()
                        ckpt_param_new[:, : ckpt_param.shape[1]] = ckpt_param
                        ckpt[k] = ckpt_param_new.to(ckpt_param)
                    else:
                        ckpt[k] = ckpt_param[:, : model_param.shape[1]]

        return ckpt

    def load(self, pth_path: str, epoch=-1, strict=True, adapt=True):
        pth_path = find_ckpt(pth_path, epoch=epoch)
        checkpoint = jt.load(pth_path, map_location="cpu")
        model_state_dict = checkpoint["model"]
        if adapt:
            model_state_dict = self.adapt_ckpt(model_state_dict)
        self.load_state_dict(model_state_dict, strict=strict)
        print(f"Loaded model from {pth_path}")
        return self

    def load_base(self, pth_path: str):
        self.base.load_state_dict(jt.load(pth_path, map_location="cpu")["model"], strict=True)
        print(f"Loaded base model from {pth_path}")
        return self

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False
        tune_module_list = []
        if self.tune_decoder_self_attn:
            tune_module_list.append(self.base.layers)
        if self.tune_decoder_cross_attn:
            tune_module_list.append(self.base.decoder_cross_attn)
            if self.base.decoder_ff is not None:
                tune_module_list.append(self.base.decoder_ff)
        for module in tune_module_list:
            for param in module.parameters():
                param.requires_grad = True
        return self

    def fps(self, pc: jt.Var) -> jt.Var:
        """
        Args:
            pc: [B, `self.N`, D]
        Returns:
            [B, `self.base.num_latents`, D]
        """
        B, N, D = pc.shape
        assert N == self.base.num_inputs
        assert D == self.input_dim

        # flattened = pc.view(B * N, D)
        # batch = torch.arange(B).to(pc.device)
        # batch = torch.repeat_interleave(batch, N)
        # pos = flattened
        # ratio = 1.0 * self.base.num_latents / N
        # idx = fps(pos, batch, ratio=ratio)
        # sampled_pc = pos[idx]
        # sampled_pc = sampled_pc.view(B, -1, 3)

        N_hier = int(N * self.hierarchical_ratio)
        N_pc = [N - N_hier, N_hier]
        num_latents_hier = int(self.base.num_latents * self.hierarchical_ratio)
        N_latents = [self.base.num_latents - num_latents_hier, num_latents_hier]
        latents_begin_idx = [0, self.base.num_latents - num_latents_hier]

        sampled_pc = jt.empty(B, self.base.num_latents, D, dtype=pc.dtype)

        sampler = fps(512)
          # [B, npoint]

        for i, pc_ in enumerate(jt.split(pc, N_pc, dim=1)):
            N_ = pc_.shape[1]
            if N_ == 0:
                continue
            pos = pc_.reshape(-1, D)
            _, idx = sampler(pc_)
            sampled_pc[:, latents_begin_idx[i] : latents_begin_idx[i] + N_latents[i], :] = pos[idx].view(B, -1, D)[
                :, : N_latents[i], :
            ]
        # import trimesh; trimesh.Trimesh(sampled_pc[0].cpu()).export("sample.ply")  # fmt: skip
        # although 32768 so larger, but fps only sample 512 for encode
        return sampled_pc

    def embed(self, pc: jt.Var) -> jt.Var:
        """
        Args:
            pc: [B, N, D]
        Returns:
            [B, N, 512]
        """
        extra_embeddings = []
        if self.input_dim > 3:
            pc, normal = jt.split(pc, self.input_dims, dim=-1)
            if normal.shape[-1] > 0:
                extra_embeddings.append(self.normal_embed(normal))

        pc_embeddings = self.base.point_embed(pc)

        if extra_embeddings:
            if self.input_attention:
                pc_embeddings = self.input_attn(pc_embeddings, jt.stack(extra_embeddings, dim=-2))
            else:
                pc_embeddings = pc_embeddings + jt.stack(extra_embeddings, dim=0).sum(0)

        return pc_embeddings

    def encode(self, pc: jt.Var) -> jt.Var:
        """
        Args:
            pc: [B, `self.N`, 3]
        Returns:
            [B, 512, 512]
        """
        # _, x = self.base.encode(pc)

        sampled_pc = self.fps(pc)
        sampled_pc_embeddings = self.embed(sampled_pc)  # mlp
        pc_embeddings = self.embed(pc)
        cross_attn, cross_ff = self.base.cross_attend_blocks
        x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        return x

    def decode(self, x: jt.Var, queries: jt.Var, learnable_embeddings: jt.Var = None) -> jt.Var:
        """
        Args:
            x: [B, 512, 3]
            queries: [B, N, 3]
        Returns:
            [B, N, 512]
        """
        # o = self.base.decode(x, queries)
        if hasattr(self.base, "proj"):
            x = self.base.proj(x)
        for self_attn, self_ff in self.base.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        # cross attend from decoder queries to latents
        queries_embeddings = self.embed(queries)
        if learnable_embeddings is not None:
            queries_embeddings = jt.concat((queries_embeddings, learnable_embeddings), dim=1)
        latents = self.base.decoder_cross_attn(queries_embeddings, context=x)
        # optional decoder feedforward
        if self.base.decoder_ff is not None:
            latents = latents + self.base.decoder_ff(latents)
        return latents

    def forward_base(self, pc: jt.Var, queries: jt.Var) -> jt.Var:
        """encode + decode + occupancy mlp
        Args:
            pc: [B, `self.N`, 3]
            queries: [B, N2, 3]
        Returns:
            [B, N2, 1]
        """
        if self.tune_decoder_cross_attn:
            warnings.warn(
                "Decoder cross-attn layers are tuned, so the output of occupancy MLP is not reliable anymore."
            )
        return self.base.forward(pc, queries)["logits"].unsqueeze(-1)

    def execute(
        self, pc: jt.Var, queries: jt.Var = None, joints: jt.Var = None, pose: jt.Var = None
    ):
        """
        Args:
            pc: [B, `self.N`, 3]
            queries: [B, N2, 3]
        Returns:
            [B, N2, `self.output_dim`]
        """
        if pc.shape[-1] > self.input_dim:
            pc = pc[..., : self.input_dim]
        x = self.encode(pc) # 1 512 512

        learnable_embeddings = (
            self.joints_embed if self.predict_joints else None,
            self.global_embed if self.predict_global_trans else None,
            self.pose_embed if self.predict_pose_trans else None,
        )
        learnable_embeddings_length = [0 if x is None else x.shape[1] for x in learnable_embeddings]
        learnable_embeddings = [x for x in learnable_embeddings if x is not None]
        if learnable_embeddings:
            learnable_embeddings = jt.concat(learnable_embeddings, dim=1)
            learnable_embeddings = learnable_embeddings.expand(pc.shape[0], -1, -1).clone()
        else:
            learnable_embeddings = None
        if queries is None:
            assert not self.predict_bw and learnable_embeddings is not None, "Nothing to predict"
            queries = jt.empty((pc.shape[0], 0, pc.shape[2]), dtype=pc.dtype)
        elif queries.shape[-1] > self.input_dim:
            queries = queries[..., : self.input_dim]



        logits = self.decode(x, queries, learnable_embeddings) # 1 4882 512
        if learnable_embeddings is not None:
            logits, logits_joints, logits_global, logits_pose = jt.split(
                logits, [queries.shape[1]] + learnable_embeddings_length, dim=1
            )

        if self.predict_bw:
            bw: jt.Var = self.bw_head(logits)
            bw = self.actvn(bw)
            if not isinstance(self.actvn, nn.Softmax):
                bw = bw / (bw.sum(dim=-1, keepdim=True) + 1e-10)
            if self.output_actvn_log and self.training:
                bw = jt.log(bw)
        else:
            bw = None

        if self.predict_joints:
            if self.joints_attn_causal:
                joints = self.joints_head(logits_joints, out_gt=joints)
            else:
                joints = self.joints_head(logits_joints)
        else:
            joints = None

        global_trans = self.global_head(logits_global) if self.predict_global_trans else None

        if self.predict_pose_trans:
            if self.pose_attn_causal:
                pose_trans = self.pose_head(logits_pose, out_gt=pose)
            else:
                pose_trans = self.pose_head(logits_pose)
        else:
            pose_trans = None

        return Output(bw, joints, global_trans, pose_trans)

    def get_grid(self):
        if self.grid is None:
            x = np.linspace(-1, 1, self.grid_density + 1)
            y = np.linspace(-1, 1, self.grid_density + 1)
            z = np.linspace(-1, 1, self.grid_density + 1)
            xv, yv, zv = np.meshgrid(x, y, z)

            self.grid = jt.array(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1).reshape(1, 3, -1)
        return self.grid

