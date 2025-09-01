from typing import Iterator, List, Dict, Tuple, Optional, Union, Any
from jittor import nn
import jittor as jt
import os
import numpy as np
from PCT.misc.ops import FurthestPointSampler as fps

from jittor.einops import rearrange, repeat
# from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


import sys


# 获取指定数据类型的最大有限值（安全替代 inf）
def get_max_neg_value(dtype):
    # 将 Jittor 数据类型映射到 Python float 类型
    if dtype == jt.float16 or dtype == "float16":
        max_val = 65504.0  # float16 的最大有限值
    elif dtype == jt.float32 or dtype == "float32":
        max_val = sys.float_info.max
    elif dtype == jt.float64 or dtype == "float64":
        max_val = sys.float_info.max
    else:
        raise ValueError(f"不支持的 dtype: {dtype}")

    return jt.array(-max_val, dtype=dtype)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        # self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path = nn.Identity()

    def execute(self, x, context=None, mask=None, return_score=False):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)  # 展开Q的多头维度
        k = rearrange(k, 'b n (h d) -> (b h) n d', h=h)  # 展开K的多头维度
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)  # 展开V的多头维度

        sim = jt.matmul(q, k.transpose(1, 2)) * self.scale

        def jt_nan_to_num(tensor, replace_nan=0.0):
            # 创建 NaN 掩码
            nan_mask = jt.isnan(tensor)
            # 替换 NaN 为指定值，非 NaN 保留原值
            result = jt.where(nan_mask, jt.array(replace_nan), tensor)
            return result

        sim = jt_nan_to_num(sim)
        attn = nn.softmax(sim, dim=-1)

        out = jt.matmul(attn, v)


        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.drop_path(self.to_out(out))
        if return_score:
            return out, rearrange(attn, '(b h) i j -> b h i j', h=h)
        return out



def manual_matmul(input, basis):
    b, n, d = input.shape
    _, e = basis.shape
    # 合并 b 和 n 维度：[b, n, d] -> [b*n, d]
    input_reshaped = input.reshape(b * n, d)
    # 矩阵乘法：[b*n, d] @ [d, e] -> [b*n, e]
    output_reshaped = jt.matmul(input_reshaped, basis)
    # 恢复原始维度：[b*n, e] -> [b, n, e]
    output = output_reshaped.reshape(b, n, e)
    return output


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        self.dim = dim
        assert hidden_dim % 6 == 0, "hidden_dim 必须是6的倍数"

        self.embedding_dim = hidden_dim

        # 创建频域基础向量
        e = jt.array(np.power(2, np.arange(self.embedding_dim // 6)) * np.pi, dtype=jt.float32)

        e = jt.stack([
            jt.concat([e, jt.zeros(self.embedding_dim // 6),
                        jt.zeros(self.embedding_dim // 6)]),
            jt.concat([jt.zeros(self.embedding_dim // 6), e,
                        jt.zeros(self.embedding_dim // 6)]),
            jt.concat([jt.zeros(self.embedding_dim // 6),
                        jt.zeros(self.embedding_dim // 6), e]),
        ])

        # 注册为持久化参数（相当于register_buffer）
        self.basis = e

        # 创建MLP层
        self.mlp = nn.Linear(self.embedding_dim + 3, dim)  # 注意维度变化

    @staticmethod
    def embed(input, basis):
        # 计算投影: bnd @ de -> bne
        if input.shape[1] == 0:  # 第2维（点数）为0时
            projections = jt.empty((input.shape[0], 0, basis.shape[1]), dtype=input.dtype)
        else:
            projections = manual_matmul(input, basis)

        # 创建sin和cos嵌入
        sin_emb = jt.sin(projections)
        cos_emb = jt.cos(projections)

        # 拼接嵌入
        embeddings = jt.concat([sin_emb, cos_emb], dim=2)
        return embeddings

    def execute(self, input):
        # input: B x N x 3
        # 计算频率嵌入
        freq_emb = self.embed(input, self.basis)
        # 拼接原始输入和频率嵌入
        combined = jt.concat([freq_emb, input], dim=2)
        # 通过MLP

        if input.shape[1] == 0:
            embed = jt.empty((input.shape[0], 0, self.dim), dtype=input.dtype)
        else:
            embed = self.mlp(combined)  # B x N x C
        return embed



class GEGLU(nn.Module):
    def execute(self, x):
        # 将输入沿最后一个维度分割为两部分
        x_part, gates = jt.split(x, x.shape[-1] // 2, dim=-1)
        # 使用 Jittor 的 GeLU 激活函数
        return x_part * jt.nn.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
        self.drop_path = nn.Identity()
        # self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def execute(self, x):
        return self.drop_path(self.net(x))
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def execute(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)


def cache_fn(f):
    cache = None
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


class AutoEncoder(nn.Module):
    def __init__(
            self,
            *,
            depth=24,
            dim=512,
            queries_dim=512,
            output_dim=1,
            num_inputs=2048,
            num_latents=512,
            heads=8,
            dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads=1, dim_head=dim),
                                          context_dim=dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        # self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    # def encode(self, pc):
    #     # pc: B x N x 3
    #     B, N, D = pc.shape
    #     assert N == self.num_inputs
    #
    #     ###### fps
    #     flattened = pc.view(B * N, D)
    #     batch = jt.arange(B).to(pc.device)
    #
    #     # 每个索引重复 N 次
    #     batch = batch.repeat_interleave(repeats=N)
    #
    #     pos = flattened
    #
    #     ratio = 1.0 * self.num_latents / self.num_inputs
    #
    #     idx = fps(pos, batch, ratio=ratio)
    #
    #     sampled_pc = pos[idx]
    #     sampled_pc = sampled_pc.view(B, -1, 3)
    #     ######
    #
    #     sampled_pc_embeddings = self.point_embed(sampled_pc)
    #
    #     pc_embeddings = self.point_embed(pc)
    #
    #     cross_attn, cross_ff = self.cross_attend_blocks
    #
    #     x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
    #     x = cross_ff(x) + x
    #
    #     return x
    #
    # def decode(self, x, queries):
    #
    #     for self_attn, self_ff in self.layers:
    #         x = self_attn(x) + x
    #         x = self_ff(x) + x
    #
    #     # cross attend from decoder queries to latents
    #     queries_embeddings = self.point_embed(queries)
    #     latents = self.decoder_cross_attn(queries_embeddings, context=x)
    #
    #     # optional decoder feedforward
    #     if exists(self.decoder_ff):
    #         latents = latents + self.decoder_ff(latents)
    #
    #     return self.to_outputs(latents)

    # def execute(self, pc, queries):
    #     x = self.encode(pc)
    #
    #     o = self.decode(x, queries).squeeze(-1)
    #
    #     return {'logits': o}


def create_autoencoder(dim=512, M=512, latent_dim=64, N=2048, deterministic=False):
    model = AutoEncoder(
        depth=24,
        dim=dim,
        queries_dim=dim,
        output_dim = 1,
        num_inputs = N,
        num_latents = M,
        heads = 8,
        dim_head = 64,
    )

    return model

def find_ckpt(ckpt_dir: str, epoch=-1, prefix="checkpoint-", suffix=".pth"):
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(os.path.abspath(ckpt_dir))
    if os.path.isfile(ckpt_dir):
        return ckpt_dir
    elif not os.path.isdir(ckpt_dir):
        raise NotADirectoryError(os.path.abspath(ckpt_dir))
    if epoch >= 0:
        filepath = os.path.join(ckpt_dir, f"{prefix}{epoch}{suffix}")
        if os.path.isfile(filepath):
            return filepath
        else:
            raise FileNotFoundError(os.path.abspath(filepath))
    file_list = [p for p in os.listdir(ckpt_dir) if p.startswith(prefix) and p.endswith(suffix)]
    if not file_list:
        raise FileNotFoundError(os.path.abspath(ckpt_dir))
    file_list.sort(
        key=lambda x: (
            int(x.split(prefix)[-1].split(suffix)[0]) if x.split(prefix)[-1].split(suffix)[0].isdigit() else -99
        ),
        reverse=True,
    )
    return os.path.join(ckpt_dir, file_list[0])

# class Joint:
#     name: str
#     index: int
#     parent: Optional["Joint"]  # 使用字符串类型提示和 Optional
#     children: List["Joint"]
#     template_joints: Tuple[str, ...]  # 添加 ... 表示可变元组
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.name})"
#
#     def __iter__(self) -> Iterator["Joint"]:
#         yield self
#         for child in self.children:
#             yield from child
#
#     def children_recursive(self) -> List["Joint"]:
#         children_list = []
#         if not self.children:
#             return children_list
#         for child in self.children:
#             children_list.append(child)
#             children_list.extend(child.children_recursive())
#         return children_list
#
#     def __len__(self):
#         return len(self.children_recursive()) + 1
#
#     def __contains__(self, item: Union["Joint", str]) -> bool:
#         if isinstance(item, str):
#             return item == self.name or any(item == child.name for child in self.children_recursive())
#         elif isinstance(item, Joint):
#             return item is self or item in self.children_recursive()
#         else:
#             raise TypeError(f"Item must be Joint or str, not {type(item)}")
#
#     def children_recursive_dict(self) -> Dict[str, "Joint"]:
#         return {child.name: child for child in self.children_recursive()}
#
#     def __getitem__(self, index: Union[int, str]) -> "Joint":
#         if index == 0 or index == self.name:
#             return self
#
#         recursive_children = self.children_recursive()
#         if isinstance(index, int):
#             return recursive_children[index - 1]
#         elif isinstance(index, str):
#             for child in recursive_children:
#                 if child.name == index:
#                     return child
#             raise KeyError(f"Joint '{index}' not found in children")
#         else:
#             raise TypeError(f"Index must be int or str, not {type(index)}")
#
#     def parent_recursive(self) -> List["Joint"]:
#         parent_list = []
#         if self.parent is None:
#             return parent_list
#         parent_list.append(self.parent)
#         parent_list.extend(self.parent.parent_recursive())
#         return parent_list
#
#     def joints_list(self) -> List["Joint"]:
#         joints_list = [None] * (len(self.children_recursive()) + 1)
#         for joint in self:
#             if joint.index < len(joints_list):
#                 joints_list[joint.index] = joint
#         return joints_list
#
#     def parent_indices(self) -> List[int]:
#         joints = self.joints_list()
#         return [-1 if j.parent is None else j.parent.index for j in joints]
#
#     def get_first_valid_parent(self, valid_names: List[str]) -> Optional["Joint"]:
#         for parent in self.parent_recursive():
#             if parent.name in valid_names:
#                 return parent
#         return None
#
#     def tree_levels(self) -> Dict[int, List["Joint"]]:
#         levels = {0: [self]}
#         for child in self.children:
#             child_levels = child.tree_levels()
#             for level, nodes in child_levels.items():
#                 current_level = level + 1
#                 if current_level not in levels:
#                     levels[current_level] = []
#                 levels[current_level].extend(nodes)
#         return levels
#
#     def tree_levels_name(self) -> Dict[int, List[str]]:
#         levels = self.tree_levels()
#         return {level: [j.name for j in nodes] for level, nodes in levels.items()}
#
#     def tree_levels_index(self) -> Dict[int, List[int]]:
#         levels = self.tree_levels()
#         return {level: [j.index for j in nodes] for level, nodes in levels.items()}
#
#     def tree_levels_mask(self) -> List[List[bool]]:
#         levels = self.tree_levels_name()
#         max_level = max(levels.keys()) if levels else 0
#
#         masks = []
#         for level in range(max_level + 1):
#             level_names = levels.get(level, [])
#             mask = [name in level_names for name in self.template_joints]
#             masks.append(mask)
#
#         return masks