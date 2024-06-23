# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import math
from typing import Tuple, Type

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        r: int = 0,
        lora_scale: float = 1.0,
        num_lora: int = 0,
        enable_dora: bool = False
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    r=r,
                    lora_scale=lora_scale,
                    num_lora=num_lora,
                    enable_dora=enable_dora
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, r=r, lora_scale=lora_scale, num_lora=num_lora, enable_dora=enable_dora
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attenion layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        r: int = 0,
        lora_scale: float = 1.0,
        num_lora: int = 0,
        enable_dora: bool = False
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads, r=r, lora_scale=lora_scale, num_lora=num_lora, enable_dora=enable_dora)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, r=r, lora_scale=lora_scale, num_lora=num_lora, enable_dora=enable_dora
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, r=r, lora_scale=lora_scale, num_lora=num_lora, enable_dora=enable_dora
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        r: int = 0,
        lora_scale: float = 1.0,
        num_lora: int = 0,
        enable_dora: bool = False
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        # DoRA
        self.enable_dora = enable_dora
        if self.enable_dora:
            self.dora_m_q = nn.Parameter(self.q_proj.weight.norm(p=2, dim=0, keepdim=True))
            self.dora_m_v = nn.Parameter(self.v_proj.weight.norm(p=2, dim=0, keepdim=True))
            
        # LoRA
        self.lora_scale = num_lora if num_lora == 0 else [lora_scale] if num_lora == 1 else nn.Parameter(torch.randn(num_lora))
        self.num_lora = num_lora
        if num_lora > 0:
            self.lora_w_a_q = nn.ModuleList()
            self.lora_w_b_q = nn.ModuleList()
            self.lora_w_a_v = nn.ModuleList()
            self.lora_w_b_v = nn.ModuleList()
            for i in range(num_lora):
                self.lora_w_a_q.append(nn.Linear(embedding_dim, r, bias=False))
                self.lora_w_b_q.append(nn.Linear(r, self.internal_dim, bias=False))
                self.lora_w_a_v.append(nn.Linear(embedding_dim, r, bias=False))
                self.lora_w_b_v.append(nn.Linear(r, self.internal_dim, bias=False))
        elif num_lora < 0:
            raise ValueError(f"num_lora must be non-negative, got {num_lora}")

        # self.lora_scale = lora_scale
        # self.lora_w_a_q = nn.Linear(embedding_dim, r, bias=False)
        # self.lora_w_b_q = nn.Linear(r, self.internal_dim, bias=False)
        # self.lora_w_a_v = nn.Linear(embedding_dim, r, bias=False)
        # self.lora_w_b_v = nn.Linear(r, self.internal_dim, bias=False)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # DoRA
        alphas = self.lora_scale if self.num_lora <= 1 else F.softmax(torch.sigmoid(self.lora_scale), dim=0)
        # breakpoint()
        if self.enable_dora:
            # numerator_q = self.q_proj(q) + self.lora_w_b_q[0](self.lora_w_a_q[0](q)) * alphas[0]
            numerator_q = self.q_proj.weight + (self.lora_w_b_q[0].weight @ self.lora_w_a_q[0].weight) * alphas[0]
            denominator_q = numerator_q.norm(p=2, dim=0, keepdim=True)
            weight_q = self.dora_m_q * (numerator_q / denominator_q)
            q = F.linear(q, weight_q, self.q_proj.bias)
            k = self.k_proj(k)
            # numerator_v = self.v_proj(v) + self.lora_w_b_v[0](self.lora_w_a_v[0](v)) * alphas[0]
            numerator_v = self.v_proj.weight + (self.lora_w_b_v[0].weight @ self.lora_w_a_v[0].weight) * alphas[0]
            denominator_v = numerator_v.norm(p=2, dim=0, keepdim=True)
            weight_v = self.dora_m_v * (numerator_v / denominator_v)
            v = F.linear(v, weight_v, self.v_proj.bias)
        else:
            # Input projections for LoRA
            new_q = 0
            new_v = 0
            if self.num_lora >= 1:
                for i in range(self.num_lora):
                    new_q += self.lora_w_b_q[i](self.lora_w_a_q[i](q)) * alphas[i]
                    new_v += self.lora_w_b_v[i](self.lora_w_a_v[i](v)) * alphas[i]
            q = self.q_proj(q) + new_q
            k = self.k_proj(k)
            v = self.v_proj(v) + new_v
            # q = self.q_proj(q) + self.lora_w_b_q(self.lora_w_a_q(q)) * self.lora_scale
            # k = self.k_proj(k)
            # v = self.v_proj(v) + self.lora_w_b_v(self.lora_w_a_v(v)) * self.lora_scale

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
