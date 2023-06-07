import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PatchEmbed(layers.Layer):
    def __init__(self, patch_size=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = (patch_size, patch_size)
        self.norm = norm_layer(epsilon=1e-6)
        self.project = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="SAME",
            kernel_initializer=tf.keras.initializers.LecunNormal(),
        )

    def call(self, x, **kwargs):
        _, H, W, _ = x.shape

        # padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            paddings = tf.constant(
                [
                    [0, 0],
                    [0, self.patch_size[0] - H % self.patch_size[0]],
                    [0, self.patch_size[1] - W % self.patch_size[1]],
                ]
            )
            x = tf.pad(x, paddings)

        x = self.project(x)
        _, H, W, C = x.shape
        # [B, H, W, C] -> [B, H * W, C]
        x = tf.reshape(x, [-1, H * W, C])
        x = self.norm(x)
        return x, H, W


class ShiftedPatchTokenization(layers.Layer):
    def __init__(self, patch_size=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.norm = norm_layer(epsilon=1e-6)
        self.project = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="SAME",
            kernel_initializer=tf.keras.initializers.LecunNormal(),
        )
        self.to_patch_tokens = tf.keras.Sequential(
            [
                # PatchEmbedding(image_size, patch_size, dimension),
                # layers.LayerNormalization(),
                layers.Dense(units=embed_dim),
            ]
        )

    def shift(self, x):
        # _, height, width, channel = x.shape
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channel = tf.shape(x)[3]

        shifted_x = []
        shifts = [1, -1]

        # width
        z = tf.zeros([batch_size, height, 1, channel], dtype=tf.float32)
        for index, shift in enumerate(shifts):
            if index == 0:
                s = tf.roll(x, shift, axis=2)[:, :, shift:, :]
                concat = tf.concat([z, s], axis=2)
            else:
                s = tf.roll(x, shift, axis=2)[:, :, :shift, :]
                concat = tf.concat([s, z], axis=2)
            shifted_x.append(concat)

        # height
        z = tf.zeros([batch_size, 1, width, channel], dtype=tf.float32)
        for index, shift in enumerate(shifts):
            if index == 0:
                s = tf.roll(x, shift, axis=1)[:, shift:, :, :]
                concat = tf.concat([z, s], axis=1)
            else:
                s = tf.roll(x, shift, axis=1)[:, :shift, :, :]
                concat = tf.concat([s, z], axis=1)
            shifted_x.append(concat)

        return shifted_x

    def call(self, x):
        _, H, W, _ = x.shape

        # padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            paddings = tf.constant(
                [
                    [0, 0],
                    [0, self.patch_size[0] - H % self.patch_size[0]],
                    [0, self.patch_size[1] - W % self.patch_size[1]],
                ]
            )
            x = tf.pad(x, paddings)

        shifted_x = self.shift(x)
        x_with_shifts = tf.concat([x, *shifted_x], axis=-1)

        x = self.project(x)
        _, H, W, C = x.shape
        # [B, H, W, C] -> [B, H * W, C]
        x = tf.reshape(x, [-1, H * W, C])
        x = self.norm(x)

        x = self.to_patch_tokens(x)

        return x, H, W


class PatchMerging(layers.Layer):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.reduction = layers.Dense(
            2 * dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )

    def call(self, x, H, W):
        """
        x shape: [B, H*W, C]
        """
        _, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = tf.reshape(x, [-1, H, W, C])
        # padding
        pad_input = (H % 2 != 0) or (W % 2 != 0)
        if pad_input:
            paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
            x = tf.pad(x, paddings)
            H = H + 1
            W = W + 1

        x0 = x[:, 0::2, 0::2, :]  # [B, H / 2, W / 2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H / 2, W / 2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H / 2, W / 2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H / 2, W / 2, C]
        x = tf.concat([x0, x1, x2, x3], -1)  # [B, H / 2, W / 2, 4*C]
        x = tf.reshape(x, [-1, H // 2 * W // 2, 4 * C])  # [B, H / 2 * W / 2, 4 * C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H / 2 * W / 2, 2 * C]
        return x


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    _, H, W, C = x.shape
    x = tf.reshape(
        x, [-1, H // window_size, window_size, W // window_size, window_size, C]
    )
    # transpose: [B, H // Mh, Mh, W // Mw, Mw, C] -> [B, H // Mh, W // Mh, Mw, Mw, C]
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    # reshape: [B, H // Mh, W // Mw, Mh, Mw, C] -> [B * num_windows, Mh, Mw, C]
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[-1]
    # reshape: [B * num_windows, Mh, Mw, C] -> [B, H // Mh, W // Mw, Mh, Mw, C]
    x = tf.reshape(
        windows,
        [
            -1,
            H // window_size,
            W // window_size,
            window_size,
            window_size,
            C,
        ],
    )
    # permute: [B, H // Mh, W // Mw, Mh, Mw, C] -> [B, H // Mh, Mh, W // Mw, Mw, C]
    # reshape: [B, H // Mh, Mh, W // Mw, Mw, C] -> [B, H, W, C]
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, H, W, C])
    return x


class WindowAttention(layers.Layer):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop_ratio (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop_ratio (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads=8,
        qkv_bias=False,
        attn_drop_ratio=0.0,
        proj_drop_ratio=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Locality Self-Attention
        self.temperature = tf.Variable(tf.math.log(head_dim**-0.5))
        self.softmax = layers.Softmax()

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=qkv_bias,
        )
        self.attn_drop = layers.Dropout(attn_drop_ratio)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop_ratio)

    def build(self, input_shape):
        # define a parameter table of relative position bias
        # [2 * Mh - 1 * 2 * Mw - 1, nH]
        self.relative_position_bias_table = self.add_weight(
            shape=[
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=tf.float32,
        )

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = np.reshape(coords, [2, -1])  # [2, Mh * Mw]
        # [2, Mh * Mw, 1] - [2, 1, Mh * Mw]
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, Mh * Mw, Mh * Mw]
        relative_coords = np.transpose(
            relative_coords, [1, 2, 0]
        )  # [Mh * Mw, Mh * Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh * Mw, Mh * Mw]

        self.relative_position_index = tf.Variable(
            tf.convert_to_tensor(relative_position_index),
            trainable=False,
            # dtype=tf.int32,
            dtype=tf.int64,
        )

    def call(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            training: whether training mode
        """
        # [batch_size * num_windows, Mh * Mw, total_embed_dim]
        _, N, C = x.shape

        # qkv(): -> [batch_size * num_windows, Mh * Mw, 3 * total_embed_dim]
        qkv = self.qkv(x)
        # reshape: -> [batch_size * num_windows, Mh * Mw, 3, num_heads, embed_dim_per_head]
        qkv = tf.reshape(qkv, [-1, N, 3, self.num_heads, C // self.num_heads])
        # transpose: -> [3, batch_size * num_windows, num_heads, Mh * Mw, embed_dim_per_head]
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        # [batch_size * num_windows, num_heads, Mh * Mw, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size * num_windows, num_heads, embed_dim_per_head, Mh * Mw]
        # multiply -> [batch_size * num_windows, num_heads, Mh * Mw, Mh * Mw]
        # attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale

        # Locality Self-Attention
        attn = tf.matmul(a=q, b=k, transpose_b=True) * tf.math.exp(self.temperature)
        lsa_mask = tf.eye(attn.shape[-1], dtype=tf.bool)
        lsa_mask_value = np.finfo(attn.dtype.as_numpy_dtype).min
        attn = tf.where(lsa_mask, lsa_mask_value, attn)

        # relative_position_bias(reshape): [Mh * Mw * Mh * Mw, nH] -> [Mh * Mw, Mh * Mw, nH]
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, [-1]),
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            [
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ],
        )
        # [nH, Mh * Mw, Mh * Mw]
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, 0)

        if mask is not None:
            # mask shape: [nW, Mh * Mw, Mh * Mw]
            nW = mask.shape[0]  # num_windows
            # attn(reshape): [batch_size, num_windows, num_heads, Mh * Mw, Mh * Mw]
            # mask(expand_dim): [1, nW, 1, Mh * Mw, Mh * Mw]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N]) + tf.expand_dims(
                tf.expand_dims(mask, 1), 0
            )
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # attn shape: [batch_size * num_windows, num_heads, Mh * Mw, Mh * Mw]
        # v shape: [batch_size * num_windows, num_heads, Mh * Mw, embed_dim_per_head]
        # multiply -> [batch_size * num_windows, num_heads, Mh * Mw, embed_dim_per_head]
        x = tf.matmul(attn, v)
        # transpose: -> [batch_size * num_windows, Mh * Mw, num_heads, embed_dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        # reshape: -> [batch_size * num_windows, Mh * Mw, total_embed_dim]
        x = tf.reshape(x, [-1, N, C])

        # x shape: [batch_size * num_windows, Mh * Mw, dim (C)]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(layers.Layer):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.fc1 = layers.Dense(
            int(in_features * mlp_ratio),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )
        self.act = layers.Activation("gelu")
        self.fc2 = layers.Dense(
            in_features,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(layers.Layer):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_ratio=attn_drop,
            proj_drop_ratio=drop,
        )
        self.drop_path = (
            layers.Dropout(rate=drop_path, noise_shape=(None, 1, 1))
            if drop_path > 0.0
            else layers.Activation("linear")
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(dim, drop=drop)

    def call(self, x, attn_mask):
        H, W = self.H, self.W
        # x shape: [B, L, C] = [B, H * W, C]
        _, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, [-1, H, W, C])

        # pad feature maps to multiples of window size
        padding_r = (self.window_size - W % self.window_size) % self.window_size
        padding_b = (self.window_size - H % self.window_size) % self.window_size
        if padding_r > 0 or padding_b > 0:
            paddings = tf.constant([[0, 0], [0, padding_r], [0, padding_b], [0, 0]])
            x = tf.pad(x, paddings)

        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        # x_windows shape: [nW * B, Mh, Mw, C]
        x_windows = window_partition(shifted_x, self.window_size)
        # x_windows shape: [nW * B, Mh * Mw, C]
        x_windows = tf.reshape(x_windows, [-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW * B, Mh * Mw, C]

        # merge windows
        # [nW * B, Mh, Mw, C]
        attn_windows = tf.reshape(
            attn_windows, [-1, self.window_size, self.window_size, C]
        )
        # [B, H', W', C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        else:
            x = shifted_x

        if padding_r > 0 or padding_b > 0:
            # remove the padding data
            x = tf.slice(x, begin=[0, 0, 0, 0], size=[-1, H, W, C])

        x = tf.reshape(x, [-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(layers.Layer):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (layer.Layer | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        downsample=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = [
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ]

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def create_mask(self, H, W):
        # calculate attention mask for SW-MSA
        # make sure Hp and Wp are multiples of window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # made the channels the same as feature map, for the convenience of window_partition
        img_mask = np.zeros([1, Hp, Wp, 1])  # [1, Hp, Wp, 1]
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        count = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = count
                count += 1

        img_mask = tf.convert_to_tensor(img_mask, dtype=tf.float32)
        # mask_windows shape: [1 * nW, Mh, Mw, 1]
        mask_windows = window_partition(img_mask, self.window_size)
        # mask_windows shape: [1 * nW, Mh * Mw]
        mask_windows = tf.reshape(
            mask_windows, [-1, self.window_size * self.window_size]
        )
        # [1 * nW, 1, Mh * Mw] - [1 * nW, Mh * Mw, 1]
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
        # attn_mask shape: [1 * nW, Mh * Mw, Mh * Mw]
        return attn_mask

    def call(self, x, H, W):
        #  attn_mask shape: [nW, Mh * Mw, Mh * Mw]
        attn_mask = self.create_mask(H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            # x shape: [B, L, C] = [B, H * W, C]
            x = blk(x, attn_mask)

        if self.downsample is not None:
            # downsampled x shape: [B, H / 2 * W / 2, 2 * C]
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class LSwinTransformer(tf.keras.Model):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
            https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        image_size,
        patch_size=4,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=layers.LayerNormalization,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer
        )

        # # Shifted Patch Tokenization
        # def _pair(t):
        #     return t if isinstance(t, tuple) else (t, t)

        # image_height, image_width = _pair(image_size)
        # patch_height, patch_width = _pair(patch_size)
        # assert (
        #     image_height % patch_height == 0 and image_width % patch_width == 0
        # ), "Image dimensions must be divisible by the patch size."
        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # self.patch_embed = ShiftedPatchTokenization(
        #     patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer
        # )
        # # self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, embed_dim]))
        # self.pos_embedding = tf.Variable(
        #     initial_value=tf.random.normal([1, num_patches + 1, embed_dim])
        # )

        self.pos_drop = layers.Dropout(drop_rate)

        # stochastic depth decay rule
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.stage_layers = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.stage_layers.append(layer)

        self.norm = norm_layer(epsilon=1e-6)
        self.head = layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )

    def call(self, x):
        # x shape: [B, L, C] = [B, H * W, C]
        x, H, W = self.patch_embed(x)

        # # Shifted Patch Tokenization
        # n = x.shape[1]
        # # the class token is not used
        # x = x + self.pos_embedding[:, :n]

        x = self.pos_drop(x)

        for layer in self.stage_layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]
        x = tf.reduce_mean(x, axis=1)
        x = self.head(x)

        return x
