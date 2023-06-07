import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PatchEmbedding(layers.Layer):
    def __init__(self, image_size=96, patch_size=16, dimension=512):
        super().__init__()

        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x

        image_size, patch_size = _make_tuple(image_size), _make_tuple(patch_size)
        self.dimension = dimension
        self.num_patches = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )
        self.conv = layers.Conv2D(
            dimension,
            kernel_size=patch_size,
            strides=patch_size,
        )

    def call(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return tf.reshape(
            self.conv(X),
            [
                -1,
                self.num_patches,
                self.dimension,
            ],
        )


class ShiftedPatchTokenization(layers.Layer):
    def __init__(self, image_size, patch_size, dimension):
        super().__init__()
        self.to_patch_tokens = tf.keras.Sequential(
            [
                PatchEmbedding(image_size, patch_size, dimension),
                layers.LayerNormalization(),
                layers.Dense(units=dimension),
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
        shifted_x = self.shift(x)
        x_with_shifts = tf.concat([x, *shifted_x], axis=-1)
        x = self.to_patch_tokens(x_with_shifts)
        return x


class LocalitySelfAttention(layers.Layer):
    def __init__(self, dimension, heads, head_dimension, dropout=0.1):
        super().__init__()
        inner_dimension = heads * head_dimension
        self.heads = heads
        self.temperature = tf.Variable(tf.math.log(head_dimension**-0.5))
        self.to_qkv = layers.Dense(units=inner_dimension * 3, use_bias=False)
        self.softmax = layers.Softmax()
        self.to_output = tf.keras.Sequential(
            [layers.Dense(units=dimension), layers.Dropout(rate=dropout)]
        )

    def call(self, x):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(
            lambda t: tf.reshape(
                t, (-1, self.heads, t.shape[1], t.shape[2] // self.heads)
            ),
            qkv,
        )
        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dot_products = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * tf.math.exp(
            self.temperature
        )
        mask = tf.eye(dot_products.shape[-1], dtype=tf.bool)
        mask_value = np.finfo(dot_products.dtype.as_numpy_dtype).min
        dot_products = tf.where(mask, mask_value, dot_products)
        attention_score = self.softmax(dot_products)
        output = tf.matmul(attention_score, v)
        output = tf.reshape(
            output, (-1, output.shape[2], output.shape[1] * output.shape[-1])
        )
        # output = rearrange(output, "b h n d -> b n (h d)")
        output = self.to_output(output)
        return output


class MLP(layers.Layer):
    def __init__(self, hidden_dimension, dimension, dropout=0.5):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dimension)
        self.activation = layers.Activation("gelu")
        self.dropout1 = layers.Dropout(dropout)
        self.dense2 = layers.Dense(dimension)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x):
        return self.dropout2(
            self.dense2(self.dropout1(self.activation(self.dense1(x))))
        )


class ViTBlock(layers.Layer):
    def __init__(self, dimension, heads, head_dimension, mlp_dimension, dropout):
        super().__init__()
        self.layernormalization = layers.LayerNormalization()
        self.attention = LocalitySelfAttention(
            dimension=dimension,
            heads=heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.mlp = MLP(
            hidden_dimension=mlp_dimension, dimension=dimension, dropout=dropout
        )

    def call(self, x):
        x = self.layernormalization(x)
        x = self.attention(x) + x
        x = self.layernormalization(x)
        x = self.mlp(x) + x
        return x


class SLViT(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        dimension,
        head_dimension,
        mlp_dimension,
        heads,
        depth,
        emb_dropout,
        blk_dropout,
        num_classes,
        pool="cls",
    ):
        super().__init__()

        def _pair(t):
            return t if isinstance(t, tuple) else (t, t)

        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pool = pool
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.patch_embedding = ShiftedPatchTokenization(
            image_size=image_size, dimension=dimension, patch_size=patch_size
        )
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dimension]))
        self.pos_embedding = tf.Variable(
            initial_value=tf.random.normal([1, num_patches + 1, dimension])
        )
        self.dropout = layers.Dropout(rate=emb_dropout)
        self.blks = [
            ViTBlock(
                dimension,
                heads,
                head_dimension,
                mlp_dimension,
                blk_dropout,
            )
            for _ in range(depth)
        ]
        self.mlp_head = tf.keras.Sequential(
            [layers.LayerNormalization(), layers.Dense(units=num_classes)]
        )

    def call(self, img):
        x = self.patch_embedding(img)
        n = x.shape[1]
        cls_tokens = tf.tile(self.cls_token, (tf.shape(x)[0], 1, 1))
        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        for blk in self.blks:
            x = blk(x)

        if self.pool == "mean":
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x
