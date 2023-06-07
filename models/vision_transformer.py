import tensorflow as tf
from tensorflow.keras import layers


class PatchEmbedding(layers.Layer):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()

        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x

        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_hiddens = num_hiddens
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )
        self.conv = layers.Conv2D(
            num_hiddens,
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
                self.num_hiddens,
            ],
        )


def masked_softmax(X, valid_lens):
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[None, :] < tf.cast(
            valid_len[:, None], dtype=tf.float32
        )

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)

    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])

        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens, value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)


class DotProductAttention(layers.Layer):
    def __init__(self, dropout, num_heads=None):
        super().__init__()
        self.dropout = layers.Dropout(dropout)
        self.num_heads = num_heads  # To be covered later

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def call(self, queries, keys, values, valid_lens=None, window_mask=None):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(
            tf.cast(d, dtype=tf.float32)
        )
        if window_mask is not None:  # To be covered later
            num_windows = window_mask.shape[0]
            n, num_queries, num_kv_pairs = scores.shape
            # Shape of window_mask: (num_windows, no. of queries,
            # no. of key-value pairs)
            scores = tf.reshape(
                scores,
                (
                    n // (num_windows * self.num_heads),
                    num_windows,
                    self.num_heads,
                    num_queries,
                    num_kv_pairs,
                ),
            ) + tf.expand_dims(tf.expand_dims(window_mask, 1), 0)
            scores = tf.reshape(scores, (n, num_queries, num_kv_pairs))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights), values)


class MultiHeadAttention(layers.Layer):
    def __init__(
        self,
        num_hiddens,
        num_heads,
        dropout,
        bias=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout, num_heads)
        self.W_q = layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = layers.Dense(num_hiddens, use_bias=bias)

    def call(self, queries, keys, values, valid_lens, window_mask=None):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens, window_mask)

        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        # X = tf.reshape(X, shape=(X.shape[0], X.shape[1], self.num_heads, -1))
        X = tf.reshape(
            X, shape=(-1, X.shape[1], self.num_heads, X.shape[2] // self.num_heads)
        )
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))

    def transpose_output(self, X):
        # Shape of X: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        X = tf.reshape(X, shape=(-1, self.num_heads, X.shape[1], X.shape[2]))
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        # return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))
        return tf.reshape(X, shape=(-1, X.shape[1], X.shape[3] * self.num_heads))


class ViTMLP(layers.Layer):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = layers.Dense(mlp_num_hiddens)
        self.activation = layers.Activation("gelu")
        self.dropout1 = layers.Dropout(dropout)
        self.dense2 = layers.Dense(mlp_num_outputs)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x):
        return self.dropout2(
            self.dense2(self.dropout1(self.activation(self.dense1(x))))
        )


class ViTBlock(layers.Layer):
    def __init__(
        self,
        num_hiddens,
        mlp_num_hiddens,
        num_heads,
        dropout,
        bias=False,
    ):
        super().__init__()
        self.ln1 = layers.LayerNormalization()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, bias)
        self.ln2 = layers.LayerNormalization()
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def call(self, X, valid_lens=None):
        X = self.ln1(X)
        return X + self.mlp(self.ln2(X + self.attention(X, X, X, valid_lens)))


class ViT(tf.keras.Model):
    def __init__(
        self,
        img_size,
        patch_size,
        num_hiddens,
        mlp_num_hiddens,
        num_heads,
        num_blks,
        emb_dropout,
        blk_dropout,
        num_classes,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = tf.Variable(initial_value=tf.zeros([1, 1, num_hiddens]))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = tf.Variable(
            initial_value=tf.random.normal([1, num_steps, num_hiddens])
        )
        self.dropout = layers.Dropout(emb_dropout)
        self.blks = [
            ViTBlock(
                num_hiddens,
                mlp_num_hiddens,
                num_heads,
                blk_dropout,
                bias=False,
            )
            for _ in range(num_blks)
        ]
        self.head = tf.keras.Sequential(
            [
                layers.LayerNormalization(),
                layers.Dense(num_hiddens, activation="sigmoid"),
                layers.Dense(num_classes),
            ]
        )

    def call(self, X):
        X = self.patch_embedding(X)
        # batch_cls_token = tf.tile(self.cls_token, (tf.shape(X).numpy()[0], 1, 1))
        batch_cls_token = tf.tile(self.cls_token, (tf.shape(X)[0], 1, 1))
        X = tf.concat([batch_cls_token, X], 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
