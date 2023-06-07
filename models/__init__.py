import tensorflow as tf

from models.soft_net import SOFTNet
from models.soft_net_cbam import SOFTNetCBAM
from models.vision_transformer import ViT
from models.sl_vision_transformer import SLViT
from models.swin_transformer import SwinTransformer
from models.l_swin_transformer import LSwinTransformer
from models.s_swin_transformer import SSwinTransformer
from models.sl_swin_transformer import SLSwinTransformer

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005)  # SOFTNet default = 0.0005
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# optimizer = tf.keras.optimizers.experimental.AdamW(
#     learning_rate=0.01, weight_decay=0.0001
# )

# loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn = tf.keras.losses.MeanSquaredError()


def load_compile_model(model_name):
    if model_name == "SOFTNet":
        model = SOFTNet()
    elif model_name == "SOFTNetCBAM":
        model = SOFTNetCBAM()
    elif model_name == "ViT":
        model = ViT(
            img_size=42,
            patch_size=14,
            num_hiddens=768,
            mlp_num_hiddens=3072,
            num_heads=12,
            num_blks=6,
            emb_dropout=0.1,
            blk_dropout=0.1,
            num_classes=1,
        )
    elif model_name == "SL-ViT":
        model = SLViT(
            image_size=42,
            patch_size=16,
            dimension=768,
            head_dimension=16,
            mlp_dimension=3072,
            heads=8,
            depth=2,
            emb_dropout=0.1,
            blk_dropout=0.1,
            num_classes=1,
        )
    elif model_name == "Swin-T":
        model = SwinTransformer(
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "Swin-S":
        model = SwinTransformer(
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "L-Swin-T":
        model = LSwinTransformer(
            image_size=42,
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "S-Swin-T":
        model = SSwinTransformer(
            image_size=42,
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "SL-Swin-T":
        model = SLSwinTransformer(
            image_size=42,
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "SL-Swin-S":
        model = SLSwinTransformer(
            image_size=42,
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.MeanAbsoluteError()],
    )
    return model
