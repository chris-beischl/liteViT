from timm.models.vision_transformer import VisionTransformer


def build_timm_vit(
    embed_dim: int,
    img_size: int | tuple[int, int],
    patch_size: int | tuple[int, int],
    in_chans: int,
    depth: int,
    num_heads: int,
    num_classes: int,
    pos_embed_type: str = "learn",  # different from our ViT implementation
    dropout: float = 0.0,
    drop_path_rate: float = 0.0,
) -> VisionTransformer:

    return VisionTransformer(
        embed_dim=embed_dim,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        depth=depth,
        num_heads=num_heads,
        num_classes=num_classes,
        pos_embed=pos_embed_type,
        pos_drop_rate=dropout,
        drop_path_rate=drop_path_rate,
    )
