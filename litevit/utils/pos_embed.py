import torch

def get_sincos_positional_embeddings(
    grid_size: tuple[int, ...], embed_dim: int, temperature: int =10000
) -> torch.Tensor:
    """
    Generate fixed sine-cosine positional embeddings for 2D or 3D grids.

    Args:
        grid_size: Tuple of spatial dimensions (H, W) or (D, H, W)
        embed_dim: embedding dimension (d_model)

    Returns:
        pos_embed: tensor of shape (1, num_tokens, embed_dim)
    """
    if not len(grid_size) in (2, 3): 
        raise ValueError("Only 2D or 3D grids are supported")
    if not embed_dim % len(grid_size) == 0: 
        raise ValueError("Embedding dimension must be divisible by number of axes")

    num_axes = len(grid_size)
    dim_per_axis = embed_dim // num_axes

    # Create coordinate grid
    grids = torch.meshgrid(
        [torch.arange(s, dtype=torch.float32) for s in grid_size], indexing="ij"
    )
    # flatten to (num_positions,)
    grids = [g.flatten() for g in grids]

    embeddings = []
    for coords in grids:
        omega = 1 / (
            temperature ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis)
        )
        # shape: (num_positions, dim_per_axis/2)
        coords = coords[:, None] * omega[None, :]
        sin_emb = torch.sin(coords)
        cos_emb = torch.cos(coords)
        # interleave sin and cos
        emb = torch.empty(coords.shape[0], dim_per_axis)
        emb[:, 0::2] = sin_emb
        emb[:, 1::2] = cos_emb
        embeddings.append(emb)

    # concatenate along embedding dimension
    pos_embed = torch.cat(embeddings, dim=1)

    return pos_embed.unsqueeze(0)