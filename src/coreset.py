import torch
import numpy as np
import src.model.cdqbm as cdqbm


def euclidean_dist( x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def _to_tensor_2d(emb, device):
    if isinstance(emb, np.ndarray):
        assert emb.ndim == 2
        t = torch.from_numpy(emb).to(device)
    elif isinstance(emb, torch.Tensor):
        assert emb.dim() == 2
        t = emb.to(device)
    else:
        raise TypeError("embedding_matrix must be np.ndarray or torch.Tensor")
    # cast to float32 to save memory and for speed
    return t.float().contiguous()

def k_center_greedy(
    embedding_matrix,
    budget: int,
    metric=euclidean_dist,
    device=torch.device("cpu"),
    random_seed=None,
    index=None,
    already_selected=None,
    print_freq: int = 20,
):
    emb = _to_tensor_2d(embedding_matrix, device)
    budget += 1
    sample_num = emb.shape[0]
    if budget < 0:
        raise ValueError("Illegal budget size.")
    if budget > sample_num:
        budget = sample_num

    if index is not None:
        assert sample_num == len(index)
        index = np.asarray(index)
    else:
        index = np.arange(sample_num)

    if already_selected is None:
        already_selected = []
    else:
        already_selected = list(already_selected)

    if not callable(metric):
        raise ValueError("metric must be callable")

    with torch.no_grad():
        np.random.seed(random_seed)
        select_mask = np.zeros(sample_num, dtype=bool)

        # seed center
        if len(already_selected) == 0:
            first = np.random.randint(0, sample_num)
            already_selected = [first]
        select_mask[already_selected] = True

        num_seed = select_mask.sum()
        remaining_budget = budget - num_seed
        if remaining_budget <= 0:
            return index[select_mask]

        # Distance rows: [seed centers; newly picked centers]
        dis_matrix = torch.full(
            (num_seed + remaining_budget, sample_num),
            -1.0,
            dtype=emb.dtype,
            device=device,
        )

        # distances from existing centers to non-selected points
        if num_seed > 0:
            dis_matrix[:num_seed, ~select_mask] = metric(
                emb[select_mask], emb[~select_mask]
            )

        mins = dis_matrix[:num_seed].min(dim=0).values
        # For uninitialized columns (all -1 because all are selected), set +inf to avoid picking them
        mins[select_mask] = float("inf")

        for i in range(remaining_budget):
            if i % print_freq == 0:
                print(f"| Selecting [{i+1:3d}/{remaining_budget:3d}]")
            p = torch.argmax(mins).item()
            select_mask[p] = True
            if i == remaining_budget - 1:
                break
            mins[p] = -1  # exclude it
            dis_matrix[num_seed + i, ~select_mask] = metric(
                emb[[p]], emb[~select_mask]
            )
            mins = torch.min(mins, dis_matrix[num_seed + i])

    return index[select_mask]


def downsample_kcenter_with_light_model(qbm: cdqbm.Conv_Deep_Disc_QBM, train_x, train_y, budget: int, random_seed: int):
    print("Gathering hidden embeddings...")
    train_hidden_emb = qbm.get_last_hidden_embedding(train_x)
    print("finished\n")

    print("Selecting coreset...")
    coreset_indices = k_center_greedy(train_hidden_emb, budget=budget, random_seed=random_seed)
    print("finished\n")

    train_x = train_x[coreset_indices]
    train_y = train_y[coreset_indices]

    return train_x, train_y

def random_downsampling(x, y, downsample_size, seed: int = 42):
    """
    Randomly downsample the dataset to a smaller size.

    :param x: Input data (numpy array).
    :param y: Corresponding labels (numpy array).
    :return: Downsampled x and y.
    """

    if len(x) < downsample_size:
        raise ValueError("Downsample size must be less than the number of samples in the dataset.")

    np.random.seed(seed)
    indices = np.random.choice(len(x), downsample_size, replace=False)
    x_downsampled = x[indices]
    y_downsampled = y[indices]

    return x_downsampled, y_downsampled




