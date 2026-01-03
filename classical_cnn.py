import torch
import torch.nn as nn
import torch.nn.functional as F
import src.data_loader as data_loader
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class SimpleCNN(nn.Module):
    """
    CNN for classification on 28x28 grayscale images.
    Supports binary (num_classes=2) and multi-class (num_classes>2).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.num_classes = num_classes

        # Conv: 28x28 -> (28-3+1)=26 (no padding)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=0)

        # Pool 4x4 stride 4: 26x26 -> 6x6 (since floor((26-4)/4)+1 = 6)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.fc1 = nn.Linear(1 * 6 * 6, 31)

        # Binary -> 1 logit, Multi-class -> C logits
        out_dim = 1 if num_classes == 2 else num_classes
        self.fc2 = nn.Linear(31, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # (N,1) for binary or (N,C) for multi
        return x


def create_dataloaders(
    data_dir: str,
    classes: list[str],
    batch_size: int = 64,
    seed: int = 42,
    num_samples_per_class=None,
):
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.get_NEU_CLS_64(
        file=data_dir,
        classes=classes,
        num_samples_per_class=num_samples_per_class,
        image_size=(28, 28), contrast_factor=1.5,
        seed=seed,
    )

    # X_train, y_train = data_loader.get_mnist('src/data/mnist/train-images-idx3-ubyte.gz',
    #                                          'src/data/mnist/train-labels-idx1-ubyte.gz', classes=[0, 1]
    #                                          , samples_per_class=50)
    # X_val, y_val = data_loader.get_mnist('src/data/mnist/t10k-images-idx3-ubyte.gz',
    #                                        'src/data/mnist/t10k-labels-idx1-ubyte.gz', classes=[0, 1],
    #                                        samples_per_class=20)

    X_test = X_val
    y_test = y_val

    # tensors + channel dim
    X_train_t = torch.from_numpy(X_train).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val).unsqueeze(1)
    X_test_t = torch.from_numpy(X_test).unsqueeze(1)

    num_classes = len(classes)
    if num_classes == 2:
        # BCE expects float targets {0,1}
        y_train_t = torch.from_numpy(y_train.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.float32))
        y_test_t = torch.from_numpy(y_test.astype(np.float32))
    else:
        # CE expects int64 class indices {0..C-1}
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        y_val_t = torch.from_numpy(y_val.astype(np.int64))
        y_test_t = torch.from_numpy(y_test.astype(np.int64))

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    is_binary = (model.num_classes == 2)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)

        if is_binary:
            # logits: (N,1), targets: (N,) -> (N,1)
            yb_ = yb.float().unsqueeze(1)
            loss = criterion(logits, yb_)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().squeeze(1)
            correct += (preds == yb.long()).sum().item()
            total += yb.size(0)
        else:
            # logits: (N,C), targets: (N,)
            loss = criterion(logits, yb.long())
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb.long()).sum().item()
            total += yb.size(0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """
    Returns: avg_loss, accuracy, auc
    - Binary: AUC = ROC-AUC (binary)
    - Multi-class: AUC = ROC-AUC OvR macro
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    is_binary = (model.num_classes == 2)

    all_targets = []
    all_scores = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            if is_binary:
                yb_ = yb.float().unsqueeze(1)
                loss = criterion(logits, yb_)
                probs = torch.sigmoid(logits).squeeze(1)   # (N,)
                preds = (probs >= 0.5).long()
                correct += (preds == yb.long()).sum().item()
                total += yb.size(0)

                all_targets.append(yb.long().cpu().numpy())  # (N,)
                all_scores.append(probs.cpu().numpy())       # (N,)
            else:
                loss = criterion(logits, yb.long())
                probs = torch.softmax(logits, dim=1)         # (N,C)
                preds = torch.argmax(probs, dim=1)
                correct += (preds == yb.long()).sum().item()
                total += yb.size(0)

                all_targets.append(yb.long().cpu().numpy())  # (N,)
                all_scores.append(probs.cpu().numpy())       # (N,C)

            total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    # AUC
    from sklearn.metrics import roc_auc_score
    try:
        y_true = np.concatenate(all_targets, axis=0)
        y_score = np.concatenate(all_scores, axis=0)

        if is_binary:
            # y_score shape (N,) ok
            auc = roc_auc_score(y_true, y_score)
        else:
            # y_score shape (N,C)
            auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return avg_loss, acc, auc

import matplotlib.pyplot as plt
import numpy as np

def plot_filters(weights, title):
    """
    weights: (out_channels, in_channels, H, W)
    """
    n_filters = weights.shape[0]

    fig, axes = plt.subplots(1, n_filters, figsize=(2*n_filters, 2))
    if n_filters == 1:
        axes = [axes]

    for i in range(n_filters):
        filt = weights[i, 0].numpy()  # single input channel
        ax = axes[i]
        im = ax.imshow(filt, cmap="gray")
        ax.set_title(f"Filter {i}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    data_dir = "src/data/NEU-CLS-64"

    # Binary: provide 2 classes
    # Multi-class: provide >= 3 classes
    classes = ["gg", "rp"]  # works for both (len decides mode)

    batch_size = 2
    lr = 0.001
    num_epochs = 20
    seed = 55

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        classes=classes,
        batch_size=batch_size,
        seed=seed,
    )

    num_classes = len(classes)
    model = SimpleCNN(num_classes=num_classes).to(device)

    initial_weights = model.conv1.weight.detach().cpu().clone()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}"
        )

    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, Test AUC={test_auc:.4f}")

    trained_weights = model.conv1.weight.detach().cpu()

    plot_filters(initial_weights, title="Initial Conv1 Filters")
    plot_filters(trained_weights, title="Trained Conv1 Filters")

if __name__ == "__main__":
    main()
