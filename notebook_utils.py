import os
import numpy as np
from PIL import Image

def load_neu_cls_images(classes: list[str]):
    """
    Load NEU-CLS images as raw arrays (no preprocessing).

    Returns:
        x : list[np.ndarray]  # raw images
        y : np.ndarray        # labels
        classes : list[str]
    """
    file = "src/data/NEU-CLS-64"
    if classes is None:
        classes = sorted([
            d for d in os.listdir(file)
            if os.path.isdir(os.path.join(file, d))
        ])

    print("Using classes:", classes)

    x = []
    y = []

    for label, cls in enumerate(classes):
        cls_dir = os.path.join(file, cls)

        imgs = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for p in imgs:
            with Image.open(p) as im:
                x.append(np.asarray(im))  # raw image
                y.append(label)

    return x, np.array(y, dtype=np.uint8)


from sklearn.model_selection import train_test_split
from PIL import ImageEnhance

def process_neu_cls_images(
    x,
    y,
    train_test_val_split=None,
    num_samples_per_class=None,
    seed=42,
    image_size=(64, 64),
    contrast_factor=None,
    normalize=True,
    grayscale=True
):

    if train_test_val_split is None:
        train_test_val_split = [0.7, 0.15, 0.15]

    processed = []

    for img in x:
        im = Image.fromarray(img)

        if grayscale:
            im = im.convert("L")

        im = im.resize(image_size, Image.Resampling.LANCZOS)

        if contrast_factor is not None:
            enhancer = ImageEnhance.Contrast(im)
            im = enhancer.enhance(contrast_factor)

        arr = np.asarray(im).astype(np.float32)

        if normalize:
            arr /= 255.0

        processed.append(arr)

    x = np.stack(processed)

    # optional balanced sampling
    if num_samples_per_class is not None:
        np.random.seed(seed)
        selected_x = []
        selected_y = []

        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            chosen = np.random.choice(idx, size=num_samples_per_class, replace=False)
            selected_x.append(x[chosen])
            selected_y.append(y[chosen])

        x = np.concatenate(selected_x)
        y = np.concatenate(selected_y)

    # split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        x, y,
        test_size=train_test_val_split[2],
        random_state=seed,
        shuffle=True
    )

    relative_val_size = train_test_val_split[1] / (
        train_test_val_split[0] + train_test_val_split[1]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_size,
        random_state=seed,
        shuffle=True
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_label_batch(model, one_hot, Y):
    if one_hot:
        lab_batch = np.zeros((len(Y), model.num_label_nodes), dtype=float)
        for idx, yy in enumerate(Y):
            lab_batch[idx, int(yy)] = 1.0
    else:
        lab_batch = np.array([[int(yy)] for yy in Y], dtype=float)

    return lab_batch

