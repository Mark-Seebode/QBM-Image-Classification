import numpy as np
import gzip

import numpy.random
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#from skimage.transform import resize
#import torchvision.transforms as transforms
#from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
#import torchvision
from sklearn.model_selection import train_test_split
import os, random
from PIL import Image

# # Convert to grayscale, then tensor, then normalize
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),  # Convert RGB → 1 channel grayscale
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale values
# ])

def get_imagenet(
        root: str,
        classes: list[str] = None,
        n_per_class: int = 100,
        image_size: tuple[int,int] = (224,224),
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
        to_grayscale: bool = True,   # <- default: make 2D images
    ):
    """returns train, val, test split as numpy arrays from a subset of ImageNet dataset stored in `root` directory."""
    rng = np.random.default_rng(seed)
    all_classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if classes is None:
        classes = all_classes[:2]
    print("Using classes:", classes)

    X, y = [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(root, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        rng.shuffle(imgs)
        imgs = imgs[:n_per_class]
        for p in imgs:
            with Image.open(p) as im:
                if to_grayscale:
                    im = im.convert('L')  # <- 1 channel
                else:
                    im = im.convert('RGB')
                im = im.resize(image_size, Image.Resampling.LANCZOS)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                X.append(arr)
                y.append(label)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.uint8)

    # If grayscale, X shape is (N, H, W); if RGB, (N, H, W, 3)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_ratio+test_ratio, stratify=y, random_state=seed
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_mnist(file_image: str, file_labels: str, classes: list[int] = None, size: int = None, samples_per_class:int =None, seed: int = 42) -> tuple[np.array, np.array]:
    '''
        Read MNIST dataset and return it as numpy array.

        Parameters
        ----------
        file_image: str
            the path and file name of the image dataset, for example, file_image='../mnist/train-images-idx3-ubyte.gz'
        file_labels: str
            the path and file name of the label dataset, for example, file_labels='../mnist/train-labels-idx1-ubyte.gz'
    
        Return
        ----------
        np.array
            the images as numpy tensor with the shape (n_images, x_dim, y_dim)
        np.array
            the labels as numpy tensor with the shape (n_images, x_dim, y_dim)
    '''
    with gzip.open(file_image, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(),'B',offset=16).reshape(-1, 28, 28).astype('float32') / 255
    with gzip.open(file_labels, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(),'B',offset=8)

    if classes is not None:
        mask = np.isin(labels, classes)

        images = images[mask]
        labels = labels[mask]

    if samples_per_class is not None:
        selected_images = []
        selected_labels = []
        np.random.seed(seed)
        for cls in np.unique(labels):
            class_indices = np.where(labels == cls)[0]
            selected_indices = np.random.choice(class_indices, size=samples_per_class, replace=False)
            selected_images.append(images[selected_indices])
            selected_labels.append(labels[selected_indices])
        images = np.concatenate(selected_images)
        labels = np.concatenate(selected_labels)

    if size is not None:
        images, _, labels, _ = train_test_split(images, labels, train_size=size, random_state=seed)

    # Necessary because of the one hot encoding inside the cross entropy function
    if classes is not None and len(classes) == 2:
        labels = [0 if label == classes[0] else 1 for label in labels]

    print("Images: ", len(images))

    return images, labels


def get_fashionmnist(file_image: str, file_labels: str, classes: list[int] = None, size: int = None, samples_per_class:int = None, seed: int = 42) -> tuple[np.array, np.array]:
    '''
        Read fashionMNIST dataset and return it as numpy array.
    '''
    with open(file_image, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(),'B',offset=16).reshape(-1, 28, 28).astype('float32') / 255
    with open(file_labels, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(),'B',offset=8)


    if classes is not None:
        mask = np.isin(labels, classes)

        images = images[mask]
        labels = labels[mask]

    if samples_per_class is not None:
        selected_images = []
        selected_labels = []
        np.random.seed(seed)
        for cls in np.unique(labels):
            class_indices = np.where(labels == cls)[0]
            selected_indices = np.random.choice(class_indices, size=samples_per_class, replace=False)
            selected_images.append(images[selected_indices])
            selected_labels.append(labels[selected_indices])
        images = np.concatenate(selected_images)
        labels = np.concatenate(selected_labels)

    if size is not None:
        images, _, labels, _ = train_test_split(images, labels, train_size=size, random_state=seed)

    if classes is not None and len(classes) == 2:
        labels = [0 if label == classes[0] else 1 for label in labels]

    print("Images: ", len(images))

    return images, labels


def get_medmnist(file: str, index: int = 0, duplicate_positives_n_times: int = 0, do_balance: bool = False, size= None, seed=42) -> tuple[tuple[np.array, np.array], tuple[np.array, np.array], tuple[np.array, np.array]]:
    ''''
        Read medMNIST dataset and return it as numpy array.
    '''
    # all data is one file
    data = np.load(file)
    np.random.seed(seed)

    # process images data
    train_images = data['train_images'].astype('float32') / 255
    val_images = data['val_images'].astype('float32') / 255
    test_images = data['test_images'].astype('float32') / 255

    # transform multi-label classification to one-label classification
    train_labels = data['train_labels'][:,index]
    val_labels = data['val_labels'][:,index]
    test_labels = data['test_labels'][:,index]

    if size is not None:
        selected_images = []
        selected_labels = []
        for cls, cls_size in enumerate(size):
            class_indices = np.where(train_labels == cls)[0]
            if len(class_indices) > cls_size:
                selected_indices = np.random.choice(class_indices, size=cls_size, replace=False)
            else:
                selected_indices = np.random.choice(class_indices, size=cls_size, replace=True)
            selected_images.append(train_images[selected_indices])
            selected_labels.append(train_labels[selected_indices])
        train_images = np.concatenate(selected_images)
        train_labels = np.concatenate(selected_labels)

    if duplicate_positives_n_times > 0:
        pos_train_indices = np.where(train_labels == 1)[0]
        for i in range(duplicate_positives_n_times):
            # duplicate positive samples in training set
            train_images = np.concatenate([train_images, train_images[pos_train_indices]])
            train_labels = np.concatenate([train_labels, train_labels[pos_train_indices]])


    if do_balance:
        # balance training set
        pos_train_indices = np.where(train_labels == 1)[0]
        neg_train_indices = np.where(train_labels == 0)[0]
        num_pos = len(pos_train_indices)

        neg_indices = np.random.choice(neg_train_indices, num_pos)

        train_images = np.concatenate([train_images[pos_train_indices], train_images[neg_indices]])
        train_labels = np.concatenate([train_labels[pos_train_indices], train_labels[neg_indices]])


    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def get_NEU_CLS_64(file: str, classes: list[str] = None, num_samples_per_class=100, train_test_percentage: float = 0.8, seed: int = 42, image_size: tuple[int,int]=(64,64)):
    """
    Load NEU-CLS-64 dataset from class folders containing jpg images


    """
    if classes is None: # all classes
        classes = sorted([d for d in os.listdir(file) if os.path.isdir(os.path.join(file, d))])
    print("Using classes:", classes)

    x, y = [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(file, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for p in imgs:
            with Image.open(p) as im:
                im = im.convert('L')  # <- 1 channel
                im = im.resize(image_size, Image.Resampling.LANCZOS)
                arr = np.asarray(im) / 255.0
                x.append(arr)
                y.append(label)

    x = np.stack(x).astype(np.float32)
    y = np.array(y, dtype=np.uint8)

    # randomly pick samples_per_class from each class
    selected_x = []
    selected_y = []
    np.random.seed(seed)
    for cls in np.unique(y):
        class_indices = np.where(y == cls)[0]
        selected_indices = np.random.choice(class_indices, size=num_samples_per_class, replace=False)
        selected_x.append(x[selected_indices])
        selected_y.append(y[selected_indices])

    x = np.concatenate(selected_x)
    y = np.concatenate(selected_y)

    # show sample image after resizing from both classes
    import matplotlib.pyplot as plt
    plt.imshow(x[0], cmap='gray')
    plt.title(f"Sample image from class {y[0]}")
    plt.show()
    plt.imshow(x[-1], cmap='gray')
    plt.title(f"Sample image from class {y[-1]}")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=train_test_percentage, test_size=1-train_test_percentage, random_state=seed, shuffle=True)

    return X_train, y_train, X_test, y_test



def balance_by_undersampling(train_images, train_labels):
    """
    Balances the dataset by randomly undersampling the majority class.

    Parameters:
    - train_images (np.ndarray): Array of input images.
    - train_labels (np.ndarray): Corresponding binary labels (0 and 1).

    Returns:
    - balanced_images (np.ndarray): Balanced image set.
    - balanced_labels (np.ndarray): Corresponding labels.
    """
    pos_indices = np.where(train_labels == 1)[0]
    neg_indices = np.where(train_labels == 0)[0]

    if len(pos_indices) == len(neg_indices):
        return train_images, train_labels  # already balanced

    # Identify the majority and minority
    if len(pos_indices) > len(neg_indices):
        majority_indices = pos_indices
        minority_indices = neg_indices
    else:
        majority_indices = neg_indices
        minority_indices = pos_indices

    # Randomly sample the majority class to match the size of the minority
    np.random.shuffle(majority_indices)
    majority_sampled = majority_indices[:len(minority_indices)]

    # Combine balanced indices
    balanced_indices = np.concatenate([minority_indices, majority_sampled])
    np.random.shuffle(balanced_indices)

    return train_images[balanced_indices], train_labels[balanced_indices]


def flatten_images(train_x, test_x, val_x=None):
    #[resize(image, outputshape, anti_aliasing=True)
    train_x = np.array([image.flatten() for image in train_x])
    if val_x is not None:
        val_x = np.array([image.flatten() for image in val_x])
    test_x = np.array([image.flatten() for image in test_x])

    if val_x is not None:
        return train_x,  test_x, val_x
    else:
        return train_x, test_x, None


def pca_transform(pca_n_components, train_x: np.ndarray, test_x: np.ndarray, val_x:np.ndarray= None) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply PCA transformation to the data.

    :param train_x: train data to process
    :param test_x: test data to process
    :param pca_n_components: the number of components for PCA. If None, PCA is not used. If you want to use PCA let the outputshape be default
    :return: fully processed training and test data for encoding
    """
    pca = PCA(n_components=pca_n_components)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    val_x = pca.transform(val_x)

    return train_x, test_x, val_x,


def shuffle_images(x, y, seed=44):
    assert len(x) == len(y), "Input data and labels must have the same length."
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    x_shuf = np.take(x, indices, axis=0)
    y_shuf = np.take(y, indices, axis=0)
    return x_shuf, y_shuf


def encode_labels_to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Encode the labels to one-hot encoding.

    :param labels: the labels to encode
    :param num_classes: the number of classes
    :return: the one-hot encoded labels
    """
    return np.eye(num_classes)[labels]



def reshape_pad_and_flatten(images, original_shape, pad):
    """
    Reshape flattened images, pad with zeros, and flatten them again.

    Parameters:
    - flattened_images (np.ndarray): shape (N, H*W), flattened images.
    - original_shape (tuple): (H, W) shape of the original unflattened image.
    - pad (int): number of zero pixels to pad on each side.

    Returns:
    - np.ndarray: shape (N, (H+2*pad)*(W+2*pad)), padded and re-flattened images.
    """
    H, W = original_shape
    N = images.shape[0]

    # Reshape to (N, H, W)
    images = images.reshape((N, H, W))

    # Apply symmetric zero padding
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad, pad), (pad, pad)),  # (batch, height, width)
        mode='constant',
        constant_values=0
    )

    return padded_images





