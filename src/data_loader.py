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

# # Convert to grayscale, then tensor, then normalize
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),  # Convert RGB → 1 channel grayscale
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale values
# ])


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

import numpy as np

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



# def resize_and_flatten(train_x, test_x,  val_x=None, outputshape=None, do_flatten=True):
#     #[resize(image, outputshape, anti_aliasing=True)
#     if outputshape is not None:
#         resized_train_x = np.array([resize(image, outputshape, anti_aliasing=True).flatten() for image in train_x])
#         if val_x is not None:
#             resized_val_x = np.array([resize(image, outputshape, anti_aliasing=True).flatten() for image in val_x])
#         resized_test_x = np.array([resize(image, outputshape, anti_aliasing=True).flatten() for image in test_x])
#     else:
#         resized_train_x = np.array([image.flatten() for image in train_x])
#         if val_x is not None:
#             resized_val_x = np.array([image.flatten() for image in val_x])
#         resized_test_x = np.array([image.flatten() for image in test_x])
#
#     if val_x is not None:
#         return resized_train_x,  resized_test_x, resized_val_x
#     else:
#         return resized_train_x, resized_test_x, None

def resize_and_flatten(train_x, test_x,  val_x=None, outputshape=None):
    #[resize(image, outputshape, anti_aliasing=True)
    resized_train_x = np.array([image.flatten() for image in train_x])
    if val_x is not None:
        resized_val_x = np.array([image.flatten() for image in val_x])
    resized_test_x = np.array([image.flatten() for image in test_x])

    if val_x is not None:
        return resized_train_x,  resized_test_x, resized_val_x
    else:
        return resized_train_x, resized_test_x, None


def preprocess_images(train_x: np.ndarray, test_x: np.ndarray, val_x:np.ndarray= None,outputshape=None,
                      pca_n_components=None, do_flatten=True) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the image for encoding. The images can be either be resized and directly returned for amplitude-encoding or
    preprocessed by PCA for amplitude-encoding or angle-encoding.

    :param train_x: train data to process
    :param test_x: test data to process
    :param outputshape: the shape of the output image (e.g. (16, 16)). If you want to use PCA, the outputshape should be (28, 28) which is the default
    :param pca_n_components: the number of components for PCA. If None, PCA is not used. If you want to use PCA let the outputshape be default
    :return: fully processed training and test data for encoding
    """
    print("Start preprocessing images for encoding...")
    if do_flatten:
        train_x,  test_x, val_x, = resize_and_flatten(train_x, test_x, val_x, outputshape)

    # train_images_resized = train_images_resized / 255.0
    # test_images_resized = test_images_resized / 255.0
    #
    # imagesX_train_flat_normalized = np.array([image / np.linalg.norm(image) for image in train_images_resized])
    # imagesX_test_flat_normalized = np.array([image / np.linalg.norm(image) for image in test_images_resized])
    #
    # # Padding
    # # is hier vllt ga nicht nötig weil wir die bilder schon auf 16x16 = 256 resized haben
    #train_images_resized = imagesX_train_flat_normalized  # np.array([np.pad(image, (0, 2 ** num_qubits - len(image)), 'constant') for image in imagesX_train_flat_normalized])
    #test_images_resized = imagesX_test_flat_normalized  # np.array([np.pad(image, (0, 2 ** num_qubits - len(image)), 'constant') for image in imagesX_test_flat_normalized])

    if pca_n_components is not None:
        pca = PCA(n_components=pca_n_components)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)
        val_x = pca.transform(val_x)


    print("Preprocessing images for encoding finished!!")
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





