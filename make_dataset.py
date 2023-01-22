import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST

from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union


def custom_transform(img):
    # Input: (28, 28) uint8 [0, 255] torch.Tensor, Output: (28, 28, 1) float32 [0, 1] np array
    return np.expand_dims(np.array(img, dtype=np.float32), axis=2) / 255.

def numpy_collate(batch: Any):
    """
    Function to provide batches numpy arrays instead of torch tensors.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def create_data_loaders(*datasets: Sequence[data.Dataset],
                        train: Union[bool, Sequence[bool]] = True,
                        batch_size: int = 128,
                        num_workers: int = 4,
                        seed: int = 42):
    """
    Creates data loaders for a set of datasets to be compatible with JAX.

    Args:
    -----
        datasets: Sequence[data.Dataset]
            Datasets for which data loaders are created
        train: Union[bool, Sequence[bool]]
            Sequence indicating which datasets are used for training
            and which are not. If single bool, the same value is used
            for all datasets.
        batch_size: int
            Batch size to use in the data loaders.
        num_workers: int
            Number of workers for each dataset
        seed: int
            Seed to initialize the workers and shuffling with.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=is_train,
                                 drop_last=is_train,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers,
                                 persistent_workers=is_train,
                                 generator=torch.Generator().manual_seed(seed))
        loaders.append(loader)
    return loaders


def load_data_loaders(batch_size: int = 128,
                      ds_path: str = 'saved_data/', 
                      val_on: bool = True):
    """
    Function to load the created dataloaders.

    Args:
    -----
        ds_path: str
            Path where the datasets should be saved
        val_on: bool
            Toggle to decide if we want a validation set or just train/test sets.
    """
    
    # Converting a uint8 [0, 255] torch.Tensor to float32 [0,1] np.array
    test_transform = custom_transform
    # For training, we add some augmentation to reduce overfitting.
    train_transform = custom_transform

    if val_on:
        # Loading the training dataset. Because val_on = True we need to split it into
        # training and validation sets. We also need to do a little trick because the
        # validation set should not use the augmentation (ie. having same behavior as
        # the test set).
        train_dataset = MNIST(root=ds_path + "train", 
                              train=True,
                              transform=train_transform,
                              download=True)
        val_dataset = MNIST(root=ds_path + "val",
                            train=True,
                            transform=test_transform,
                            download=True)
        # Randomly splitting (with the same seed) the training/validation training sets and then only saving the
        # respective datasets for each one. I.e. the training set gets 50,000 samples, while the val set gets 10,000.
        train_set, _ = data.random_split(train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))
        _ , val_set = data.random_split(val_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))

        # Loading the test set
        test_set = MNIST(root=ds_path + "test",
                         train=False,
                         transform=test_transform,
                         download=True)

        # Create the train/val/test data loaders
        train_loader, val_loader, test_loader = create_data_loaders(train_set, val_set, test_set,
                                                                    train=[True, True, False],
                                                                    batch_size=batch_size)

        return train_loader, val_loader, test_loader

    else:
        # Create train and test sets
        train_dataset = MNIST(root=ds_path + "train", 
                              train=True,
                              transform=train_transform,
                              download=True)
        test_set = MNIST(root=ds_path + "test",
                         train=False,
                         transform=test_transform,
                         download=True)

        # Create train/test dataloaders
        train_loader, test_loader = create_data_loaders(train_set, test_set,
                                                        train=[True, False],
                                                        batch_size=batch_size)

        return train_loader, test_loader



def perturb(samples_batch: np.ndarray,
            rng: Any,
            sigma: float = 0.01,
            tau: float = 0.03,
            M: int = 291,
            restrict_M: bool = True):
    """
    Perturbing the augmented training data. See algorithm 2 in the PFGM paper
    (https://arxiv.org/pdf/2209.11178.pdf.). Found under models/utils_poisson.py
    on Github.
    
    Args:
    -----
        samples_batch: np.ndarray
            A batch of un-augmented training data.
        rng: np.random._generator.Generator
            rng needed for sampling the necessary hyperparameters.
        sigma: float
            Noise parameter. Specifically, it's the standard deviation of the 
            gaussian distribution that is sampled from to get the noise in x/y
            (eps_x/eps_y).
        tau: float
            Hyperparameter. Not sure what it really is.. :)
        M: float
            Measure how far out you go from the distribution. 
            Used to sample m, which is the exponent of (1 + \tau).
        restrict_M: bool
            Flag to allow confing the norms of the data to be....

    Returns:
    --------
        perturbed_samples_vec: np.ndarray
            The perturbed samples.
    """
    
    # Sample the exponents of (1+tau) from m ~ U[0,M], should be 1D
    m = rng.uniform(size=len(samples_batch), low=0, high=M)
    # Sample the noise parameter for the perturbed augmented data, z
    # Multiplying by sigma changes the variance of the gaussian but not the mean,
    # which is what Algorithm 2 dictates.
    eps_z = rng.standard_normal(size=(len(samples_batch), 1, 1, 1)) * sigma
    eps_z = np.abs(eps_z)

    # Confine the norms of perturbed data.
    # See Appendix B.1.1 of paper
    if restrict_M:
        idx = np.squeeze(eps_z < 0.005)
        num = int(np.sum(idx))
        restrict_m = int(M* 0.7)
        m[idx] = rng.uniform(size=(num,)) * restrict_m

    # data_dim = data_channels (1) * data_size (28) * data_size (28)
    data_dim = samples_batch.shape[1] * samples_batch.shape[2] * samples_batch.shape[3]
    factor = (1 + tau) ** m

    # Create the noise parameter for the perturbed data, x.
    eps_x = rng.standard_normal(size=(len(samples_batch), data_dim)) * sigma
    norm_eps_x = np.linalg.norm(eps_x, ord=2, axis=1) * factor
    # Perturb z 
    z = eps_z.squeeze() * factor 

    # Sample uniform angle over unit sphere (u in Eqn. 5)
    # gaussian = rng.standard_normal((samples_batch), data_dim)
    gaussian = rng.standard_normal(size=(len(samples_batch), data_dim))
    u = gaussian / np.linalg.norm(gaussian, ord=2, axis=1, keepdims=True)

    # Construct the perturbation for x
    perturbation_x = u * norm_eps_x[:, None]
    perturbation_x = np.reshape(perturbation_x, newshape=(samples_batch.shape))

    # Perturb x (ie. y in Algorithm 2)
    y = samples_batch + perturbation_x
    # Augment the data with extra dimension z
    perturbed_samples_vec = np.concatenate((y.reshape(len(samples_batch), -1), 
                                            z[:, None]), axis=1)
    return perturbed_samples_vec


def get_perturbed_img(perturbed_data: np.ndarray):
    """
    Function to get pertubed specified input image and reshape to correct form
    for plotting.

    Args:
    -----
        perturbed_data: np.ndarray
            The output of the 'perturb' function.

    Returns:
    --------
        img: np.ndarray
            Array containing the perturbed input image.
    """
    # Index specific sample and drop last dimension in second axis, due to that
    # being the augmented z axis.
    img = perturbed_data[:, :-1]
    img = img.reshape(-1, 28, 28)
    return img

def get_perturbed_z(perturbed_data: np.ndarray):
    """
    Function to get a specified samples augmented z value.

    Args:
    -----
        perturbed_data: np.ndarray
            The output of the 'perturb' function.

    Returns:
    --------
        z: np.ndarray
            Array containing the augmented z dimension.
    """
    z = perturbed_data[:, -1]
    return z

def get_perturbed(batch: np.ndarray,
                  rng: Any, 
                  sigma: float = 0.01,
                  tau: float = 0.03,
                  M: int = 291, 
                  restrict_M: bool = True):
    """
    Function that returns the three variables presented at the end of 
    Algorithm 2 in the paper. I.e.:

        y_tilde = (y, z) where:
            y_tilde = data
            y = img
            z = z

    Args:
    -----
        batch: np.ndarray
            Batch of data to perturb for NN training.
        rng: np.random._generator.Generator
            rng number used for sampling the hyperparameters to perturb
            the data to be used in the NN.
        M: int
            M hyperparameter found in Algorithm 2.

    Returns:
    --------
        img: np.ndarray
            Array containing the perturbed input image.    
        z: np.ndarray
            Array containing the augmented z dimension.
        data: np.ndarray
            The perturbed samples of shape (batch_size, img_size*img_size+1)
    """
    data = perturb(batch, rng, sigma=sigma, tau=tau, M=M, restrict_M=restrict_M)
    img = get_perturbed_img(data)
    z = get_perturbed_z(data)
    return (img, z), data