import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import os
import jax
import jax.numpy as jnp
from jax import random
import pickle

from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union


def custom_transform(img):
    # Input: (28, 28) uint8 [0, 255] torch.Tensor, Output: (28, 28, 1) float32 [0, 1] np array
    return np.expand_dims(np.array(img, dtype=np.float32), axis=2) / 255.

def cifar_transform(img):
    return np.array(img, dtype=np.float32) / 255.

def save_data(data: Any, directory: str, filename: str):
    """
    Function to save the 'data' you want to save, to the 'directory' you want,
    with the 'filename' you want to save it as.
    
    Args:
    -----
        data: Any
            The data you want to save e.g. CSV, np.ndarray, etc.
        directory: str
            The string representing the path to the directory where 
            you want to save the data.
            
            For example, 'directory = "./data/train"' would specify
            that you want to save the data in the 'train' directory
            within the current working directory.
        filename: str
            The string that specifies the name of the file to be saved
            
    Returns:
    --------
        The saved data to the specified directory.
    """
    # Check if the directory eexists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filepath = os.path.join(directory, filename)
    # Save the data into the specified file in the directory
    if type(data) == np.ndarray: # use 'np.save' if data is an numpy array
        np.save(filepath, data)
    else:
        # Open a file to save the data
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
def load_data(data_dir: str, data_file: str):
    """
    Loads data from a specified file in a specified directory.
    If the directory doesn't exist, raises an informative error.
    If the data is a numpy array, uses np.load for optimal loading.

    Parameters:
    ------------
        data_dir: str
            Path to directory containing data file
        data_file: str
            Name of data file to load

    Returns:
    --------
        data: loaded data
    """

    # Check if directory exists
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist.")

    # Load data using optimal method for numpy arrays
    data_path = os.path.join(data_dir, data_file)
    if data_file.endswith('.npy') or data_file.endswith('.npz'):
        data = np.load(data_path)
    else:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

    return data

def download_MNIST(root_dir: str = 'saved_data/',
                   download: bool = True,
                   dataset: Any = MNIST,
                   transform: Callable = custom_transform):
    """
    Download the raw MNIST train/test data.
    
    NOTE: Need to update this code to allow for a lot of other datasets to be downloaded
          (e.g. Cifar10). Just need to change the name of the function and some of the documentation.
    
    Args:
    -----
        root_dir: str
            Path to where the raw data should be saved.
        download: bool
            If True, downloads the dataset from 'train-images-idx3-ubyte',
            otherwise from 't10k-images-idx3-ubyte'.
        dataset: Any
            The dataset that comes pre-installed in PyTorch
        transform: callable 
            Function/transform that takes in an image and returns a transformed
            version
            
    Returns:
    --------
        train_dataset: Any
            MNIST training images/labels pairs
        test_dataset: Any
            MNIST testing images/labels pairs.
    """
    # Create the root directory if it doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    # Pass the root directory to the MNIST dataset function
    train_dataset = dataset(root=root_dir,
                            train=True,
                            transform=transform,
                            download=download)
    test_dataset = dataset(root=root_dir,
                           train=False,
                           transform=transform,
                           download=download)
    
    return train_dataset, test_dataset

def partition_MNIST(root_dir: str = 'saved_data/',
                   download: bool = True,
                   dataset: Any = MNIST,
                   transform: Callable = custom_transform,
                   val_on: bool = True):
    """
    Function to partition the raw training/test data into training, validation,
    and test datasets. The split will be 50K, 10K, 10K, where the validation set
    will be a random sampling without replacement from the raw training set.
    
    The data is not downloaded, because these partitioned sets will be immediately
    passed to either a dataloader or a new custom dataset object.
    
    Args:
    -----
        root_dir: str
            Path to where the raw data should be saved.
        download: bool
            If True, downloads the dataset from 'train-images-idx3-ubyte',
            otherwise from 't10k-images-idx3-ubyte'.
        dataset: Any
            The dataset that comes pre-installed in PyTorch
        transform: callable 
            Function/transform that takes in an image and returns a transformed
            version.
        val_on: bool
            If True, paritions the raw MNIST training dataset into a two separate
            datasets, i.e a training set consisting of 50,000 samples/labels and a
            validation set consisting of 10,000 samples/labels., otherwise just passes
            the raw training/testing datasets.
            
    Returns:
    --------
        train_dataset: Any
            Partitioned MNIST training images/label pairs.
        validation_dataset: Any
            MNIST validation image/label pairs. This dataset was partitioned from
            the raw 60K training dataset.
        test_dataset: Any
            MNIST testing image/label pairs. This dataset is unchanged i.e. there
            is no partitioning done on this dataset.
    """
    # Download/instiate the raw MNIST data
    training_dataset, test_dataset = download_MNIST(root_dir = root_dir,
                                                    download = download,
                                                    dataset = dataset,
                                                    transform = transform)
    
    # Instantiate the seed we'll use for the random (w/o replacement) for the train/val set partitioning
    partition_gen = torch.Generator().manual_seed(42)
    
    # If we want to test with other datasets (e.g. Cifar10) can create if/else statements
    # within the 'if val_on' statement where we just make a check for the dataset we are
    # wanting to partition.
    if val_on:
        # Randomly splitting (with the same seed) the training dataset into a training/validation
        # sets. 
        train_set, _ = data.random_split(training_dataset, [50000, 10000], generator=partition_gen)
        _, val_set = data.random_split(training_dataset, [50000, 10000], generator=partition_gen)

        return train_set, val_set, test_dataset
    
    else:
        return train_set, test_dataset

def reshape_with_channel_dim(arr: np.ndarray):
    """
    Helper function to add a channel dimension to the MNIST dataset.
    It takes the input array ('arr') that we want to add the new dimension to
    
    Args:
    -----
        arr: np.ndarray
            The data array we want to add the channel dimension to
            
    Returns:
    --------
        Returns the reshaped data with a new channel dimension at axis 1
    """
    n_samples = arr.shape[0]
    original_shape = arr.shape[1:]
    new_shape = (n_samples, 1) + original_shape
    return np.reshape(arr, new_shape)


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
                        seed: int = 32):
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
                      root_dir: str = 'saved_data/', 
                      val_on: bool = True,
                      download: bool = True,
                      dataset: Any = MNIST,
                      transform: Callable = custom_transform):
    """
    Function to load the created dataloaders.

    Args:
    -----
        batch_size: int
            The number of samples per batch to load.
        root_dir: str
            Path where the raw datasets are saved.
        val_on: bool
            If True, paritions the raw MNIST training dataset into a two separate
            datasets, i.e a training set consisting of 50,000 samples/labels and a
            validation set consisting of 10,000 samples/labels., otherwise just passes
            the raw training/testing datasets.
        download: bool
            If True, downloads the dataset from 'train-images-idx3-ubyte',
            otherwise from 't10k-images-idx3-ubyte'.
        dataset: Any
            The dataset that comes pre-installed in PyTorch
        transform: callable 
            Function/transform that takes in an image and returns a transformed
            version.
    """
    # Converting a uint8 [0, 255] torch.Tensor to float32 [0,1] np.array
    test_transform = transform
    train_transform = transform    
    
    if val_on:
        train_set, val_set, test_set = partition_MNIST(root_dir=root_dir,
                                                       download=download,
                                                       dataset=dataset,
                                                       transform=transform,
                                                       val_on=val_on)
        # Create the train/val/test data loaders
        train_loader, val_loader, test_loader = create_data_loaders(train_set, val_set, test_set,
                                                                    train=[True, True, False],
                                                                    batch_size=batch_size)

        return train_loader, val_loader, test_loader
    
    
    elif not val_on:
        train_set, test_set = partition_MNIST(root_dir=root_dir,
                                               download=download,
                                               dataset=dataset,
                                               transform=transform,
                                               val_on=val_on)
        # Create train/test dataloaders
        train_loader, test_loader = create_data_loaders(train_set, test_set,
                                                        train=[True, False],
                                                        batch_size=batch_size)

        return train_loader, test_loader
    
    else:
        pass
# def load_data_loaders(batch_size: int = 128,
#                       ds_path: str = 'saved_data/', 
#                       val_on: bool = True,
#                       download: bool = True,
#                       dataset: Any = MNIST,
#                       data_transform: Any = custom_transform):
#     """
#     Function to load the created dataloaders.

#     Args:
#     -----
#         ds_path: str
#             Path where the datasets should be saved
#         val_on: bool
#             Toggle to decide if we want a validation set or just train/test sets.
#     """

#     # Converting a uint8 [0, 255] torch.Tensor to float32 [0,1] np.array
#     test_transform = data_transform
#     # For training, we add some augmentation to reduce overfitting.
#     train_transform = data_transform

#     if val_on:
#         # Loading the training dataset. Because val_on = True we need to split it into
#         # training and validation sets. We also need to do a little trick because the
#         # validation set should not use the augmentation (ie. having same behavior as
#         # the test set).
#         train_dataset = dataset(root=ds_path + "train", 
#                               train=True,
#                               transform=train_transform,
#                               download=download)
#         val_dataset = dataset(root=ds_path + "val",
#                             train=True,
#                             transform=test_transform,
#                             download=download)

#         if dataset == MNIST:
#             # Randomly splitting (with the same seed) the training/validation training sets and then only saving the
#             # respective datasets for each one. I.e. the training set gets 50,000 samples, while the val set gets 10,000.
#             train_set, _ = data.random_split(train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))
#             _ , val_set = data.random_split(val_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))

#         elif dataset == CIFAR10:
#             # Randomly splitting (with the same seed) the training/validation training sets and then only saving the
#             # respective datasets for each one. I.e. the training set gets 50,000 samples, while the val set gets 10,000.
#             train_set, _ = data.random_split(train_dataset, [40000, 10000], generator=torch.Generator().manual_seed(42))
#             _ , val_set = data.random_split(val_dataset, [40000, 10000], generator=torch.Generator().manual_seed(42))
#         else:
#             pass

#         # Loading the test set
#         test_set = dataset(root=ds_path + "test",
#                          train=False,
#                          transform=test_transform,
#                          download=download)

#         # Create the train/val/test data loaders
#         train_loader, val_loader, test_loader = create_data_loaders(train_set, val_set, test_set,
#                                                                     train=[True, True, False],
#                                                                     batch_size=batch_size)

#         return train_loader, val_loader, test_loader

#     else:
#         # Create train and test sets
#         train_set = dataset(root=ds_path + "train", 
#                               train=True,
#                               transform=train_transform,
#                               download=download)
#         test_set = dataset(root=ds_path + "test",
#                          train=False,
#                          transform=test_transform,
#                          download=download)

#         # Create train/test dataloaders
#         train_loader, test_loader = create_data_loaders(train_set, test_set,
#                                                         train=[True, False],
#                                                         batch_size=batch_size)

#         return train_loader, test_loader



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

def jax_perturb(samples_batch: np.ndarray,
            prng: Any,
            sigma: float = 0.05,
            tau: float = 0.05,
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
    # Splitting of keys to allow for PRNG.
    # rng, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(prng, num=7)
    
    # Sample the exponents of (1+tau) from m ~ U[0,M], should be 1D
    # m = random.uniform(subkey2, shape=(len(samples_batch),1), minval=0, maxval=M)
    m = random.uniform(prng, shape=(len(samples_batch),1), minval=0, maxval=M)

    
    # Sample the noise parameter for the perturbed augmented data, z
    # Multiplying by sigma changes the variance of the gaussian but not the mean,
    # which is what Algorithm 2 dictates.                       
    # eps_z = 0 + sigma * random.normal(subkey3, shape=(len(samples_batch), 1))
    eps_z = 0 + sigma * random.normal(prng, shape=(len(samples_batch), 1))
    eps_z = jnp.abs(eps_z)

    # Confine the norms of perturbed data.
    # See Appendix B.1.1 of paper
    if restrict_M:
        idx = jnp.squeeze(eps_z < 0.005)
        # num = int(jnp.sum(idx))
        num = jnp.array(jnp.sum(idx), int) # rewrite this as `x.astype(int)` or `jnp.array(x, int)` to work
        # restrict_m = int(M* 0.7)
        restrict_m = jnp.array(M*0.7, int)

        # m = m.at[idx, 0].set(random.uniform(subkey4, shape=(num,), minval=0, maxval=restrict_m))
        m = m.at[idx, 0].set(random.uniform(prng, shape=(num,), minval=0, maxval=restrict_m))

    # data_dim = data_channels (1) * data_size (28) * data_size (28)
    data_dim = samples_batch.shape[1] * samples_batch.shape[2] * samples_batch.shape[3]
    factor = (1 + tau) ** m

    # Create the noise parameter for the perturbed data, x.
    # eps_x = rng.standard_normal(size=(len(samples_batch), data_dim)) * sigma
    # norm_eps_x = np.linalg.norm(eps_x, ord=2, axis=1) * factor                  
    # eps_x = 0 + sigma * random.normal(subkey5, shape=(len(samples_batch), data_dim))
    eps_x = 0 + sigma * random.normal(prng, shape=(len(samples_batch), data_dim))

    norm_eps_x = jnp.linalg.norm(eps_x, ord=2, axis=1) * factor[:, 0]
                             
    # Perturb z 
    z = jnp.squeeze(eps_z) * factor[:,0]

    # Sample uniform angle over unit sphere (u in Eqn. 5)
    # gaussian = random.normal(subkey6, shape=(len(samples_batch), data_dim))
    gaussian = random.normal(prng, shape=(len(samples_batch), data_dim))
    unit_gaussian = gaussian / jnp.linalg.norm(gaussian, ord=2, axis=1, keepdims=True)

    # Construct the perturbation for x
    perturbation_x = unit_gaussian * norm_eps_x[:, None]
    perturbation_x = jnp.reshape(perturbation_x, newshape=(samples_batch.shape))
    # Perturb x (ie. y in Algorithm 2)
    y = samples_batch + perturbation_x

    # Augment the data with extra dimension z
    perturbed_samples_vec = np.concatenate((y.reshape(len(samples_batch), -1), 
                                            z[:,None]), axis=1)
    return perturbed_samples_vec


def get_perturbed_img(perturbed_data: np.ndarray, input_data_shape: Tuple):
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
    img = img.reshape(-1, input_data_shape[1], input_data_shape[2])
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
        x: np.ndarray
            Array containing the unperturbed samples
        y: tuple
            Tuple containing the perturbed data & the augmented z dimension.
            img: np.ndarray
                Array containing the perturbed input image.    
            z: np.ndarray
                Array containing the augmented z dimension.
        data: np.ndarray
            The perturbed samples of shape (batch_size, img_size*img_size+1)
    """
    data = perturb(batch, rng, sigma=sigma, tau=tau, M=M, restrict_M=restrict_M)
    img = get_perturbed_img(data, batch.shape)
    z = get_perturbed_z(data)
    y = (img, z)
    x = batch
    return x, y, data


def empirical_field(batch: np.ndarray, rng: Any):
    """
    Function to calculate the empirical (ie. seen) Poisson field.
    This function does the brute force calculation of what the field 
    looks like and its output are the "labels" for our supervised learning
    problem. This is the answer that we want the NN to learn to emulate.
    Found in losses.py under function 'get_loss_fn'

    Args:
    -----
        batch: np.ndarray
            Batch of unperturbed sample data.
        rng: np.random.generator_Generator
            rng used for calculating the perturbing hyperparameters (tau/m,sigma)
        
    Returns:
    --------
        target: np.ndarray
            Array containing the empirical field for every pixel (ie. data point).
    """
    # Create unperturbed batch data vec with an un-augmented extra dimension
    # (i.e. append an extra dimension to the pixel data with the last dimension being 0)
    z = np.zeros(len(batch))
    # z[:, None] to create a 2D array with nothing in the 2nd dim b/c it's about to be concatenated
    unperturbed_samples_vec = np.concatenate((batch.reshape(len(batch), -1), 
                                            z[:, None]), axis=1)

    # Perturb the (augmented) batch data
    perturbed_samples_vec = perturb(samples_batch=batch,
                                        rng=rng)
    # batch.shape[1/2] = img_size, batch.shape[3] = n_channels
    data_dim = batch.shape[1] * batch.shape[2] * batch.shape[3]

    # Get distance between the unperturbed vector on hyperplane (z=0) and their perturbed versions
    # Expand dims here, so that the 2nd dim of the array doesn't collapse
    # ie. make sure that gt_distance.shape = (batchsize, batchsize), which corresponds to a vector
    # in the N+1 dimension space per sample <-- MAKE THIS CLEARER
    gt_distance = np.sqrt(np.sum((np.expand_dims(perturbed_samples_vec, axis=1) - unperturbed_samples_vec) ** 2,
                                    axis=-1, keepdims=False))

    # For numerical stability, we multiply each row by its minimum value
    # keepdims=True, so we don't lose a dimension
    # Figure out why my code doesn't need a [0] in the numerator of the first distance var
    distance = np.min(gt_distance, axis=1, keepdims=True) / (gt_distance + 1e-7)
    distance = distance ** (data_dim + 1)
    distance = distance[:, :, None]


    # Normalize the coefficients (effectively multiply by c(x_tilde))
    # Expand dims again to avoid losing a dimension
    coeff = distance / (np.sum(distance, axis=1, keepdims=True) + 1e-7)
    diff = - ((np.expand_dims(perturbed_samples_vec, axis=1) - unperturbed_samples_vec))

    # Calculate the empirical Poisson Field (N+1 dimension in the augmented space)
    gt_direction = np.sum(coeff * diff, axis=1, keepdims=False)
    gt_norm = np.linalg.norm(gt_direction, axis=1)
    # Normalize 
    gt_direction /= np.reshape(gt_norm, (-1,1))
    gt_direction *= np.sqrt(data_dim)

    target = gt_direction
    return perturbed_samples_vec, target

def jax_empirical_field(batch: np.ndarray, 
                        prng: Any, 
                        sigma: float = 0.05,
                        tau: float = 0.05,
                        M: int = 291,
                        restrict_M: bool = True):
    """
    Function to calculate the empirical (ie. seen) Poisson field.
    This function does the brute force calculation of what the field 
    looks like and its output are the "labels" for our supervised learning
    problem. This is the answer that we want the NN to learn to emulate.
    Found in losses.py under function 'get_loss_fn'

    Args:
    -----
        batch: np.ndarray
            Batch of unperturbed sample data.
        rng: np.random.generator_Generator
            rng used for calculating the perturbing hyperparameters (tau/m,sigma)
        
    Returns:
    --------
        target: np.ndarray
            Array containing the empirical field for every pixel (ie. data point).
    """
    # Create unperturbed batch data vec with an un-augmented extra dimension
    # (i.e. append an extra dimension to the pixel data with the last dimension being 0)
    z = jnp.zeros(len(batch))
    # z[:, None] to create a 2D array with nothing in the 2nd dim b/c it's about to be concatenated
    unperturbed_samples_vec = jnp.concatenate((batch.reshape(len(batch), -1), 
                                            z[:, None]), axis=1)

    # Perturb the (augmented) batch data
    perturbed_samples_vec = jax_perturb(samples_batch=batch,
                                        prng=prng,
                                        sigma=sigma,
                                        tau=tau,
                                        M=M,
                                        restrict_M=restrict_M)
    # batch.shape[1/2] = img_size, batch.shape[3] = n_channels
    data_dim = batch.shape[1] * batch.shape[2] * batch.shape[3]

    # Get distance between the unperturbed vector on hyperplane (z=0) and their perturbed versions
    # Expand dims here, so that the 2nd dim of the array doesn't collapse
    # ie. make sure that gt_distance.shape = (batchsize, batchsize), which corresponds to a vector
    # in the N+1 dimension space per sample <-- MAKE THIS CLEARER
    gt_distance = jnp.sqrt(jnp.sum((jnp.expand_dims(perturbed_samples_vec, axis=1) - unperturbed_samples_vec) ** 2,
                                    axis=-1, keepdims=False))

    # For numerical stability, we multiply each row by its minimum value
    # keepdims=True, so we don't lose a dimension
    # Figure out why my code doesn't need a [0] in the numerator of the first distance var
    distance = jnp.min(gt_distance, axis=1, keepdims=True) / (gt_distance + 1e-7)
    distance = distance ** (data_dim + 1)
    distance = distance[:, :, None]


    # Normalize the coefficients (effectively multiply by c(x_tilde))
    # Expand dims again to avoid losing a dimension
    coeff = distance / (jnp.sum(distance, axis=1, keepdims=True) + 1e-7)
    diff = - ((jnp.expand_dims(perturbed_samples_vec, axis=1) - unperturbed_samples_vec))

    # Calculate the empirical Poisson Field (N+1 dimension in the augmented space)
    gt_direction = jnp.sum(coeff * diff, axis=1, keepdims=False)
    gt_norm = jnp.linalg.norm(gt_direction, axis=1)
    # Normalize 
    gt_direction /= jnp.reshape(gt_norm, (-1,1))
    gt_direction *= jnp.sqrt(data_dim)

    target = np.asarray(gt_direction)
    return perturbed_samples_vec, target

def process_perturbed_data(dataset: np.ndarray, prng: jax.random.PRNGKey,
                           sigma: float = 0.05,
                            tau: float = 0.05,
                            M: int = 291,
                            restrict_M: bool = True):
    """
    Function that perturbs the raw MNIST files (ie. train data file/test data file)
    and saves them into a tuple of (perturbed_data, empirical_field).
    
    It does this by calculating the number of passes needed to go through the entire
    dataset if we used a batchsize of 1000. The reason for the seemingly arbitrary choice
    of 1000, is that I'm running out of memory at when my batchsize is larger than 1000.
    
    Args:
    -----
        dataset: np.ndarray
            Either the raw MNIST training/testing dataset.
            NOTE: Dataset here refers to the pixel data only, we discard the labels because
                  in PFGM the NN learns the empirical field at each pixel location. One can
                  think of the empirical field at each pixel location as that pixels "label".
                  
        prng: jax.random.PRNGKey
            Source of randomness for generating random numbers  
            
    Returns:
    --------
        tuple of the perturbed data for each sample and the empirical field for each image
    """
    # Get the number of samples in the dataset
    num_samples = dataset.shape[0]
    # Calculate the number of passes needed to go through the entire dataset
    # in batches of 1000
    num_passes = int(np.ceil(num_samples / 1000))
    # Initialize empty list to store the output of the function
    outputs = []
    # Loop over the number passes needed to go through entire dataset
    for i in range(num_passes):
        # Calculate start and end indices for the current batch
        start_idx = i * 1000
        end_idx = min((i+1)*1000, num_samples)
        # Get the current batch from the dataset
        batch = dataset[start_idx:end_idx]
        # Get the perturbed data & empirical field for that batch
        output = jax_empirical_field(batch, prng, sigma=sigma, tau=tau, M=M, restrict_M=restrict_M)
        outputs.append(output)
        del output
    # Concatenate outputs along the first axis
    return tuple(np.concatenate(o, axis=0) for o in zip(*outputs))