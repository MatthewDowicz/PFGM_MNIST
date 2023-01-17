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
    Creates data loaders used in JAX for a set of datasets.

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
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          custom_transform])

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