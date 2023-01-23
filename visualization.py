import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import make_dataset as mkds

# Note: show_img & show_img_grid were taken from the flax MNIST tutorial
# found here (https://colab.research.google.com/github/google/flax/blob/main/examples/mnist/mnist.ipynb#scrollTo=7O2C7AY3p4ZF)

def show_img(img, ax=None, title=None):
  """Shows a single image."""
  if ax is None:
    ax = plt.gca()
  ax.imshow(img[..., 0], cmap='gray')
  ax.set_xticks([])
  ax.set_yticks([])
  if title:
    ax.set_title(title)

def show_img_grid(imgs, titles):
  """Shows a grid of images."""
  n = int(np.ceil(len(imgs)**.5))
  _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
  for i, (img, title) in enumerate(zip(imgs, titles)):
    show_img(img, axs[i // n][i % n], title)

def plot_MNIST(dataloader: data.DataLoader,
               num_samps: int = 25):

    # Note: If using the training dataloader some numbers will be
    # horizontally flipped due to the transforms applied during
    # dataloader creation.
    
    # Load in the dataloader, get a batch from the dataloader,
    # and instantiate the labels and images, respectively.
    dataset = next(iter(dataloader))
    imgs = dataset[0]
    lbls = dataset[1]

    show_img_grid([imgs[idx] for idx in range(num_samps)],
              [f'label={lbls[idx]}' for idx in range(num_samps)])


def visualize_single_perturbed(batch, sample_idx, rng, tau=0.01, sigma=0.03, M=291):
    """
    Function to visualize how Algorithm 2 of the paper effects an MNIST sample.

    Args:
    -----
        batch: np.ndarray   
            A batch of images of shape (batchsize, n_channels, img_size, img_size)
        sample_idx: int
            Index of the specific sample to plot.
        rng: np.random.generator_.Generator
            rng used in sampling the hyperparameters in Algorithm 2
        tau: float
            Hyperparameter. Not really clear on what it does :)
        sigma: float
            Noise parameter. Specifically, it's the standard deviation of the
            gaussian distribution that is sampled from to get the noise in for
            the data (y) and the extra dimension (z).
        M: int
            Measure of how far out you go from the distribution.

    Returns:
    -------
        A side by side plot of the input image and it's perturbed version
    """
    y_tilde, _ = mkds.get_perturbed(batch, rng=rng, tau=tau, sigma=sigma, M=M)
    perturbed_img = y_tilde[0]

    fig, ax = plt.subplots(1, 2, figsize=(10,6))
    ax[0].imshow(batch[sample_idx])
    ax[0].set(title='Unperturbed Sample')
    ax[1].imshow(perturbed_img[sample_idx])
    ax[1].set_title('Perturbed Sample')
    plt.show()

def perturb_scatter(batch, rng, tau=0.01, sigma=0.03, M=291):
    """
    Function to visualize how Algorithm 2 "lifts" the data off the hyperplane into the
    N+1 dimension. This function does this by using the X-value as the mean of the each
    MNIST sample (might not be the best idea because most of the MNIST images are black,
    so the means are roughly 0, as can be seen by the small scatter on x).

    Args:
    -----
        batch: np.ndarray   
            A batch of images of shape (batchsize, n_channels, img_size, img_size)
        sample_idx: int
            Index of the specific sample to plot.
        rng: np.random.generator_.Generator
            rng used in sampling the hyperparameters in Algorithm 2
        tau: float
            Hyperparameter. Not really clear on what it does :)
        sigma: float
            Noise parameter. Specifically, it's the standard deviation of the
            gaussian distribution that is sampled from to get the noise in for
            the data (y) and the extra dimension (z).
        M: int
            Measure of how far out you go from the distribution.

    Returns:
    -------
        A scatter plot showing how Algorithm 2 lifts the data into the N+1 dimension
        compared to the original data. 

        NOTE: To create a data point it's we take the mean of the image/perturbed image
              as the x-coordinate and use the z-values as the y-coordinate, thus:
                data point = (mean(image), z) 

    """

    y_tilde, _ = mkds.get_perturbed(batch, rng=rng, tau=tau, sigma=sigma, M=M)
    perturbed_img = y_tilde[0]
    z = y_tilde[1]

    plt.scatter(np.mean(batch, axis=(1,2,3)), np.zeros(len(batch)), label='Original samples')
    plt.scatter(np.mean(perturbed_img, axis=(1,2)), z, label='Perturbed samples')
    plt.title('Visualization of data being lifted into $N+1$')
    plt.xlabel('X-values (values on hyperplane $z=0$)')
    plt.ylabel('Z-values (values lifted into $N+1$ dimension)')
    plt.legend()
    plt.show()

def visualize_field(batch: np.ndarray, 
                    sample_idx: int, 
                    x_coord: int, 
                    y_coord: int, 
                    rng: Any, 
                    set_lims: bool = False):
    """
    Function that visualizes the E field for a user specified pixel, as well as,
    the specified pixel superimposed over a specific input image sample.

    Args:
    -----
        batch: np.ndarray
            Batch of input data
        sample_idx: int
            Index of the specified sample found within the batch
        x_coord: int
            X-coordinate of the pixel of interest
        y_coord: int
            Y-coordinate of the pixel of interest
        rng: np.random.generator_.Generator
            rng used for calculating the perturbing hyperparameters
        set_lims: bool
            Option to focus on a narrower x-range on the quiver plot.

    Returns:
    --------
        A two panel plot with:
            Left plot showing a sample image with a red circle
            overlaid on the chosen pixel who's E field will be
            plotted on the right (quiver) plot.

            Right plot is a quiver plot depicting the E field calculated 
            at the specified pixel value. The x/y coordinates for the points
            in the quiver plot are the (pixel_val, z_value) in the perturbed
            image and the (x/y) coordinates of the arrows are 
            (E_field @ pixel_coord, E_field & z_value)
    """


    E = mkds.empirical_field(batch, rng)
    x, y, data = mkds.get_perturbed(batch,rng)
    
    x_coord = x_coord
    y_coord = y_coord
    flat_coord = x_coord * x.shape[2] + y_coord

    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].imshow(x[sample_idx], label='Example of Input image')
    ax[0].scatter(x_coord, y_coord, color='red', s=40, label='Selected Pixel for every sample')
    ax[0].set(title='Input pixel')
    ax[0].legend()
    ax[1].quiver(data[:, flat_coord], data[:,-1], E[:, flat_coord], E[:,-1], label='True Poisson Field');
    ax[1].set_title(f'Poisson field for pixel {x_coord, y_coord}')
    ax[1].set_xlabel(f'Pixel value of pixel {x_coord, y_coord}')
    ax[1].set_ylabel(f'Z value of pixel {x_coord, y_coord}')

    if set_lims == True:
        ax[1].set_xlim(-10, 10)
        ax[1].set_ylim(0, 2)
    else:
        pass
    ax[1].legend()
    plt.show()