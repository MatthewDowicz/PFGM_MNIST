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
  ax.imshow(img[..., 0], cmap='viridis')
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
    _, y_tilde, _ = mkds.get_perturbed(batch, rng=rng, tau=tau, sigma=sigma, M=M)
    perturbed_img = y_tilde[0]

    fig, ax = plt.subplots(1, 2, figsize=(10,6))
    ax[0].imshow(batch[sample_idx])
    ax[0].set(title='Unperturbed Sample ($\\tau=0$, $\sigma=0$, $M=0$)')
    ax[0].axis('off')
    ax[1].imshow(perturbed_img[sample_idx])
    ax[1].set_title(f'Perturbed Sample ($\\tau=${tau}, $\sigma=${sigma}, $M=${M})')
    ax[1].axis('off')
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

    _, y_tilde, _ = mkds.get_perturbed(batch, rng=rng, tau=tau, sigma=sigma, M=M)
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


    y, E = mkds.empirical_field(batch, rng)
    x, _, _ = mkds.get_perturbed(batch,rng)
    
    x_coord = x_coord
    y_coord = y_coord
    flat_coord = x_coord * x.shape[2] + y_coord

    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].imshow(x[sample_idx], label='Example of Input image')
    ax[0].scatter(x_coord, y_coord, color='red', s=40, label='Selected pixel')
    ax[0].set(title='Example image of selected input pixel')
    # ax[0].axis('off')
    ax[0].legend()
    ax[1].quiver(y[:, flat_coord], y[:,-1], E[:, flat_coord], E[:,-1], label='Poisson Field of other samples');
    # This quiver plot highlights the selected pixel in the left panel. Allowing the audience to see how 
    # this one pixel is mapped into the N+1 dimension above the hyperplane.
    ax[1].quiver(y[sample_idx, flat_coord], y[sample_idx,-1], E[sample_idx, flat_coord], E[sample_idx,-1], color='red', alpha=0.7, label='This pixels Poisson Field');
    ax[1].set_title(f'Poisson field for pixel {x_coord, y_coord}')
    ax[1].set_xlabel(f'Perturbed value of pixel {x_coord, y_coord}')
    ax[1].set_ylabel(f'Z value of pixel {x_coord, y_coord}')

    if set_lims == True:
        ax[1].set_xlim(-2, 2)
        ax[1].set_ylim(0, 2)
    else:
        pass
    ax[1].legend()
    plt.show()

def empirical_field_grid(batch: np.ndarray,
                         rng: Any,
                         grid_lims: int, 
                         x_coord: int, 
                         y_coord: int):
    """
    Function to be used in visualizations of the empirical field. Due to 
    the MNIST samples being concentrated at the origin once perturbed it's 
    difficult to see what is going on. Hence, we use this function which is
    a modified version of the empirical_field function found in make_dataset.py
    that calculates the empirical field being produced by the MNIST dataset 
    over a uniform grid of points. 

    This uniform grid of points is indexed at the chosen pixel of interest
    which is specified by the (x_coord, y_coord) variables. 

    Args:
    -----
        batch: np.ndarray
            Batch of unperturbed sample data.
        rng: np.random.generator_Generator
            rng used for calculating the perturbing hyperparameters (tau/m,sigma)
        range: int
            Value that sets the limits of the uniform grid of points.
        x_coord: int
            X-coordinate of the pixel of interest
        y_coord: int
            Y-coordinate of the pixel of interest
        
    Returns:
    --------
        target: np.ndarray
            Array containing the empirical field for every pixel (ie. data point).
        perturbed_samples_vec: np.ndarray
            The perturbed input batch with the extra augmented dimension included.
    """
    # This is used to check that a square grid can be constructed from the specified
    # number of samples in the batch. 
    if np.sqrt(len(batch)) % 2 == 0: # check if even
        pass 
    elif np.sqrt(len(batch)) % 1 == 0: # Check if whole number
        pass
    else:
        AssertionError, print('Batchsize is an invalid shape for creating a uniform square grid.')
                        # Valid shapes have the square root of the length of the Batchsize
                        # being either an even number or a whole number     
        
    # Create the values to be used in the grid
    xv = np.linspace(-grid_lims/2, grid_lims/2, int(np.sqrt(len(batch))))
    zv = np.linspace(0, grid_lims, int(np.sqrt(len(batch))))

    x_bar, z_bar = np.meshgrid(xv, zv)
    # Flatten arrays to have the same shape as the rest of the perturbed data
    x_bar = x_bar.flatten()
    z_bar = z_bar.flatten()
    # Translate the two chosen coordinates ie. (x,y) to a 
    # flattened coord that can index the perturbed data array
    chosen_pixel = x_coord * batch.shape[2] + y_coord

    # Create unperturbed batch data vec with an un-augmented extra dimension
    # (i.e. append an extra dimension to the pixel data with the last dimension being 0)
    z = np.zeros(len(batch))

    # z[:, None] to create a 2D array with nothing in the 2nd dim b/c it's about to be concatenated
    unperturbed_samples_vec = np.concatenate((batch.reshape(len(batch), -1), 
                                            z[:, None]), axis=1)

    # Perturb the (augmented) batch data
    perturbed_samples_vec = mkds.perturb(samples_batch=batch,
                                        rng=rng)
    # Update the chosen_pixel to the uniform 1D grid
    perturbed_samples_vec[:, chosen_pixel] = x_bar
    # Update the augmented N+1 dimension to also be a uniform 1D grid
    perturbed_samples_vec[:, -1] = z_bar

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

def visualize_field_grid(batch: np.ndarray, 
                         sample_idx: int, 
                         x_coord: int, 
                         y_coord: int, 
                         rng: Any,
                         grid_lims: int):
    """

    Function that visualizes the empirical Poisson field created by a subset
    of the data distribution (ie. the Poisson field created from the samples
    in the batch) over a uniform grid of points. The visualization takes a 2D
    slice through the N+1 dimensions that PFGM needs to learning the mapping
    input distribution to target distribution.
    
    The chosen dimensions that make up the slice is the z-dimension, which is
    the extra dimension that PFGM needs to add avoid mode collapse and a 
    user specified dimension (ie. pixel) from the target distribution (eg. 
    a pixel in MNIST). With these two dimensions we can see the vector field
    of the empirical Poisson field created by the target distribution.
    
    (INCLUDE A BETTER EXPLANATION) 

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

    y, E = empirical_field_grid(batch=batch, 
                               rng=rng, 
                               grid_lims=grid_lims, 
                               x_coord=x_coord, 
                               y_coord=y_coord)
    
    flat_coord = x_coord * batch.shape[2] + y_coord

    # Get all the samples of the given batch EXCEPT for the element of the sample
    # that is being plotted in the left plot
    everything_except = np.arange(len(y)) != sample_idx

    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].imshow(batch[sample_idx], label='Example of Input image')
    ax[0].scatter(x_coord, y_coord, color='red', s=40, label='Selected pixel')
    ax[0].set(title='Example image of selected input pixel')
    ax[0].axis('off')
    ax[0].legend()
    ax[1].quiver(y[everything_except, flat_coord], y[everything_except,-1],
                 E[everything_except, flat_coord], E[everything_except,-1], label='Poisson Field of other samples');
    # This quiver plot highlights the selected pixel in the left panel. Allowing the audience to see how 
    # this one pixel is mapped into the N+1 dimension above the hyperplane.
    ax[1].quiver(y[sample_idx, flat_coord], y[sample_idx,-1],
                 E[sample_idx, flat_coord], E[sample_idx,-1], color='red', alpha=0.7, label='This pixels Poisson Field');
    ax[1].set_title(f'Poisson field for pixel {x_coord, y_coord}')
    ax[1].set_xlabel(f'Perturbed value of pixel {x_coord, y_coord}')
    ax[1].set_ylabel(f'Z value of pixel {x_coord, y_coord}')
    ax[1].legend()
    plt.show()