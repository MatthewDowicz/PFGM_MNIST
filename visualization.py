import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

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

    show_img_grid([imgs[idx] for idx in range(25)],
              [f'label={lbls[idx]}' for idx in range(25)])