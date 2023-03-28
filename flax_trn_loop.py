import numpy as np
from numpy.random import default_rng

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)
import optax 
import make_dataset as mkds
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
from tqdm.auto import tqdm

import os 

# Wandb 
import wandb
wandb.login()
import pprint


class MLP(nn.Module):
    """
    Simple MLP model for testing PFGM.

    Due to it's simplicity we use @nn.compact instead of setup
    """
    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x, **kwargs):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.silu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

def init_train_state(model: Any,
                     random_key: Any,
                     shape: tuple,
                     learning_rate: int) -> train_state.TrainState:
    """
    Function to initialize the TrainState dataclass, which represents
    the entire training state, including step number, parameters, and 
    optimizer state. This is useful because we no longer need to
    initialize the model again and again with new variables, we just 
    update the "state" of the mdoel and pass this as inputs to functions.

    Args:
    -----
        model: nn.Module    
            The model that we want to train.
        random_key: jax.random.PRNGKey()
            Used to trigger the initialization functions, which generate
            the initial set of parameters that the model will use.
        shape: tuple
            Shape of the batch of data that will be input into the model.
            This is used to trigger shape inference, which is where the model
            figures out by itself what the correct size the weights should be
            when they see the inputs.
        learning_rate: int
            How large of a step the optimizer should take.

    Returns:
    --------
        train_state.TrainState:
            A utility class for handling parameter and gradient updates. 
    """
    # Initialize the model
    variables = model.init(random_key, jnp.ones(shape))

    # Create the optimizer
    optimizer = optax.adam(learning_rate) # TODO update this to be user defined

    # Create a state
    return train_state.TrainState.create(apply_fn=model.apply,
                                         tx=optimizer,
                                         params=variables['params'])

def compute_metrics(*, pred, labels):
    """
    Function that computes metrics that will be logged
    during training
    """
    # Calculate the MSE loss
    loss = ((pred - labels) ** 2).mean()

    # Calculate the R^2 score
    residual = jnp.sum(jnp.square(labels - pred))
    total = jnp.sum(jnp.square(labels - jnp.mean(labels)))
    r2_score = 1 - (residual / total)

    # Save these metrics into a dict
    metrics = {
        'loss': loss,
        'r2': r2_score
    }

    return metrics

def accumulate_metrics(metrics):
    """
    Function that accumulates all the metrics for each batch and 
    accumulates/calculates the metrics for each epoch.
    """
    metrics = jax.device_get(metrics)
    return {
        k: np.mean([metric[k] for metric in metrics])
        for k in metrics[0]
    }

@jax.jit
def train_step(state: train_state.TrainState,
               batch: list):
    """
    Function to run training on one batch of data.
    """
    image, label = batch

    def loss_fn(params: dict):
        """
        Simple MSE loss as described in the PFGM paper.
        """
        pred = state.apply_fn({'params': params}, image)
        loss = ((pred - label) ** 2).mean()
        return loss, pred

    def r_squared(params):
        """
        Function to calculate the coefficient of determination or 
        R^2, which quantifies how well the regression model fits 
        the observed data. Or more formally, it is a statistical
        measure that represents the proportion of variance in the
        dependent variable that is explained by the independent 
        variable(s) in a regression model. R^2 ranges from 0 to 1, 
        with a higher value indicating a better fit. 

        An R^2 of 0 means that the regression model does not explain
        any of the variability in the dependent variable, while an
        R^2 of 1 indicates that the regression model explains all of
        the variability in the dependent model.
        """
        pred = state.apply_fn({'params': params}, image)
        residual = jnp.sum(jnp.square(label - pred))
        total = jnp.sum(jnp.square(label - jnp.mean(label)))
        r2_score = 1 - (residual / total)
        return r2_score

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, pred), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(pred=pred, labels=label)
    return state, metrics

@jax.jit
def eval_step(state, batch):
    image, label = batch
    pred = state.apply_fn({'params': state.params}, image)
    return compute_metrics(pred=pred, labels=label)

def save_checkpoint_wandb(ckpt_path, state, epoch):
    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))
    artifact = wandb.Artifact(
        f'{wandb.run.name}-checkpoint', type='dataset'
    )
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact, aliases=["latest", f"epoch_{epoch}"])
    
def load_checkpoint_wandb(ckpt_file, state):
    artifact = wandb.use_artifact(
        f'{wandb.run.name}-checkpoint:latest'
    )
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, ckpt_file)
    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()
    return from_bytes(state, byte_data)


def train_and_evaluate(batchsize, state, epochs, np_rng):    

    # Instantiate the data loaders
    train_dl, val_dl, test_dl = mkds.load_dataloaders(batch_size=batchsize, download=False)

    for epoch in tqdm(range(1, epochs+1)):
        best_val_loss = 1e6

        # =========== Training =========== #
        train_batch_metrics = []
        for cnt, batch in enumerate(train_dl):
            # Instantiate the imgs
            imgs, _ = batch
            # Perturb the data
            perturbed_batch = mkds.empirical_field(imgs, np_rng)
            # Do one train step with perturbed data
            state, metrics = train_step(state, perturbed_batch)
            train_batch_metrics.append(metrics)
        train_batch_metrics = accumulate_metrics(train_batch_metrics)
        print(
            'TRAIN (%d/%d): Loss: %.4f, r2: %.2f' % (
                epoch, epochs, train_batch_metrics['loss'], 
                train_batch_metrics['r2'])
        )
                
        # =========== Validation =========== #
        val_batch_metrics = []
        for cnt, batch in enumerate(val_dl):
            # Instantiate the imgs
            imgs, _ = batch
            # Perturb the data
            perturbed_batch = mkds.empirical_field(imgs, np_rng)
            metrics = eval_step(state, perturbed_batch)
            val_batch_metrics.append(metrics)
        val_batch_metrics = accumulate_metrics(val_batch_metrics)
        print(
            'Val (%d/%d): Loss: %.4f, r2: %.2f' % (
                epoch, epochs, val_batch_metrics['loss'], 
                val_batch_metrics['r2'])
        )
        
        wandb.log({
            "Train Loss": train_batch_metrics['loss'],
            "Train r2": train_batch_metrics['r2'],
            "Validation Loss": val_batch_metrics['loss'],
            "Validation r2": val_batch_metrics['r2']
        }, step=epoch)
        
        if val_batch_metrics['loss'] < best_val_loss:
            save_checkpoint_wandb("checkpoint.msgpack", state, epoch)
            
    restored_state = load_checkpoint_wandb("checkpoint.msgpack", state)
    test_batch_metrics = []
    for cnt, batch in enumerate(test_dl):
            # Instantiate the imgs
            imgs, _ = batch
            # Perturb the data
            perturbed_batch = mkds.empirical_field(imgs, np_rng)
            metrics = eval_step(state, perturbed_batch)
            test_batch_metrics.append(metrics)
        
    test_batch_metrics = accumulate_metrics(test_batch_metrics)
    print(
        'Test: Loss: %.4f, r2: %.2f' % (
            test_batch_metrics['loss'],
            test_batch_metrics['r2']
        )
    )
    
    wandb.log({
        "Test Loss": test_batch_metrics['loss'],
        "Test r2": test_batch_metrics['r2']
    })

    return state, restored_state

def train_and_evaluate_sweep(config=None):    
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        # Instantiate the data loaders
        train_dl, val_dl, test_dl = mkds.load_data_loaders(batch_size=config.batch_size, download=False)

        # Instantiate the model
        mlp = MLP(hidden_dims=[config.hidden_dims, config.hidden_dims, config.hidden_dims, config.hidden_dims, config.hidden_dims],
              output_dim=785)

        # Instantiate the TrainState to be passed down
        state = init_train_state(model=mlp, 
                             random_key=jax.random.PRNGKey(config.jax_rng), 
                             shape=(1, 785), 
                             learning_rate=config.learning_rate)

        np_rng = default_rng(seed=config.np_rng)

        for epoch in range(1, config.epochs+1):
            best_val_loss = 1e6

            # =========== Training =========== #
            train_batch_metrics = []
            for cnt, batch in enumerate(train_dl):
                # Instantiate the imgs
                imgs, _ = batch
                # Perturb the data
                perturbed_batch = mkds.empirical_field(imgs, np_rng)
                # Do one train step with perturbed data
                state, metrics = train_step(state, perturbed_batch)
                train_batch_metrics.append(metrics)
            train_batch_metrics = accumulate_metrics(train_batch_metrics)
            print(
                'TRAIN (%d/%d): Loss: %.4f, r2: %.2f' % (
                    epoch, config.epochs, train_batch_metrics['loss'], 
                    train_batch_metrics['r2'])
            )

            # =========== Validation =========== #
            val_batch_metrics = []
            for cnt, batch in enumerate(val_dl):
                # Instantiate the imgs
                imgs, _ = batch
                # Perturb the data
                perturbed_batch = mkds.empirical_field(imgs, np_rng)
                metrics = eval_step(state, perturbed_batch)
                val_batch_metrics.append(metrics)
            val_batch_metrics = accumulate_metrics(val_batch_metrics)
            print(
                'Val (%d/%d): Loss: %.4f, r2: %.2f' % (
                    epoch, config.epochs, val_batch_metrics['loss'], 
                    val_batch_metrics['r2'])
            )

            wandb.log({
                "Train Loss": train_batch_metrics['loss'],
                "Train r2": train_batch_metrics['r2'],
                "Validation Loss": val_batch_metrics['loss'],
                "Validation r2": val_batch_metrics['r2']
            }, step=epoch)

            if val_batch_metrics['loss'] < best_val_loss:
                save_checkpoint_wandb("checkpoint.msgpack", state,epoch)

        restored_state = load_checkpoint_wandb("checkpoint.msgpack", state)
        test_batch_metrics = []
        for cnt, batch in enumerate(test_dl):
                # Instantiate the imgs
                imgs, _ = batch
                # Perturb the data
                perturbed_batch = mkds.empirical_field(imgs, np_rng)
                metrics = eval_step(state, perturbed_batch)
                test_batch_metrics.append(metrics)

        test_batch_metrics = accumulate_metrics(test_batch_metrics)
        print(
            'Test: Loss: %.4f, r2: %.2f' % (
                test_batch_metrics['loss'],
                test_batch_metrics['r2']
            )
        )

        wandb.log({
            "Test Loss": test_batch_metrics['loss'],
            "Test r2": test_batch_metrics['r2']
        })


