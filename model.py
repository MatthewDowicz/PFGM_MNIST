# Standard libraries
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union

import jax 
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints

import numpy as np
from training_module import TrainerModule
import make_dataset as mkds

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int

    @ nn.compact
    def __call__(self, x, **kwargs):
        # y = jnp.concatenate((x.reshape(len(x), -1), t[:, None]), axis=1)
        # Might need to change how this works for doing likelihood estimation
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.silu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class MLPTrainer(TrainerModule):

    def __init__(self,
                 hidden_dims: Sequence[int],
                 output_dim: int,
                 np_rng: Any,
                 trial: Any = None,
                 **kwargs):
        super().__init__(model_class=MLP,
                         model_hparams={
                            'hidden_dims': hidden_dims,
                            'output_dim': output_dim},
                         **kwargs)
        self.np_rng = np_rng

    def create_functions(self):
        def mse_loss(params, batch, np_rng):
            img, _ = batch
            img, lbl = mkds.empirical_field(img, np_rng)
            pred = self.model.apply({'params': params}, img)
            loss = ((pred - lbl)**2).mean()
            return loss

        def train_step(state, batch):
            loss_fn = lambda params: mse_loss(params, batch, self.np_rng)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss}
            return state, metrics
        
        def eval_step(state, batch):
            loss = mse_loss(state.params, batch, self.np_rng)
            return {'loss': loss}
        
        return train_step, eval_step