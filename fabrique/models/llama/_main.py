import jax
import jax.numpy as jnp
from flax import nnx


class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b


def main():
    model = Linear(5, 4, rngs=nnx.Rngs(params=0))
    x = jnp.ones(5)
    model(x)
    nnx.jit(model)(x)