import jax.numpy as jnp
from flax import nnx

class DenseNetBlock(nnx.Module):
    def __init__(self, in_channels: int, growth_rate: int, rngs: nnx.Rngs):
        self.bn1 = nnx.BatchNorm(in_channels, rngs=rngs)
        self.conv1 = nnx.Conv(in_channels, growth_rate, kernel_size=(3, 3), padding='SAME', rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nnx.relu(self.bn1(x))
        new_features = self.conv1(y)
        return jnp.concatenate([x, new_features], axis=-1)

class DenseNet(nnx.Module):
    def __init__(self, num_classes: int, num_blocks: int, in_channels: int, growth_rate: int, rngs: nnx.Rngs):
        self.conv_init = nnx.Conv(3, in_channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        
        # Avoid appending issues in graph-tracing by defining list comprehensions
        self.blocks = [
            DenseNetBlock(in_channels + i * growth_rate, growth_rate, rngs=rngs)
            for i in range(num_blocks)
        ]
        
        total_channels = in_channels + num_blocks * growth_rate
        self.bn = nnx.BatchNorm(total_channels, rngs=rngs)
        self.linear = nnx.Linear(total_channels, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv_init(x)
        for block in self.blocks:
            x = block(x)
        x = nnx.relu(self.bn(x))
        x = jnp.mean(x, axis=(1, 2))
        return self.linear(x)