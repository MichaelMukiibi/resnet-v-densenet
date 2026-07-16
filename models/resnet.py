import jax.numpy as jnp
from flax import nnx

class ResNetBlock(nnx.Module):
    def __init__(self, channels: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn2 = nnx.BatchNorm(channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        y = nnx.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return nnx.relu(y + residual)

class ResNet(nnx.Module):
    def __init__(self, num_classes: int, num_blocks: int, channels: int, rngs: nnx.Rngs):
        self.conv_init = nnx.Conv(3, channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.blocks = [ResNetBlock(channels, rngs=rngs) for _ in range(num_blocks)]
        self.bn = nnx.BatchNorm(channels, rngs=rngs)
        self.linear = nnx.Linear(channels, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv_init(x)
        for block in self.blocks:
            x = block(x)
        x = nnx.relu(self.bn(x))
        x = jnp.mean(x, axis=(1, 2))  # Global Average Pooling
        return self.linear(x)