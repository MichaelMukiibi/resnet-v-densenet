import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb
from dotenv import load_dotenv
from datasets import load_dataset

# ==========================================
# 1. Dataset Pipeline (Hugging Face)
# ==========================================

class HFDatasetIterator:
    def __init__(self, hf_dataset_split, batch_size, shuffle=False):
        self.ds = hf_dataset_split.to_iterable_dataset()
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def as_numpy_iterator(self):
        ds = self.ds
        if self.shuffle:
            ds = ds.shuffle(seed=42, buffer_size=10000)
        
        for batch in ds.iter(batch_size=self.batch_size, drop_last_batch=True):
            images = np.array(batch['img'], dtype=np.float32) / 255.0
            
            mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
            std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
            images = (images - mean) / std
            
            yield {
                'image': images,
                'label': np.array(batch['label'], dtype=np.int32)
            }

def get_cifar10_datasets(batch_size: int = 64):
    ds = load_dataset("cifar10", trust_remote_code=True)
    train_ds = HFDatasetIterator(ds['train'], batch_size, shuffle=True)
    test_ds = HFDatasetIterator(ds['test'], batch_size, shuffle=False)
    return train_ds, test_ds

# ==========================================
# 2. ResNet Model Definition
# ==========================================

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

# ==========================================
# 3. DenseNet Model Definition
# ==========================================

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

# ==========================================
# 4. Training Steps & Arguments
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train JAX/Flax NNX models with W&B logging")
    parser.add_argument("--model", type=str, choices=["resnet", "densenet"], default="resnet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wand_key", type=str, default="", help="W&B API Key passed via CLI")
    parser.add_argument("--project", type=str, default="resnet-vs-densenet")
    return parser.parse_args()

def loss_fn(model, batch):
    logits = model(batch['image'])
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
    return loss, logits

@nnx.jit
def train_step(model, optimizer, metrics, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    optimizer.update(grads)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

@nnx.jit
def eval_step(model, metrics, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

# ==========================================
# 5. Main Loop Orchestrator
# ==========================================

def main():
    args = parse_args()
    
    # Authenticate with Weights & Biases
    if args.wand_key:
        wandb.login(key=args.wand_key)
    else:
        load_dotenv()
        if "WANDB_API_KEY" in os.environ:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            
    wandb.init(
        project=args.project,
        config={
            "model_architecture": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }
    )

    train_ds, test_ds = get_cifar10_datasets(args.batch_size)
    rngs = nnx.Rngs(42)
    
    if args.model == "resnet":
        model = ResNet(num_classes=10, num_blocks=3, channels=64, rngs=rngs)
    else:
        model = DenseNet(num_classes=10, num_blocks=3, in_channels=64, growth_rate=12, rngs=rngs)
        
    optimizer = nnx.Optimizer(model, optax.adam(args.lr))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average(),
        accuracy=nnx.metrics.Accuracy()
    )

    train_model = nnx.view(model, use_running_average=False)
    eval_model = nnx.view(model, use_running_average=True)

    print(f"Beginning training session for: {args.model}")
    
    for epoch in range(1, args.epochs + 1):
        metrics.reset()
        for batch in train_ds.as_numpy_iterator():
            train_step(train_model, optimizer, metrics, batch)
        train_res = metrics.compute()
        
        metrics.reset()
        for batch in test_ds.as_numpy_iterator():
            eval_step(eval_model, metrics, batch)
        eval_res = metrics.compute()
        
        t_loss, t_acc = float(train_res['loss']), float(train_res['accuracy'])
        v_loss, v_acc = float(eval_res['loss']), float(eval_res['accuracy'])
        
        print(f"Epoch {epoch:02d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
              
        wandb.log({
            "epoch": epoch,
            "train_loss": t_loss,
            "train_accuracy": t_acc,
            "val_loss": v_loss,
            "val_accuracy": v_acc
        })

    wandb.finish()

if __name__ == "__main__":
    main()