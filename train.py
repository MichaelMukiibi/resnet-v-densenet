import os
import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb
from dotenv import load_dotenv
import shutil
import subprocess

# Define the target subdirectory for the cloned repo
CLONE_DIR = os.path.join(os.getcwd(), "cloned_repo")

# Clone the repository only if it doesn't already exist in the runtime
if not os.path.exists(CLONE_DIR):
    print("Cloning repository to resolve modular relative imports...")
    REPO_URL = "https://github.com/michaelmukiibi/resnet-v-densenet.git"
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR], check=True)

# Append the cloned repository root directory to Python's module search path
sys.path.append(CLONE_DIR)

from models.resnet import ResNet
from models.densenet import DenseNet
from utils.dataset import get_cifar10_datasets

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
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
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

def main():
    args = parse_args()
    
    # Authenticate W&B via CLI argument or .env backup
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

    # Prepare loaders and seed
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

    # Establish model views to properly isolate train/eval BatchNorm statistics
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