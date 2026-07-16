# ResNet vs DenseNet on CIFAR-10 (JAX/Flax NNX)

A modular repository designed to train, evaluate, and compare ResNet and DenseNet architectures on the CIFAR-10 dataset using JAX and the Flax NNX API. All training runs are configured to execute seamlessly on Google Colab and log telemetry data directly to Weights & Biases (W&B).

## Project Structure

```text
resnet-v-densenet/
├── .env                  # Environment file for sensitive API keys
├── requirements.txt      # Project dependencies
├── train.py              # Unified training and evaluation script
├── models/
│   ├── __init__.py
│   ├── resnet.py         # ResNet architecture module
│   └── densenet.py       # DenseNet architecture module
└── utils/
    ├── __init__.py
    └── dataset.py        # HuggingFace Datasets CIFAR-10 data loading pipeline
```

## Setup Instructions
1. **Clone the repository** to your workspace
2. **Install dependencies** using your package manager:

```bash
pip install -r requirements.txt
```
3. **Configure W&B API key**:
Create a `.env` file in the root directory and add your Weights & Biases API key:
```txt
WANDB_API_KEY=your_actual_wandb_api_key
```

## Usage & Execution
Execute training loops directly via your remote Colab runner. Pass your `WANDB_API_KEY` from the environment into the execution script via the `--wand_key` argument. 

### Train ResNet
Run the following command to kick off the ResNet training loop:
```bash
source .env

colab run --gpu t4 -s resnet-cifar10 xtrain.py --model resnet --epochs 10 --batch_size 64 --lr 0.001 --wand_key $WANDB_API_KEY
```

### Train DenseNet
Run the following command to kick off the DenseNet training loop:
```bash
source .env

colab run --gpu t4 -s densenet-cifar10 xtrain.py --model densenet --epochs 10 --batch_size 64 --lr 0.001 --wand_key $WANDB_API_KEY
```

## CLI Configuration Arguments
* `--model`: Select architecture (`resnet` or `densenet`).

* `--epochs`: Specify total training iterations over the dataset.

* `--batch_size`: Define batch size for data loaders.

* `--lr`: Set the learning rate for the Adam optimizer.

* `--wand_key`: Pass the W&B API token for remote runtime authorization.

* `--project`: Declare the destination W&B project namespace.
