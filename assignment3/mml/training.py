"""
    Script that contains whole training process.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split

import wandb
from data import ImageCaptionDataset, get_loader
from model import Net, MyTrainer, MyNet
from utils import ConfigS, ConfigL, LRWarmup, QwenConfig

parser = argparse.ArgumentParser()

parser.add_argument(
    "-C", "--checkpoint-name", type=str, default="", help="Checkpoint name"
)

parser.add_argument(
    "-S",
    "--size",
    type=str,
    default="S",
    help="Config [S, L, Q]",
    choices=["S", "L", "s", "l", "Q", "q"],
)

args = parser.parse_args()

if args.size.upper() == 'L':
    config = ConfigL()
elif args.size.upper() == 'S':
    config = ConfigS()
else:
    config = QwenConfig()

# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop_training = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.stop_training = True


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    # TODO: 需要你自己实现一个ImageCaptionDataset在`data/dataset.py`中
    dataset = ImageCaptionDataset(clip_model=config.vision_model, device=device)

    config.train_size = int(config.train_size * len(dataset))
    config.val_size = int(config.val_size * len(dataset))
    config.test_size = len(dataset) - config.train_size - config.val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [config.train_size, config.val_size, config.test_size]
    )

    train_loader = get_loader(
        train_dataset,
        bs_exp=config.batch_size_exp if is_cuda else 2,
        shuffle=True,
        num_workers=config.num_workers if is_cuda else 0,
        text_model=config.text_model,
        pin_memory=is_cuda,
    )

    valid_loader = get_loader(
        val_dataset,
        bs_exp=config.batch_size_exp if is_cuda else 2,
        shuffle=False,
        num_workers=config.num_workers if is_cuda else 0,
        text_model=config.text_model,
        pin_memory=is_cuda,
    )
    if args.size.upper() == 'Q':
        model = MyNet(
            vision_model=config.vision_model,
            blip_model=config.blip_model,
            text_model=config.text_model,
            mp_hidden_size=config.mp_hidden_size,
            max_len=config.max_len,
            num_query_tokens=config.num_query_tokens,
            device=device,
        )
    else:
        model = Net(
            clip_model=config.clip_model,
            text_model=config.text_model,
            ep_len=config.ep_len,
            num_layers=config.num_layers,
            n_heads=config.n_heads,
            forward_expansion=config.forward_expansion,
            dropout=config.dropout,
            max_len=config.max_len,
            device=device,
        )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup.lr_warmup)
    scaler = torch.cuda.amp.GradScaler()

    ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)

    trainer = MyTrainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_dataset=test_dataset,
        test_path=os.path.join("data", "images/train2014"), # TODO: 请你修改这里为你自己的目录
        ckp_path=ckp_path,
        device=device,
    )
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # build train model process with experiment tracking from wandb
    wandb.init(project="captioner", config=config.__dict__)
    wandb.watch(trainer.model, log="all")
    for epoch in range(trainer.epoch, config.epochs):
        trainer.train_epoch()
        trainer.valid_epoch()
        trainer.test_step()

        metadata = trainer.get_training_data()
        val_loss = metadata["valid_loss"][-1]

        # log loss to wandb
        wandb.log(
            {
                "train_loss/loss": metadata["train_loss"][-1],
                "valid_loss/loss": val_loss,
                "lr": metadata["lr"],
                "examples": wandb.Image(metadata["examples"]),
            }
        )
        early_stopping(val_loss)

        if not os.path.exists(config.weights_dir):
            os.makedirs(config.weights_dir)

        if (epoch + 1) % 2 == 0:
            trainer.save_ckp(os.path.join(config.weights_dir, f"epoch_{epoch + 1}.pt"))
            
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break