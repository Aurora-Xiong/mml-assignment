"""
    Script to evaluate the model on the whole test set and save the results in folder.
"""

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import random_split
from tqdm import tqdm

from data import ImageCaptionDataset
from model import Net, MyNet
from utils import ConfigS, ConfigL, QwenConfig

parser = argparse.ArgumentParser()

parser.add_argument(
    "-C", "--checkpoint-name", type=str, default="model.pt", help="Checkpoint name"
)

parser.add_argument(
    "-S",
    "--size",
    type=str,
    default="S",
    help="Model size [S, L, Q]",
    choices=["S", "L", "s", "l", "q", "Q"],
)

parser.add_argument(
    "-I", "--img-path", type=str, default="", help="Path to the test image folder"
)

parser.add_argument(
    "-R", "--res-path", type=str, default="", help="Path to the results folder"
)

parser.add_argument(
    "-T", "--temperature", type=float, default=1.0, help="Temperature for sampling"
)

parser.add_argument(
    "-N", "--number", type=int, default=5, help="the number of tests."
)

args = parser.parse_args()

if args.size.upper() == "S":
    config = ConfigS()
elif args.size.upper() == "L":
    config = ConfigL()
elif args.size.upper() == "Q":
    config = QwenConfig()

ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)

path = os.path.dirname(os.path.abspath(__file__))
assert os.path.exists(os.path.join(path, args.img_path)), "Path to the test image folder does not exist"

# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = "cuda" if is_cuda else "cpu"


def evaluate_dataset(model, dataset, img_path, save_path, temperature=1.0):
    """
    Evaluate model on dataset.

    Args:
        model: model to evaluate
        dataset: dataset to evaluate on
        img_path: path to images
        save_path: path to save results
    """
    model.eval()

    loop = tqdm(dataset, total=args.number)
    count = 0
    for i, (_, _, img_name) in enumerate(loop):
        if count == args.number:
            break
        img = Image.open(os.path.join(img_path, img_name))

        with torch.no_grad():
            caption, _ = model(img, temperature)

        plt.imshow(img)
        plt.title(caption)
        plt.axis("off")

        plt.savefig(os.path.join(save_path, f'{i}.jpg'), bbox_inches="tight")

        plt.clf()
        plt.close()
        
        count += 1


if __name__ == "__main__":
    
    if args.size.upper() == "S" or args.size.upper() == "L":
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
    elif args.size.upper() == "Q":
        model = MyNet(
            vision_model=config.vision_model,
            blip_model=config.blip_model,
            text_model=config.text_model,
            mp_hidden_size=config.mp_hidden_size,
            max_len=config.max_len,
            num_query_tokens=config.num_query_tokens,
            device=device,
        )
    # TODO: 需要你自己实现一个ImageCaptionDataset在`data/dataset.py`中
    dataset = ImageCaptionDataset(clip_model=config.vision_model, device=device)

    config.train_size = int(config.train_size * len(dataset))
    config.val_size = int(config.val_size * len(dataset))
    config.test_size = len(dataset) - config.train_size - config.val_size

    _, _, test_dataset = random_split(
        dataset, [config.train_size, config.val_size, config.test_size]
    )

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    checkpoint = torch.load(ckp_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_path = os.path.join(
        args.res_path, f"{args.checkpoint_name[:-3]}_{args.size.upper()}"
    )
    save_path = os.path.join(
        path, save_path
    )

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    evaluate_dataset(model, test_dataset, args.img_path, save_path, args.temperature)