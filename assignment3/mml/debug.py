import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split

from data.dataset import ImageCaptionDataset, get_loader
from model.model import Net
from utils.config import ConfigL

config = ConfigL()

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

dataset = ImageCaptionDataset(clip_model=config.clip_model, device=device)

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

ckp_path = os.path.join(config.weights_dir, "epoch_6.pt")
checkpoint = torch.load(ckp_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

img_name = "mml/data/images/train2014/COCO_train2014_000000000009.jpg"
from PIL import Image
img = Image.open(img_name)

caption, _ = model(img)
print(caption)

img_name = "mml/data/images/train2014/COCO_train2014_000000000025.jpg"
img = Image.open(img_name)

caption, _ = model(img)
print(caption)
