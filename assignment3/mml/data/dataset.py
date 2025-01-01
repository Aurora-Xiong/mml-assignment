"""
    Module contains Dataset class, collate function for DataLoader and loader getter function.

    * ImageCaptionDataset loads data from pickle file and returns image embedding and caption.
    * cl_fn is used to process batch of data and return tensors.
    * get_loader returns DataLoader object.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from transformers import AutoTokenizer
import json
import os
from model import ImageEncoder
import tqdm
from PIL import Image

class ImageCaptionDataset(Dataset):
    def __init__(self, clip_model: str, device: str = "cpu"):
        """
        Initialize ImageCaptionDataset

        :param clip_model: CLIP model to use for image embedding
        :param device: device to run the model on
        """
        self.device = device
        self.image_encoder = ImageEncoder(clip_model, device)
        self.data = []
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.caption = os.path.join(self.path, "annotations/train_caption.json")
        self.image_dir = os.path.join(self.path, "images/train2014")
        
        # process the images in advance and save the data for future use
        file_name = clip_model.replace("/", "-")
        
        if os.path.exists(os.path.join(self.path, f"{file_name}.pkl")):
            with open(os.path.join(self.path, f"{file_name}.pkl"), "rb") as f:
                self.data = pickle.load(f)
        else:
            self.indice = {}
            with open(self.caption, "r") as f:
                self.captions = json.load(f)
                
            for _, caption in tqdm.tqdm(enumerate(self.captions), total=len(self.captions)):
                img_id = caption["image_id"]
                if img_id not in self.indice.keys():
                    img_file = os.path.join(self.image_dir, "COCO_train2014_" + img_id.zfill(12) + ".jpg")
                    img = Image.open(img_file)
                    with torch.no_grad():
                        img_emb = self.image_encoder(img)
                    self.indice[img_id] = img_emb.cpu()
                else:
                    img_emb = self.indice[img_id]
                self.data.append({"image_id": img_id, "image_embedding": img_emb, "caption": caption["caption"]})
            with open(os.path.join(self.path, f"{file_name}.pkl"), "wb") as f:
                pickle.dump(self.data, f)
        
        print("Length of data:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = os.path.join(self.image_dir, "COCO_train2014_" + self.data[idx]["image_id"].zfill(12) + ".jpg")
        return self.data[idx]["image_embedding"].to(self.device), self.data[idx]["caption"], img_file

def cl_fn(batch, tokenizer):
    """
    Collate function to process the batch of data and return tensors

    :param batch: List of (image_embedding, caption, img_file) tuples
    :param tokenizer: tokenizer
    :return: img_emb (Tensor), input_ids (Tensor), attention_mask (Tensor)
    """
    img_embs, captions, _ = zip(*batch)
    
    img_embs = torch.cat(img_embs, dim=0)
    
    encodings = tokenizer(list(captions), padding=True, return_tensors="pt", add_special_tokens=True, return_attention_mask=True)

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    return img_embs.cpu(), input_ids.cpu(), attention_mask.cpu()


def get_loader(dataset, bs_exp=5, shuffle=True, num_workers=0, pin_memory=False, text_model="gpt2-medium"):
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    tokenizer.pad_token = tokenizer.eos_token

    return DataLoader(
        dataset,
        batch_size=2**bs_exp,
        collate_fn=lambda b: cl_fn(b, tokenizer),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )