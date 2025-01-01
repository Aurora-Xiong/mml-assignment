"""
    Module contains final Model and all pieces of it.
"""
import torch
import torch.nn as nn
from transformers import Blip2Model
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import ImageEncoder

class Mapping(nn.Module):
    """
    Maps image embedding to the size of qwen2.5 embedding.
    """

    def __init__(
        self,
        im_size,
        mp_hidden_size,
        blip_model,
        td_hidden_size,
        device="cpu",
    ):
        super(Mapping, self).__init__()
        self.device = device
        model = Blip2Model.from_pretrained(blip_model)
        self.qformer = model.qformer.to(self.device)
        self.query_tokens = model.query_tokens.to(self.device)
        self.l1 = nn.Linear(im_size, mp_hidden_size).to(self.device) 
        self.l2 = nn.Linear(self.qformer.encoder.config.hidden_size, td_hidden_size).to(self.device)
        self.gelu = nn.GELU()
        self.init_weights()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.l1(x)
        x_mask = torch.ones(x.size()[:-1], dtype=torch.long, device=self.device)
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        x = self.qformer(query_embeds=query_tokens, encoder_hidden_states=x, encoder_attention_mask=x_mask)
        lm_inputs = self.gelu(self.l2(x[0]))
        return lm_inputs

    def init_weights(self):
        for m in [self.l1, self.gelu, self.l2]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

class TextDecoder(nn.Module):
    """
    Processes embedding into caption.
    """

    def __init__(self, model, device="cpu"):
        super(TextDecoder, self).__init__()

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embedding, attention_mask=None):
        text_features = self.model(
            inputs_embeds=embedding, attention_mask=attention_mask
        )

        return text_features.logits


class MyNet(nn.Module):
    """
    Final Model class. Puts all pieces together and generates caption based on image.
    """

    def __init__(
        self,
        vision_model,
        blip_model,
        text_model,
        mp_hidden_size,
        max_len,
        num_query_tokens,
        device="cpu",
    ):
        """
        Model constructor.
        Args:
            blip_model: blip model name [str]
            text_model: text model name [str]
            mp_hidden_size: hidden size of mapping module [int]
            max_len: maximum length of generated caption [int]
            num_query_tokens: number of query tokens [int]
            device: device to run model on [str]
        """
        super(MyNet, self).__init__()

        self.device = device

        self.ie = ImageEncoder(model=vision_model, device=device)
        self.td = TextDecoder(model=text_model, device=device)
        self.mp = Mapping(im_size=self.ie.model.config.hidden_size, blip_model=blip_model, td_hidden_size=self.td.model.config.hidden_size, device=device, mp_hidden_size=mp_hidden_size)

        self.max_len = max_len
        self.num_query_tokens = num_query_tokens

        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.td.tokenizer.pad_token_id) # chanded on epoch 91
        self.criterion = nn.CrossEntropyLoss()

        self.freeze_layers()

    def freeze_layers(self):
        # freeze everything, except:
        # the l1, l2, gelu and query_tokens in mapping module
        # last transformer layer of the text decoder
        # language model head of the text decoder
        for p in [
            *list(self.ie.parameters()),
            *list(self.td.parameters()),
            *list(self.mp.qformer.parameters()),
        ]:
            p.requires_grad = False
        for p in [
            *list(self.td.model.base_model.layers[-1].parameters()),
        ]:
            p.requires_grad = True


    def forward(self, img, temperature=1.0):
        """
        Caption generation for a single image.
        Args:
            img: image to generate caption for [PIL.Image]
        Returns:
            caption: generated caption [str]
            tokens: generated tokens [torch.Tensor]
        """

        if temperature <= 0.0:
            temperature = 1.0
            print("Temperature must be positive. Setting it to 1.0")

        with torch.no_grad():
            img_embedded = self.ie(img)
            # (num_query_tokens, embed_size)
            start_emb = self.mp(img_embedded)

            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    tok_emb = self.td.model.base_model.embed_tokens(
                        torch.tensor(tokens).to(self.device)
                    )
                    tok_emb = tok_emb.unsqueeze(0)
                    emb = torch.cat([start_emb, tok_emb], dim=1)
                else:
                    emb = start_emb

                pred = self.td(emb)

                pred = torch.softmax(pred / temperature, dim=-1)

                _, pred = torch.max(pred, dim=-1)
                pred = pred.squeeze(0)

                last_token = pred[-1].item()

                tokens.append(last_token)

                if last_token == self.td.tokenizer.eos_token_id:
                    break

            decoded = self.td.tokenizer.decode(tokens[:-1])

            decoded = decoded.strip()
            if len(decoded)>0:
                decoded = decoded[0].upper() + decoded[1:]

            return decoded, tokens

    def train_forward(self, img_emb, trg_cap, att_mask):
        # method should get embedded by CLIP images and trg_text without last token.
        # dataset should contain image, embedded image, text

        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]

        img_mapped = self.mp(img_emb)

        # embed all texts and concat with map sos
        text_emb = self.td.model.base_model.embed_tokens(x)

        # N, len, embed_size
        x = torch.concat([img_mapped, text_emb], dim=1)
        x_mask = torch.concat(
            [torch.ones(x_mask.shape[0], self.num_query_tokens).to(self.device), x_mask], dim=1
        )
        res = self.td(x, attention_mask=x_mask)

        loss = self.criterion(
            res[:, self.num_query_tokens :, :].reshape(-1, res.shape[-1]), y.reshape(-1)
        )

        return loss
    