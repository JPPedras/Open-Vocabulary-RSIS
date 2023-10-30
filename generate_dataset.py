import os
import json
import pandas as pd
import h5py
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
import math
from scipy.ndimage import label as lb
import random
from torchvision import transforms
from torchvision.transforms.transforms import Resize
from torch.nn import functional as nnf
from utils import *
import torch
import sys
import open_clip


text_token_size = 768
image_token_size = 1024
num_image_tokens = 257
filename = 'RemoteCLIP14'
model_name = 'ViT-L-14'


clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
path_to_your_checkpoints = 'checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'
ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location="cpu")
message = clip_model.load_state_dict(ckpt)


vision_model = clip_model.visual

for p in clip_model.parameters():
    p.requires_grad_(False)

dataset_path = '/cfs/home/u021382/IST-Thesis/datasets/data'


Potsdam_id2labels = {1: 'paved area',2: 'building',3: 'grass', 4: 'tree', 5: 'car'}

iSIAD_id2labels = {1: 'ship', 2: 'storage tank', 3: 'baseball diamond', 4: 'tennis court', 5: 'basketball court', 6: 'ground track field',
                   7: 'bridge', 8: 'large vehicle', 9: 'car', 10: 'helicopter', 11: 'swimming pool', 12: 'roundabout', 
                   13: 'soccer field', 14: 'plane', 15: 'harbor'}

LoveDA_id2labels = {2: 'building', 3: 'road', 4: 'water', 5: 'barren', 6: 'tree', 7: 'agriculture'}

non_instance_classes = ["paved area", "grass", "road", "water", "barren", "agriculture", "chaparral", "sand", "sea", "trees","ground track field","soccer field"]

with open('prompts.json', 'r') as f:
    prompt_info = json.load(f)

classes = {
    'paved area' : 0,
    'building' : 1,
    'grass' : 2,
    'tree' : 3,
    'car' : 4,
    'ship' : 5,
    'storage tank' : 6,
    'baseball diamond' : 7,
    'tennis court' : 8,
    'basketball court' : 9,
    'ground track field': 10,
    'bridge' : 11,
    'large vehicle' : 12,
    'helicopter' : 13,
    'swimming pool' : 14,
    'roundabout' : 15,
    'soccer field' : 16,
    'plane' : 17,
    'harbor' : 18,
    'road': 19,
    'water': 20,
    'barren': 21,
    'agriculture': 22
}

def forward_multihead_attention(x, b, with_aff=False, attn_mask=None):
    """ 
    Simplified version of multihead attention (taken from torch source code but without tons of if clauses). 
    The mlp and layer norm come from CLIP.
    x: input.
    b: multihead attention module. 
    """

    x_ = b.ln_1(x)
    q, k, v = nnf.linear(x_, b.attn.in_proj_weight, b.attn.in_proj_bias).chunk(3, dim=-1)
    tgt_len, bsz, embed_dim = q.size()

    head_dim = embed_dim // b.attn.num_heads
    scaling = float(head_dim) ** -0.5

    q = q.contiguous().view(tgt_len, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)

    q = q * scaling

    attn_output_weights = torch.bmm(q, k.transpose(1, 2)) #  n_heads * batch_size, tokens^2, tokens^2
    if attn_mask is not None:


        attn_mask_type, attn_mask = attn_mask
        n_heads = attn_output_weights.size(0) // attn_mask.size(0)
        attn_mask = attn_mask.repeat(n_heads, 1)
        
        if attn_mask_type == 'cls_token':
            # the mask only affects similarities compared to the readout-token.
            attn_output_weights[:, 0, 1:] = attn_output_weights[:, 0, 1:] * attn_mask[None,...]
            # attn_output_weights[:, 0, 0] = 0*attn_output_weights[:, 0, 0]

        if attn_mask_type == 'all':
            # print(attn_output_weights.shape, attn_mask[:, None].shape)
            attn_output_weights[:, 1:, 1:] = attn_output_weights[:, 1:, 1:] * attn_mask[:, None]
        
    
    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = b.attn.out_proj(attn_output)

    x = x + attn_output
    x = x + b.mlp(b.ln_2(x))

    if with_aff:
        return x, attn_output_weights
    else:
        return x


def preprocess_prompt(prompt):

    text = tokenizer(prompt)
    text_features = clip_model.encode_text(text)

    return text_features.reshape(text_token_size).detach().numpy()


def preprocess_image(img, image_id, hfile):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_fun = transforms.Normalize(mean, std)

    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()

    img = nnf.interpolate(img, (224, 224), mode='bilinear', align_corners=True)

    img = img / 255.0
    x = normalize_fun(img)

    x = x.to(vision_model.positional_embedding.device)
    x = vision_model.conv1(x)

    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    x = torch.cat([vision_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

    x = x + vision_model.positional_embedding.to(x.dtype)
    x = vision_model.ln_pre(x)
    x = x.permute(1, 0, 2)

    extract_layers=[3,7,9]
    activations = []
    for i, res_block in enumerate(vision_model.transformer.resblocks):
        
        x, aff_per_head = forward_multihead_attention(x, res_block, with_aff=True, attn_mask=None)

        if i in extract_layers:
        
            activations += [x.permute(1,0,2)]

        if len(extract_layers) > 0 and i == max(extract_layers):
            break
    
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = vision_model.ln_post(x[:, 0, :])


    if image_id != 0:
        #hfile['image_data/pixel_values'].resize(hfile['image_data/pixel_values'].shape[0]+1, axis=0)
        #hfile['image_data/img_size'].resize(hfile['image_data/img_size'].shape[0]+1, axis=0)
        hfile['image_data/hs3'].resize(hfile['image_data/hs3'].shape[0]+1, axis=0)
        hfile['image_data/hs7'].resize(hfile['image_data/hs7'].shape[0]+1, axis=0)
        hfile['image_data/hs9'].resize(hfile['image_data/hs9'].shape[0]+1, axis=0)
        hfile['image_data/cls'].resize(hfile['image_data/cls'].shape[0]+1, axis=0)

    hfile['image_data/hs3'][-1] = activations[0].reshape(num_image_tokens, image_token_size).detach().numpy()
    hfile['image_data/hs7'][-1] = activations[1].reshape(num_image_tokens, image_token_size).detach().numpy()
    hfile['image_data/hs9'][-1] = activations[2].reshape(num_image_tokens, image_token_size).detach().numpy()
    hfile['image_data/cls'][-1] = x.reshape(image_token_size).detach().numpy()

    return hfile



def add_triplet(dataset_id, class_id, img_id, prompt, mask, hfile, flag):

    cond = preprocess_prompt(prompt)

    if flag:
        hfile['triplets/img_id'].resize(hfile['triplets/img_id'].shape[0]+1, axis=0)
        hfile['triplets/cond'].resize(hfile['triplets/cond'].shape[0]+1, axis=0)
        hfile['triplets/mask'].resize(hfile['triplets/mask'].shape[0]+1, axis=0)
        hfile['triplets/class_id'].resize(hfile['triplets/class_id'].shape[0]+1, axis=0)
        hfile['triplets/dataset_id'].resize(hfile['triplets/dataset_id'].shape[0]+1, axis=0)

    hfile['triplets/img_id'][-1] = img_id
    hfile['triplets/cond'][-1] = cond
    hfile['triplets/mask'][-1] = mask
    hfile['triplets/class_id'][-1] = class_id
    hfile['triplets/dataset_id'][-1] = dataset_id

    return hfile


def generate():
    for split in ['train','val']:
        print(f"Generating {split} h5 file")
        image_id = 0
        with h5py.File(os.path.join(dataset_path, f'image_segmentation_{split}_{filename}.h5'), 'w') as hfile:

            # create folders
            hfile.create_group('image_data')
            hfile.create_group('triplets')
            resize_dset = False

            #hfile['image_data'].create_dataset('pixel_values', (1,672,672,3), maxshape=(None,672, 672,3), dtype="float32", chunks=(1,672,672,3))
            hfile['image_data'].create_dataset('hs3', (1, num_image_tokens, image_token_size), maxshape=(None, num_image_tokens, image_token_size), chunks=(1, num_image_tokens, image_token_size))
            hfile['image_data'].create_dataset('hs7', (1, num_image_tokens, image_token_size), maxshape=(None, num_image_tokens, image_token_size), chunks=(1, num_image_tokens, image_token_size))
            hfile['image_data'].create_dataset('hs9', (1, num_image_tokens, image_token_size), maxshape=(None, num_image_tokens, image_token_size), chunks=(1, num_image_tokens, image_token_size))
            hfile['image_data'].create_dataset('cls', (1,image_token_size), maxshape=(None,image_token_size), chunks=(1,image_token_size))
            #hfile['image_data'].create_dataset('img_size', (1,), maxshape=(None,), dtype='i8', chunks=(1,))

            hfile['triplets'].create_dataset('img_id', (1,), maxshape=(None,), dtype='i8', chunks=(1,))
            hfile['triplets'].create_dataset('class_id', (1,), maxshape=(None,), dtype='i8', chunks=(1,))
            hfile['triplets'].create_dataset('dataset_id', (1,), maxshape=(None,), dtype='i8', chunks=(1,))
            hfile['triplets'].create_dataset('cond', (1, text_token_size), maxshape=(None, text_token_size), chunks=(1, text_token_size))
            hfile['triplets'].create_dataset('mask', (1, 672, 672), maxshape=(None, 672, 672), dtype=bool, chunks=(1, 672, 672))

            # preprocess triplets

            for dataset_name in ['iSAID', 'Potsdam', 'LoveDA']:

                if dataset_name == 'iSAID':
                    convert_dict = iSIAD_id2labels
                    dataset_id = 0
                elif dataset_name == 'Potsdam':
                    convert_dict = Potsdam_id2labels
                    dataset_id = 1
                elif dataset_name == 'LoveDA':
                    convert_dict = LoveDA_id2labels
                    dataset_id = 2

                images = sorted(os.listdir(os.path.join(dataset_path, dataset_name, 'img_dir', split)))
                masks = sorted(os.listdir(os.path.join(dataset_path, dataset_name, 'ann_dir', split)))

                preprocess_progress = tqdm(range(len(images)), desc=datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f' - {dataset_name}')

                for image_name, mask_name in zip(images, masks):

               
                    image_path = os.path.join(dataset_path, dataset_name, 'img_dir', split, image_name)
                    mask_path = os.path.join(dataset_path, dataset_name, 'ann_dir', split, mask_name)

                    image_crops, mask_crops = generate_crops(Image.open(image_path), Image.open(mask_path))

                    for img_crop, mask_crop in zip(image_crops, mask_crops):

                        hfile = preprocess_image(np.array(img_crop), image_id, hfile)

                        #mask_crop = mask_crop.resize((224, 224), resample=Image.NEAREST)
                        inds = np.array(mask_crop)
                        labels = np.unique(inds)

                        # Look at all present labels one at a time
                        for label in labels:
                            if label not in convert_dict.keys():
                                continue
                            class_id = classes[convert_dict[label]]
                            label_mask = np.where(inds == label, 1, 0)

                            # Add masks for non instance classes (grass, paved area , etc...)
                            if convert_dict[label] in non_instance_classes:
                                prompt_list = prompt_info[convert_dict[label]]
                                for prompt in prompt_list:
                                    hfile = add_triplet(dataset_id, class_id, image_id, prompt, label_mask, hfile, resize_dset)
                                    resize_dset = True
                                continue

                            # Add masks for instance classes (car, tree, building, etc...)

                            clusters = find_clusters(label_mask)
                            n_clusters = len(np.unique(clusters))-1

                            if n_clusters == 1:
                                prompt_list = prompt_info[convert_dict[label]][0]
                                for prompt in prompt_list:
                                    hfile = add_triplet(dataset_id, class_id, image_id, prompt, label_mask, hfile, resize_dset)
                                    resize_dset = True
                                continue

                            prompt_list = prompt_info[convert_dict[label]][1]
                            for prompt in prompt_list:
                                hfile = add_triplet(dataset_id, class_id, image_id, prompt, label_mask, hfile, resize_dset)
                                resize_dset = True

                            biggest_cluster = find_big_cluster(clusters)

                            if biggest_cluster is not None:

                                big_mask = np.where(clusters == biggest_cluster, 1, 0)
                                prompt = 'The largest ' + convert_dict[label]
                                hfile = add_triplet(dataset_id, class_id, image_id, prompt, big_mask, hfile, resize_dset)

                            left_mask, right_mask = get_side_masks(clusters)

                            if len(np.unique(left_mask)) > 2:
                                prompt = prompt_info[convert_dict[label]][1][0]+' on the left'
                            else:
                                prompt = prompt_info[convert_dict[label]][0][0]+' on the left'

                            if len(np.unique(left_mask)) > 1:
                                left_mask = np.where(left_mask > 0, 1, 0)
                                hfile = add_triplet(dataset_id, class_id, image_id, prompt, left_mask, hfile, resize_dset)

                            if len(np.unique(right_mask)) > 2:
                                prompt = prompt_info[convert_dict[label]][1][0]+' on the right'
                            else:
                                prompt = prompt_info[convert_dict[label]][0][0]+' on the right'

                            if len(np.unique(right_mask)) > 1:
                                right_mask = np.where(right_mask > 0, 1, 0)
                                hfile = add_triplet(dataset_id, class_id, image_id, prompt, right_mask, hfile, resize_dset)

                        # Add adversarial case (empty mask)
                        not_present = [label for label in convert_dict.keys() if label not in labels]
                        if len(not_present) > 0:

                            adversarial_label = random.choice(not_present)
                            class_id = classes[convert_dict[adversarial_label]]

                            if convert_dict[adversarial_label] in non_instance_classes:
                                prompt = prompt_info[convert_dict[adversarial_label]][0]
                            else:
                                prompt = prompt_info[convert_dict[adversarial_label]][0][0]

                            mask = np.zeros_like(mask_crop)

                            hfile = add_triplet(dataset_id, class_id, image_id, prompt, mask, hfile, resize_dset)

                        image_id += 1
                    preprocess_progress.update(1)

                preprocess_progress.close()

            # hfile['triplets'].create_dataset('img_id', data=triplet_dataset['img_id'])
            # hfile['triplets'].create_dataset('cond', data=triplet_dataset['cond'])
            # hfile['triplets'].create_dataset('mask', data=triplet_dataset['mask'], dtype=int)


if __name__ == '__main__':

    np.random.seed(42)
    random.seed(42)

    generate()
