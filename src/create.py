import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import math

dataTypes = ["train", "test", "val"]

# output text embedding
def save_text_embedding(inputs, data_path, text_object, data_type):
    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()

    target = f"{data_path}/{text_object}_embedding_{data_type}.txt"

    if os.path.exists(target):
        os.remove(target)

    for i in tqdm(inputs):
        text_embedding = None
        sentence = ""

        for j in i:
            sentence += j

        token = clip.tokenize(sentence).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(token).to(device).float()
            text_feature = text_feature.norm(dim=-1, keepdim=True)
            text_embedding = text_feature.cpu().numpy()
            with open(f"{data_path}/{text_object}_embedding_{data_type}.txt", "a") as f:
                np.savetxt(f, text_embedding)
                
def parse_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()
    for line in data:
        yield line
        
def text_data(file_path, dataType):
    text_input = []
    
    text_data = parse_text(file_path)
    for tc in text_data:
        contents = tc.split('\t')
        contents = [c.replace("\n", "") for c in contents]
        text_input.append(contents)
    
    data_path = os.path.dirname(file_path)
    save_text_embedding(text_input, data_path, "text", dataType)


def get_color_list_bins(data, columns):
    bin_range = 16
    color_hist = ''
    for column in columns:
        if pd.notna(data[columns]):
            colors = ast.literal_eval(data[column])
            for color in colors:
                color_hist += color + ' '
                color_hist += f'{math.floor(color[0]/bin_range)}_{math.floor(color[1]/bin_range)}_{math.floor(color[2]/bin_range)}'
                
    return color_hist

def get_color_metadata(data, represantation):
    for column in ["pakette_lab_reorder"]:
        data[column] = data[column].apply(lambda x: get_color_list_bins(x, [column]))
        
    return data
