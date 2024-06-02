import os
import numpy as np
from tqdm import tqdm

# output text embedding
def save_text_embedding(inputs, data_path, text_object, data_type):
    import torch
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()
    
    target = f"{data_path}/{text_object}_embedding_{data_type}.txt"
    
    if os.path.exists(target):
            os.remove(target)
        
    
    for input in tqdm(inputs):
        text_embedding = None
        sentence = ''
        
        for i in input:
            sentence += i
            
        token = clip.tokenize(sentence).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(token).to(device).float()
            text_feature = text_feature.norm(dim=-1, keepdim=True)
            text_embedding = text_feature.cpu().numpy()
            with open(f"{data_path}/{text_object}_embedding_{data_type}.txt", 'a') as f:
                np.savetxt(f, text_embedding)
        
    
    
        
            
    
# Create corpus, text

# Create text embedding