import torch
import clip
from PIL import Image
import os


def CLIPscore(preds_dict, model, preprocess, img_dir, device):
    cossim = torch.nn.CosineSimilarity()
    avg_score = 0
    for name in preds_dict:
        if len(preds_dict[name]) > 77:
            preds_dict[name] = preds_dict[name][:77]
        image = preprocess(Image.open(os.path.join(img_dir, f"{name}.jpg"))).unsqueeze(0).to(device)
        text = clip.tokenize(preds_dict[name]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        score = 2.5 * max(cossim(image_features, text_features), 0)
        avg_score += score.item()
    return avg_score/len(preds_dict)




