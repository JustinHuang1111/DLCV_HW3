import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from PIL import Image
import clip
import torch
import language_evaluation

from clip_score import CLIPscore

def readJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except:
        return None


def getGTCaptions(annotations):
    img_id_to_name = {}
    for img_info in annotations["images"]:
        img_name = img_info["file_name"].replace(".jpg", "")
        img_id_to_name[img_info["id"]] = img_name

    img_name_to_gts = defaultdict(list)
    for ann_info in annotations["annotations"]:
        img_id = ann_info["image_id"]
        img_name = img_id_to_name[img_id]
        img_name_to_gts[img_name].append(ann_info["caption"])
    return img_name_to_gts


class CIDERScore:
    def __init__(self):
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])

    def __call__(self, predictions, gts):
        """
        Input:
            predictions: dict of str
            gts:         dict of list of str
        Return:
            cider_score: float
        """
        # Collect predicts and answers
        predicts = []
        answers = []
        for img_name in predictions.keys():
            predicts.append(predictions[img_name])
            answers.append(gts[img_name])
        
        # Compute CIDEr score
        results = self.evaluator.run_evaluation(predicts, answers)
        return results['CIDEr']


# class CLIPScore:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
#         self.model.eval()

#     def __call__(self, predictions, images_root):
#         """
#         Input:
#             predictions: dict of str
#             images_root: str
#         Return:
#             clip_score: float
#         """
#         total_score = 0.

#         for img_name, pred_caption in predictions.items():
#             image_path = os.path.join(images_root, f"{img_name}.jpg")
#             image = Image.open(image_path).convert("RGB")
#             print(image)
#             total_score += self.getCLIPScore(image, pred_caption)
#         return total_score / len(predictions)

#     def getCLIPScore(self, image, caption):
#         """
#         This function computes CLIPScore based on the pseudocode in the slides.
#         Input:
#             image: PIL.Image
#             caption: str
#         Return:
#             cilp_score: float
#         """
#         # val_clip = CLIPscore(pred_caps, clip_model, clip_preprocess, img_dir, device)
        
       
#         return val_clip



def main(args):
    # Read data
    predictions = readJSON(args.pred_file)
    annotations = readJSON(args.annotation_file)

    # Preprocess annotation file
    gts = getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CIDErScore
    cider_score = CIDERScore()(predictions, gts)

    # CLIPScore
    device = "cuda"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    clip_score = CLIPscore(predictions, clip_model, clip_preprocess, args.images_root, device)
    
    # clip_score = CLIPScore()(predictions, args.images_root)
    
    print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--pred_file", help="Prediction json file")
    parser.add_argument("--images_root", default="../hw3_data/p2_data/images/val/", help="Image root")
    parser.add_argument("--annotation_file", default="../hw3_data/p2_data/val.json", help="Annotation json file")

    args = parser.parse_args()

    main(args)