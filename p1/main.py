import argparse
import json
import os

import clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument("--label", "-l", type=str, default="./", help="output path")
    parser.add_argument("--imgpath", "-i", type=str, default="./", help="output path")
    parser.add_argument("--outpath", "-o", type=str, default="./", help="output path")

    return parser.parse_args()


class Infdataset:
    def __init__(self, datapath):
        # self.files =
        self.images_list = sorted(
            [os.path.join(datapath, x) for x in os.listdir(datapath)]
        )
        print(
            f"finish building image list at {datapath}, number of images:{len(self.images_list)}"
        )
        self.filenames = sorted([x for x in os.listdir(datapath)])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        im = Image.open(self.images_list[idx])
        im = TF.to_tensor(im)
        label = self.filenames[idx].split("_")[0]
        return im, label, self.filenames[idx]

    def get_filelist(self):
        return self.filenames


def inference():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)
    # Download the dataset

    dataset = Infdataset(args.imgpath)
    dataloader = DataLoader(dataset, batch_size=1)
    # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    with open(args.label) as f:
        classes = json.load(f)

    correct = 0
    prediction = pd.DataFrame({"filename": [], "label": []})
    for image, label, name in tqdm(dataloader, position=0, leave=True):
        # Prepare the inputs
        image = TF.to_pil_image(image.squeeze(0))
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat(
            [clip.tokenize(f"a photo of a {c}") for _, c in classes.items()]
        ).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        values = values.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy().astype(str)
        # Print the result
        # print("\nTop predictions:\n")
        # print(f"{classes[str(indices[0])]:>16s}: {100 * values[0].item():.2f}%")
        # print(type(indices[0]), type(label[0]))
        new_row = pd.DataFrame({"filename": name, "label": indices[0]}, index=[0])
        prediction = pd.concat([prediction, new_row])
        if indices[0] == label[0]:
            correct += 1
            # print(correct)
    prediction.to_csv(args.outpath, index=False)
    print(correct)
    print(correct / len(dataset))


if __name__ == "__main__":
    args = get_args()
    inference()
