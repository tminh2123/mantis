# -*- coding: utf-8 -*-
import os
import cv2
import torch
from tqdm import tqdm
import clip
import config
from utils.functions import read_text


def correct_dims(img):
    """Ensure image has channel dim"""
    if len(img.shape) == 2:  # (H, W)
        return img[..., None]  # (H, W, 1)
    return img


def generate_roi_t(text_string, keywords, token_len):
    """Generate region-of-interest tokens"""
    import re
    tokens = re.findall(r'\w+|[^\w\s]', text_string)
    processed_tokens = []
    for token in tokens:
        if token == "unilateral":
            processed_tokens.extend(["unil", "ateral"])
        else:
            processed_tokens.append(token)

    roi_t = torch.zeros(token_len)
    for i, token in enumerate(processed_tokens):
        if re.fullmatch(r'[a-z]+', token) and token in keywords and i + 1 < len(roi_t):
            roi_t[i + 1] = 1.0

    return roi_t


def preprocess_dataset(dataset_path, task_name, text_dict, image_size=224, token_len=18):
    images_path = os.path.join(dataset_path, "images")
    masks_path = os.path.join(dataset_path, "masks")
    images_list = os.listdir(images_path)

    keywords = ["bilateral", "unil", "ateral", "left", "right", "upper", "lower", "middle",
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

    all_data = {}
    for img_name in tqdm(images_list, desc=f"Preprocessing samples", unit="file"):

        # ---- image ----
        img = cv2.imread(os.path.join(images_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [C,H,W]

        # ---- mask ----
        if task_name == "MosMedData+":
            mask_name = img_name
        else:
            mask_name = "mask_" + img_name
        mask = cv2.imread(os.path.join(masks_path, mask_name), 0)
        mask = cv2.resize(mask, (image_size, image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask_tensor = torch.from_numpy(mask).long()  # [H,W]

        # ---- text ----
        text = text_dict[mask_name].split("\n")[0].lower()
        with torch.no_grad():
            text_token = clip.tokenize(text, context_length=token_len, truncate=True).squeeze()
        text_mask = (text_token != 0).int()
        roi_t = generate_roi_t(text, keywords, token_len)

        # ---- save tensors ----
        all_data[img_name] = {"image": img_tensor,       # torch.float [C,H,W]
                              "mask": mask_tensor,       # torch.long  [H,W]
                              "text_token": text_token,  # torch.long
                              "text_mask": text_mask,    # torch.int
                              "roi_t": roi_t             # torch.float
                              }

    # ---- save as .pt file ----
    save_file = os.path.join(dataset_path, "preprocessed.pt")
    torch.save(all_data, save_file)


if __name__ == "__main__":
    for split_name in ["Train_Folder", "Val_Folder", "Test_Folder"]:
        dataset_path = os.path.join(config.dataset_root, split_name)
        text_file = os.path.join(dataset_path, f"{split_name.replace('_Folder', '')}_text.xlsx")
        text_dict = read_text(text_file)
        preprocess_dataset(dataset_path, task_name=config.task_name, text_dict=text_dict,
                           image_size=config.img_size, token_len=config.token_len)

