import warnings
# Suppress warnings from huggingface (`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
import logging
# Suppress logging warnings from diffusers.configuration_utils
logging.getLogger("diffusers.configuration_utils").disabled = True
# Suppress logging warnings from diffusers.models.modeling_utils
logging.getLogger("diffusers.models.modeling_utils").disabled = True

import torch
from PIL import Image
import idm_vton.inference
import garment_classification.inference
import style_classification.inference

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TESTING GARMENT CLASSIFICATION
    print("Testing garment classification:")
    garm_imgs = []
    categories = []

    path = "./input/tshirt.png"
    garm_imgs.append((Image.open(path).convert("RGB"), path))

    path = "./input/pants.png"
    garm_imgs.append((Image.open(path).convert("RGB"), path))

    path = "./input/dress.png"
    garm_imgs.append((Image.open(path).convert("RGB"), path))

    for garm_img, file_path in garm_imgs:
        category = garment_classification.inference.classify(garm_img, device)
        categories.append(category)
        print(f"{file_path} classified as: {category}")


    # TESTING IDM-VTON
    print("Testing idm-vton:")

        # TSHIRT + PANTS
    print("     testing tshirt+pants:")
    garm_imgs = []
    categories = []

    path = "./input/model.png"
    human = Image.open(path).convert("RGB")

    path = "./input/tshirt.png"
    garm_imgs.append(Image.open(path).convert("RGB"))
    categories.append("Top")

    path = "./input/pants.png"
    garm_imgs.append(Image.open(path).convert("RGB"))
    categories.append("Bottom")

    im = idm_vton.inference.generate(human, garm_imgs, categories, device)
    im.show()
    im.save("./output/idm-vton(tshirt_pants).png")

        # DRESS
    print("     testing dress:")
    garm_imgs = []
    categories = []

    path = "./input/model.png"
    human = Image.open(path).convert("RGB")

    path = "./input/dress.png"
    garm_imgs.append(Image.open(path).convert("RGB"))
    categories.append("Dress")

    im = idm_vton.inference.generate(human, garm_imgs, categories, device)
    im.show()
    im.save("./output/idm-vton(dress).png")

    # TESTING style CLASSIFICATION
    print("Testing style classification:")
    style_imgs = []
    categories = []

    path = "./output/idm-vton(tshirt_pants).png"
    style_imgs.append((Image.open(path).convert("RGB"), path))

    path = "./output/idm-vton(dress).png"
    style_imgs.append((Image.open(path).convert("RGB"), path))

    for style_img, file_path in style_imgs:
        category = style_classification.inference.classify(style_img, device)
        categories.append(category)
        print(f"{file_path} classified as: {category}")

if __name__ == "__main__":
    main()