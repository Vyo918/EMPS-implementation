import torch
from torch.nn.functional import softmax
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])
LABEL = {
    "big_category": ['Bottom', 'Dress', 'Footwear', 'Top'],
    "Top": ['Shirt', 'Tshirt', 'Hoodie', 'Sweater', 'Polo Shirt'],
    "Bottom": ['Pants', 'Shorts', 'Skirt'],
    "Footwear": ['Flats', 'Heels', 'Shoes', 'Sneakers']
}
MODEL_PATH = {
    "big_category": "./garment_classification/model/big_category.pt",
    "Top": "./garment_classification/model/top.pt",
    "Bottom": "./garment_classification/model/bottom.pt",
    "Footwear": "./garment_classification/model/footwear.pt"
}

def classify(image, device):
    """
    classify garment into category
    image = PIL object
    device = cuda or cpu
    """

    model = torch.jit.load(MODEL_PATH["big_category"], map_location=torch.device(device))
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        image = image.to(device)
        logits = model(image)
        prob, pred = torch.max(softmax(logits, dim=1), 1)
        # if prob.item() < threshold:
        #     handle_under_threshold(image)
        big_category = LABEL["big_category"][pred.item()]

        if big_category == "Dress":
            return big_category, big_category
        else:
            model = torch.jit.load(MODEL_PATH[big_category], map_location=torch.device(device))
            logits = model(image)
            prob, pred = torch.max(softmax(logits, dim=1), 1)
            # if prob.item() < threshold:
            #    handle_under_threshold(image)
            category = LABEL[big_category][pred.item()]
        return big_category, category