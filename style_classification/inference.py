import torch
from torch.nn.functional import softmax
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
])   
LABEL = ['Bohemian', 'Casual', 'Formal', 'Semi-formal', 'Sporty', 'Streetwear']
MODEL_PATH = "./style_classification/model/model.pt"

def classify(image, device):
    """
    classify outfit into style
    image = PIL object
    device = cuda or cpu
    """

    model = torch.jit.load(MODEL_PATH, map_location=torch.device(device))
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        image = image.to(device)
        logits = model(image)
        prob, pred = torch.max(softmax(logits, dim=1), 1)
        # if prob.item() < threshold:
        #     handle_under_threshold(image)
        style = LABEL[pred.item()]

        return style