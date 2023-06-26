import logging
import warnings
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import PersusasionDataset, Collate, read_classes, id_to_classes
from models import MemeMultiLabelClassifier


def load_model(path, verbose=True):
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future")
    warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future")


    checkpoint_path = path + "persuasion_model.pt"
    classes = read_classes(path + 'techniques_list_task3.txt')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint['cfg']

    model = MemeMultiLabelClassifier(cfg, classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.joint_processing_module.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
      model.cuda()

    model.eval()
    
    if verbose:
        print("Persuasion model has been loaded correctly")

    return model

def predict_persuasion(path, model, image, text, threshold):
    text = [{'text': text.replace('\\n', '\n')}]

    test_transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    classes = read_classes(path + 'techniques_list_task3.txt')
    collate_fn = Collate(path, classes)
    dataset = PersusasionDataset(path, image, text, transforms=test_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    dataloaders = [dataloader]

    for image, text, text_len in dataloader:

        if torch.cuda.is_available():
            image = image.cuda()
            text = text.cuda()

        with torch.no_grad():
            pred_probs = model(image, text, text_len, return_probs=True)
            class_ensemble = pred_probs > threshold
            prediction = id_to_classes(class_ensemble, classes)
        return prediction[0]


    
    
    


