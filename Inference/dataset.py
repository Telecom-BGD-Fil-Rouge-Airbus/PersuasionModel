from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch

#get classes name
def read_classes(file_path):
    classes = []
    with open(file_path, 'r', encoding='utf8') as f:
        for label in f.readlines():
            label = label.strip()
            if label:
                classes.append(label)
    return classes

#transform index to class name
def id_to_classes(classes_ids, labels):
    out_classes = []
    for elem in classes_ids:
        int_classes = []
        for idx, ids in enumerate(elem):
            if ids:
                int_classes.append(labels[idx])
        out_classes.append(int_classes)
    return out_classes

#create a custom pytorch Dataset to load image and text at the same time
class PersusasionDataset(Dataset):
    def __init__(self, file_path, image, text, transforms):
        self.transforms = transforms
        self.class_list = read_classes(file_path+'techniques_list_task3.txt')

        self.targets = text
        for t in self.targets:
            t['image'] = image

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        info = self.targets[item]
        text = info['text'].split('\n\n')
        image = self.transforms(info['image'])

        return image, text

# create a custom pytorch collate class to pre process texts and images data in the dataset
class Collate:
    def __init__(self, file_path, classes):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.class_list = classes 

    def __call__(self, data):
        images, texts = zip(*data)

        tokenized_texts = []
        for ts in texts:
            tokenized = [self.tokenizer.cls_token_id]
            for c in ts:
                tokenized.extend(self.tokenizer.encode(c, add_special_tokens=False))
                tokenized.append(self.tokenizer.sep_token_id)
            tokenized_texts.append(torch.LongTensor(tokenized))

        text_lengths = [len(c) for c in tokenized_texts]
        max_len = max(text_lengths)

        bs = len(texts)
        out_texts = torch.zeros(bs, max_len).long()
        for ot, tt, l in zip(out_texts, tokenized_texts, text_lengths):
            ot[:l] = tt

        images = torch.stack(images, 0)

        return images, out_texts, text_lengths
