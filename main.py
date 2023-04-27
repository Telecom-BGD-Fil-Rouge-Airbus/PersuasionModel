from PIL import Image
from inference_persuasion_taskC import load_model, predict_persuasion

#added function to load the text an the image of the exemple
def load_image_and_text(image_path, text_path):
    with open(image_path, 'rb') as f:
        image = Image.open(f).convert('RGB')

    with open(text_path, 'r', encoding='utf8') as f:
        text = f.read()

    return image, text

if __name__ == '__main__':
    
    #the threshold at which point a probability is considered high enough to make the label appear
    threshold = 0.5

    path = "./"

    #load image and text using a function but it's just for the exemple and it's not necessary  
    image, text = load_image_and_text(path + "data/img1.jpg", path + "data/text1.txt")

    model = load_model(path, verbose=True)

    prediction = predict_persuasion(path, model, image, text, threshold)

    print("The presuasion techniques used in this meme are :")
    for technique in prediction:
        print(" - ",technique)