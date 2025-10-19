import pickle
from PIL import Image
from img2vec_pytorch import Img2Vec

with open('./model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = '/Users/nisumlimbu/PycharmProjects/OPEN_CV/data/weather_dataset2/val/rain/rain182.jpg'

img = Image.open(image_path)

img_features = img2vec.get_vec(img)

result = model.predict([img_features])
print(result)