from img2vec_pytorch import Img2Vec
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle

# Prepare the data
img2vec = Img2Vec()
data_dir = '../data/weather_dataset2'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
data = {}

for j , dir_ in enumerate([train_dir,val_dir]):
    features=[]
    labels=[]
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_).convert('RGB')

            img_features = img2vec.get_vec(img)

            features.append(img_features)
            labels.append(category)
    if j == 0:
        data['training_data'] = features
        data['training_labels'] = labels
    else:
        data['validation_data'] = features
        data['validation_labels'] = labels

# train model

model = RandomForestClassifier()
model.fit(data['training_data'], data['training_labels'])

# test performance
y_pred = model.predict(data['validation_data'])
score = accuracy_score(data['validation_labels'], y_pred)
print(score)

# save the model
with open('./model.p', 'wb') as f:
    pickle.dump(model, f)
    f.close()