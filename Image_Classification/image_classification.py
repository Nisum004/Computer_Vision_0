import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Prepare data
input_dir = "/Users/nisumlimbu/PycharmProjects/OPEN_CV/data/clf-data"
categories = ['empty', 'not_empty']

data = []
labels = []

# loop through categories (empty, not_empty) and assign category_index and category
for category_idx, category in enumerate(categories): #(eg: category_idx:1,category:not_empty)
    for file in os.listdir(os.path.join(input_dir, category)):

        # path of each image ( eg: input_dir->not_empty->image1.jpg)
        img_path = os.path.join(input_dir, category, file)

        # convert image into numerical array
        img = imread(img_path)

        # resize the numerical array of image
        resized_img = resize(img, (15, 15))

        # flatten -> convert 2d/3d image array to 1d vector
        data.append((resized_img.flatten())/255.0)
        # flatten converts image into feature vector that is useful for feeding into classifier

        # 0 for empty 1 for not empty
        labels.append(category_idx)

# convert python lists into numpy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# VISUALIZE THE DATA
print('Data Shape: ',data.shape)  # shape of each data (100, 675)->(no of image, pixels each image)
print('Label shape: ',labels.shape) # shape of each data, also gives no of values in labels
print ('Unique values of labels: ',np.unique(labels)) # how many unique value in labels
print ('no of 0 labels: ',np.sum(labels == 0))
print ('no of 1 labels: ', np.sum(labels == 1)) # how many times each integer appears , here 0,1
# print(labels[3], data[3])

# TRAIN /TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# TRAIN CLASSIFIER

# we choose svm  ml algorithm
classifier = SVC()

# train 12 classifies with 3 and 4 combination
parameters = [{'gamma':[0.01, 0.001, 0.0001],'C':[1, 10, 100, 1000]}]

# It systematically works through every combination of a predefined grid of hyperparameter values,
# trains a model with each combination, and uses cross-validation to assess its performance.
grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# TEST performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_test, y_prediction)

print(f'{format(str(score*100))}% of samples were correctly classified')
print(f"score:{score}")
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

save_path = os.path.join('../saved', 'best_model.pkl')
pickle.dump(best_estimator, open(save_path, 'wb'))

print(f"model saved to {save_path}")