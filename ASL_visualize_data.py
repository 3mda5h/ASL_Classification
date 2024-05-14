"""
visualize ASL data using matplot

last updated:1/26/23
Emily MacPherson
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

images_path = "C:\\Users\\MacAttack\\Downloads\\archive\\X.npy"
labels_path = "C:\\Users\\MacAttack\\Downloads\\archive\\Y.npy"
#load .npy files
images = np.load(images_path)
string_labels = np.load(labels_path)

#for some reason the labels array is a 2d array with a 2nd dimension of 1, change it to 1d array
string_labels = string_labels.flatten()

s_labels_unique = np.unique(string_labels)

#le = preprocessing.LabelEncoder()
#tf wants classes to be numbers not strings, labelEncoder assigns a number to each class
#int_labels = le.fit_transform(string_labels)

x_train, x_test, y_train, y_test = train_test_split(images, string_labels, test_size=0.20, random_state=42)

print("shape of data set:", images.shape)

#for i in range(len(string_labels)):
    #print(string_labels[i], end = " ")

print("unique string labels: ", s_labels_unique)
    

print("5 examples of every class:-----------------------------------------------")
i = 0
found = False
while(i < 27):
    found = False
    for j in range(22801):
        if(string_labels[j] == s_labels_unique[i]):
            found = True
            new_fig = plt.figure(figsize=(20, 20))
            for k in range(5): 
                new_fig.add_subplot(1, 5, k+1)
                plt.imshow(images[j+k])
                plt.axis('off')
                plt.title(string_labels[j+k])
            i+=1
        if(found == True): break
plt.show()
print()
print()



print("first 5 training images: ----------------------------------------------------")
fig1 = plt.figure(figsize=(20, 20))
for i in range(5):
    fig1.add_subplot(1, 5, i+1)
    plt.imshow(x_train[i])
    plt.axis('off')
    plt.title(y_train[i])
plt.show()

fig2 = plt.figure(figsize=(20, 20))
print("first 5 testing images: ----------------------------------------------------")
for i in range(5):
    fig2.add_subplot(1, 5, i+1)
    plt.imshow(x_test[i])
    plt.axis('off')
    plt.title(y_test[i])
plt.show()

"""
failed_pred3 = [408, 821, 850, 7124, 7175, 7219, 7223, 7243, 7256, 7269, 7326, 7354, 7780, 7901, 7914, 7969, 8164, 8307, 8375, 8474, 8514, 8602, 8678, 8750, 8885, 9256, 9403, 10084, 10312, 10665, 10818, 10865, 10867, 10911, 12628, 13364, 13410, 13415, 13423, 13426, 13459, 13466, 13504, 13532, 13544]

for i in range(len(failed_pred3)):
    plt.imshow(images[failed_pred3[i]])
    plt.axis('off')
    print()
    print()
    print("index of BELOW image:", failed_pred3[i])
    plt.title(string_labels[failed_pred3[i]])
    plt.show()
"""

