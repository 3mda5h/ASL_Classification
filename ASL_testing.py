"""
Test model using images it hasn't seen before

Last Updated: 1/26/23
Emily MacPherson
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 


images_path = "C:\\Users\\MacAttack\\Downloads\\archive\\X.npy"
labels_path = "C:\\Users\\MacAttack\\Downloads\\archive\\Y.npy"
#load .npy files
images = np.load(images_path)
string_labels = np.load(labels_path)
#for some reason the labels array is a 2d array with a 2nd dimension of 1, change it to 1d array
string_labels = string_labels.flatten()

indexes = [1853, 6835, 10084, 16926, 17451, 17477, 18088, 10911, 13656, 13840, 13885, 14602, 19038,
          20018, 16926, 15009, 10037, 7919, 6849, 6835, 6037, 5471, 4773, 4637, 4617, 4414, 2980,
          2057, 1241, 1208, 1082, 1166, 902, 6999, 6866, 6644, 2980, 2119, 2057, 1532, 1420, 1323,
          914, 910, 861, 21565, 20065, 13623, 13621, 13544, 7243, 7175, 7124, 850, 821]

correct_labels = ['yes', '8', 'please', '6', '6', '6', '2', 'hello', 'a', 'a', 'bye', 'good', 'd',
                  '1', '6', 'e', 'please', '9', 'whats up', 'whats up', '2', '4', '8', '8', '8',
                  '8', '8', 'bye', '8', '8', '8', '8', '8', 'whats up', 'whats up', 'good', '8', 'e',
                  'bye', '7', 'please', 'please', '8', 'please', '8', 'please', '7', 'bye', 'bye', 'a',
                  'good', 'good', 'good', 'please', 'please']

for i in range(len(indexes)):
    string_labels[indexes[i]] = correct_labels[i]

unique_s_labels = np.unique(string_labels)

"""
le = preprocessing.LabelEncoder()
#print(string_labels)
#print(string_labels)
#tf wants classes to be numbers not strings, labelEncoder assigns a number to each class
int_labels = le.fit_transform(string_labels)
"""

#best: model6, model8, model9
TF_MODEL_FILE_PATH = 'model9.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

print(interpreter.get_signature_list())

classify_lite = interpreter.get_signature_runner('serving_default')

#print("Model validation accuracy: ", interpreter.metrics['accuracy'])

"""
#test with image AI has already seen to make sure not literally everything is broken
print("Test with already seen image:")
i = 1230;
img1 = images[i]
#print(img1)
print(img1.shape)
plt.imshow(img1, cmap='gray')
plt.show()
img1 = tf.expand_dims(img1, 0)

predictions_lite = classify_lite(sequential_25_input=img1)['outputs']
score_lite = tf.nn.softmax(predictions_lite)

print("scores:", score_lite)
print("already seen image most likely belongs to class ", unique_s_labels[np.argmax(score_lite)])
print("The certainty is", 100 * np.max(score_lite))
print("the correct class is", string_labels[i])
"""
#test with own image
print("----------------------------------------------------------")
print("Test with my own image:")
img2 = tf.keras.utils.img_to_array(
    tf.keras.utils.load_img("C:\\Users\\MacAttack\\Downloads\\ASL_test\\test_A_4.jpg", #<------
                            target_size=(128, 128))) /255
#print(img2)
#fix image rotation
#img2 =  np.rot90(img2, 3)
print(img2.shape)
plt.imshow(img2, cmap='gray')
plt.show()
img2 = tf.expand_dims(img2, 0) # Create a batch

predictions_lite = classify_lite(sequential_27_input=img2)['outputs']
score_lite = tf.nn.softmax(predictions_lite)

#print("scores:", score_lite)
print("this image most likely belongs to class", unique_s_labels[np.argmax(score_lite)])
print("The certainty is", 100 * np.max(score_lite))

#find indexes where it incorrectly predicts images it has already seen
#mislabled images more likely be found here
"""
total = 0
for i in range(22801):
    img = images[i]
    img = tf.expand_dims(img, 0)
    predictions_lite = classify_lite(sequential_10_input=img)['outputs']
    score_lite = tf.nn.softmax(predictions_lite)
    prediction = unique_s_labels[np.argmax(score_lite)]
    if(prediction != string_labels[i]):
        total+=1
        print(f'{i}, ', end = "")
print("total:", total)
"""
