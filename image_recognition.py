from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import random

def unpickle(file, enc):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = enc)
    return dict

def master_labels():
    return unpickle("CIFAR-100/meta", "ASCII")["fine_label_names"]

labels = master_labels()
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode = "fine")
model = keras.models.load_model("image-recognition-model")

# This code was initially used to create the model
# model = Sequential()
# model.add(keras.Input(shape = x_train.shape[1:]))
# model.add(Conv2D(8, (3, 3)))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# model.add(Conv2D(16, (3, 3)))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Activation("softmax"))

# model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
# model.fit(x_train, y_train, batch_size = 25, epochs = 3)

# model.evaluate(x_test, y_test, batch_size = 25)

# Determines the model's accuracy on test images with some number of answer choices
def multiple_choice_model_accuracy(model, test_data, test_labels, num_choices = 4, repeats = 1000):
    num_correct = 0
    total = 0
    for repeat in range(repeats):
        true_answer_index = random.randint(0, len(test_labels)-1)
        true_answer = test_labels[true_answer_index][0]
        answers = list()
        answers.append(true_answer)
        while len(answers) < num_choices:
            potential_answer = test_labels[random.randint(0, len(test_labels)-1)][0]
            if potential_answer not in answers:
                answers.append(potential_answer)
        predictions = model.predict_on_batch(test_data[true_answer_index:true_answer_index+1])
        prediction_index = 0
        current_max = 0
        for i in answers:
            if predictions[0][i] > current_max:
                current_max = predictions[0][i]
                prediction_index = i
        if prediction_index == true_answer:
            num_correct += 1
        total += 1
    return num_correct/total

print("Four Choice Accuracy =", multiple_choice_model_accuracy(model, x_test, y_test))
