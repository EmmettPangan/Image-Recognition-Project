from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import random
from datetime import datetime


start=datetime.now()

def unpickle(file, enc):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = enc)
    return dict

def master_labels():
    return unpickle("C:/Users/emmet/Documents/College/VS Code/CIFAR-100/meta", "ASCII")["fine_label_names"]

# def training_materials():
#     training = unpickle("C:/Users/emmet/Documents/College/VS Code/CIFAR-100/train", "bytes")
#     return (training[b"data"], training[b"fine_labels"])

# def testing_materials():
#     testing = unpickle("C:/Users/emmet/Documents/College/VS Code/CIFAR-100/test", "bytes")
#     return (testing[b"data"], testing[b"fine_labels"])

# def convert_img(img):
#     (r, g, b) = np.array_split(img, 3)
#     converted_img = np.empty((32, 32, 3), dtype = "uint8")
#     for i in range(32):
#         for j in range(32):
#             converted_img[i][j][0] = r[i*32+j]
#             converted_img[i][j][1] = g[i*32+j]
#             converted_img[i][j][2] = b[i*32+j]
#     return converted_img

# def create_data(img_data, img_labels):
#     data = list()
#     for i in range(img_data.shape[0]):
#         data.append([convert_img(img_data[i]), img_labels[i]])
#     return data

# (t1, l1) = training_materials()
# training_data = create_data(t1, l1)

# random.shuffle(training_data)

# x_train = list()
# y_train = list()

# for features, label in training_data:
#     x_train.append(features)
#     y_train.append(label)

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

# def multiple_choice_model_accuracy(model, test_data, test_labels, num_choices = 4, repeats = 1000):
#     num_correct = 0
#     total = 0
#     for repeat in range(repeats):
#         true_answer_index = random.randint(0, len(test_labels)-1)
#         true_answer = test_labels[true_answer_index][0]
#         answers = list()
#         answers.append(true_answer)
#         while len(answers) < num_choices:
#             potential_answer = test_labels[random.randint(0, len(test_labels)-1)][0]
#             if potential_answer not in answers:
#                 answers.append(potential_answer)
#         predictions = model.predict_on_batch(test_data[true_answer_index:true_answer_index+1])
#         prediction_index = 0
#         current_max = 0
#         for i in answers:
#             if predictions[0][i] > current_max:
#                 current_max = predictions[0][i]
#                 prediction_index = i
#         if prediction_index == true_answer:
#             num_correct += 1
#         total += 1
#     return num_correct/total

# print("Accuracy =", multiple_choice_model_accuracy(model, x_test, y_test))

labels = master_labels()
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode = "fine")
model = keras.models.load_model("image-recognition-model")

def random_image(image_data, label_data):
    '''
    Takes two tensors, image_data and label_data
    Returns a random image tensor, label index, and data index
    '''
    random_index = random.randint(0, len(image_data)-1)
    return (image_data[random_index], label_data[random_index][0], random_index)

def random_indices(correct_label_index, labels, num_labels = 4):
    '''
    Takes a value correct_label_index, a list labels, and the number of indices to generate
    '''
    indices = list()
    indices.append(correct_label_index)
    while len(indices) < num_labels:
        random_label_index = random.randint(0, len(labels)-1)
        if random_label_index not in indices:
            indices.append(random_label_index)
    return indices

def random_choices(indices, labels):
    '''
    Returns the string label choices in a list for the given indices
    '''
    choices = list()
    for i in indices:
        choices.append(labels[i])
    return choices

def predicted_index(model, image_data, correct_data_index, correct_label, indices):
    '''
    Returns the predicted label index for an image
    '''
    predictions = model.predict_on_batch(image_data[correct_data_index:correct_data_index+1])
    max_index = 0
    current_max = 0
    for i in indices:
        if predictions[0][i] > current_max:
            current_max = predictions[0][i]
            max_index = i
    return max_index

def new_round(model, x_test, y_test, labels):
    '''
    Start a new round with a random image and random choices
    '''
    (correct_image, correct_label_index, correct_data_index) = random_image(x_test, y_test)
    correct_label = labels[correct_label_index]
    indices = random_indices(correct_label_index, labels)
    choices = random_choices(indices, labels)
    prediction_index = predicted_index(model, x_test, correct_data_index, indices)
    prediction_label = labels[prediction_index]

# test = model.predict_on_batch(x_test[0:1])
# print(test)
# index = 0
# maximum = 0
# for i in test:
#     for j in range(len(i)):
#         if i[j] > maximum:
#             index = j
#             maximum = i[j]
# print("index =", index, "maximum =", maximum)
# print("prediction =", labels[index])

print(datetime.now()-start)
