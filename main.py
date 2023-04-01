# This file implements basic web app functionality using Flask,
# creating a home page and two game mode pages.

from flask import Flask
from flask import render_template
from random import randint, shuffle
from tensorflow import keras

app = Flask(__name__)

# These variables will be used to preload game data into each
# game page so that they can be played asynchronously.
# c contains every set of four answer choices to be displayed to the user.
# a contains the correct answer choice for every image.
# i contains the pixel data for every image to be displayed.
# model_predictions contains the prediction of the model for every image.
c = ""
a = ""
i = ""
model_predictions = ""

@app.route("/")
def render_home():
    return render_template("home.html")

# Generate data for 150 rounds, with each one having answer choices,
# a correct answer, and the corresponding image to be displayed.
# This data is passed to the page using Jinja2 templating.
@app.route("/blitz")
def render_blitz():
    generate_data(150)
    return render_template("blitz.html", choices = c, answers = a, images = i)

# Generate data for 150 rounds, with each one having answer choices,
# a correct answer, and the corresponding image to be displayed.
# This data is passed to the page using Jinja2 templating.
@app.route("/mvm")
def render_mvm():
    generate_data(150)
    return render_template("mvm.html", choices = c, answers = a, images = i, model = model_predictions)

def unpickle(file, enc):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = enc)
    return dict

def master_labels():
    return unpickle("CIFAR-100/meta", "ASCII")["fine_label_names"]

# Unpack labels, training data, testing data, and the pre-trained model.
labels = master_labels()
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode = "fine")
model = keras.models.load_model("image-recognition-model")

def generate_data(num_rounds):
    '''
    Returns the data for some number of rounds,
    including the answer choices, correct answers, and image data for each round.
    '''
    global c, a, i, model_predictions, x_test, y_test, labels
    c = a = i = model_predictions = ""
    for round in range(num_rounds):
        # Concatenate the data for each round to the running strings c, a, i, and model_predictions,
        # with rounds delimited by semicolons.
        if round == num_rounds-1:
            (x, y, z, s) = random_image(x_test, y_test, labels)
            c += x
            a += y
            i += z
            model_predictions += s
        else:
            (x, y, z, s) = random_image(x_test, y_test, labels)
            c += x + ";"
            a += y + ";"
            i += z + ";"
            model_predictions += s + ";"

def random_image(image_data, label_data, labels):
    '''
    Tests the model on a random image with 4 answer choices,
    returning the answer choices, correct answer, image data, and model prediction
    '''
    global model
    converted_labs = ans = converted_im = ""

    # Randomly select an image and retrieve its image data.
    index = randint(0, len(image_data)-1)
    im = image_data[index].flatten()

    # Determine the correct classification for the iamge.
    correct_lab_index = label_data[index][0]
    ans = labels[correct_lab_index]

    # Generate a list for the index position of every selected label choice, and
    # generate a list for the actual label choices themselves. These will include
    # the correct label as well as alternate, incorrect answer choices.
    lab_indices = list()
    lab_indices.append(correct_lab_index)
    labs = list()
    labs.append(ans)
    while len(labs) < 4:
        potential_index = randint(0, len(labels)-1)
        potential_label = labels[potential_index]
        if potential_label not in labs:
            lab_indices.append(potential_index)
            labs.append(potential_label)
    
    # Shuffle the answer choices.
    shuffle(labs)

    # Convert the answer choices to a string, with choices delimited by commas.
    for i in range(len(labs)):
        if i == len(labs)-1:
            converted_labs += labs[i]
        else:
            converted_labs += labs[i] + ","

    # Convert the image data to a string, with RGB values delimited by commas.
    # Individual pixels are also delimited by commas and can be determined
    # by from every set of three values, which correspond to RGB.
    for i in range(len(im)):
        if i == len(im)-1:
            converted_im += str(im[i]) + ",255"
        elif (i+1) % 3 == 0:
            converted_im += str(im[i]) + ",255,"
        else:
            converted_im += str(im[i]) + ","
        
    # Make a prediction on the image using the model.
    prediction = predicted_label(model, x_test, index, lab_indices, labels)
    
    return (converted_labs, ans, converted_im, prediction)

def predicted_label(model, image_data, correct_data_index, indices, labels):
    '''
    Returns the predicted label for an image given the indices of the possible answer choices.
    '''
    predictions = model.predict_on_batch(image_data[correct_data_index:correct_data_index+1])
    max_index = 0
    current_max = 0
    for i in indices:
        if predictions[0][i] > current_max:
            current_max = predictions[0][i]
            max_index = i
    return labels[max_index]

if __name__ == "__main__":
    app.run(host = "127.0.0.1")