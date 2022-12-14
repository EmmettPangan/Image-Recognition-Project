from lib2to3.pytree import convert
from flask import Flask
from flask import render_template
from flask import request
from random import randint, shuffle
from tensorflow import keras

app = Flask(__name__)

# @jsf.use(app)
# class App():
#     def __init__(self):
#         self.start = False
#         self.score = 0
    
#     def start(self):
#         self.start = True
#         self.js.document.getElementByID("start").innerHTML = "Started"

c = ""
a = ""
i = ""
model_score = ""

@app.route("/")
def render_home():
    return render_template("home.html")

@app.route("/blitz")
def render_blitz():
    generate_data(200)
    return render_template("blitz.html", choices = c, answers = a, images = i)

@app.route("/mvm")
def render_mvm():
    generate_data(300)
    # # print("c=", c)
    # print("a=", a)
    # # # print("i=", i)
    # print("model_score=", model_score)
    return render_template("mvm.html", choices = c, answers = a, images = i, model = model_score)

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

def generate_data(num_rounds):
    global c, a, i, model_score, x_test, y_test, labels
    c = a = i = model_score = ""
    for round in range(num_rounds):
        if round == num_rounds-1:
            (x, y, z, s) = random_image(x_test, y_test, labels)
            c += x
            a += y
            i += z
            model_score += s
        else:
            (x, y, z, s) = random_image(x_test, y_test, labels)
            c += x + ";"
            a += y + ";"
            i += z + ";"
            model_score += s + ";"

def random_image(image_data, label_data, labels):
    global model
    converted_labs = ans = converted_im = m_score = ""

    index = randint(0, len(image_data)-1)
    im = image_data[index].flatten()
    correct_lab_index = label_data[index][0]
    ans = labels[correct_lab_index]
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
    
    shuffle(labs)

    for i in range(len(labs)):
        if i == len(labs)-1:
            converted_labs += labs[i]
        else:
            converted_labs += labs[i] + ","

    for i in range(len(im)):
        if i == len(im)-1:
            converted_im += str(im[i]) + ",255"
        elif (i+1) % 3 == 0:
            converted_im += str(im[i]) + ",255,"
        else:
            converted_im += str(im[i]) + ","
        
    m_score = predicted_label(model, x_test, index, lab_indices, labels)
    
    return (converted_labs, ans, converted_im, m_score)

def predicted_label(model, image_data, correct_data_index, indices, labels):
    '''
    Returns the predicted label for an image
    '''
    predictions = model.predict_on_batch(image_data[correct_data_index:correct_data_index+1])
    max_index = 0
    current_max = 0
    for i in indices:
        if predictions[0][i] > current_max:
            current_max = predictions[0][i]
            max_index = i
    return labels[max_index]

# @app.route("/update-mvm", methods = ["GET", "POST"])
# def update_mvm():
#     global start, score, choices, answer
#     if request.method == "POST":
#         print(request.form)
#         if request.form.get("start") == "True":
#             start = True
#             return ("", 204)
#         elif request.form.get("a") == "True":
#             check_player_answer(choices, 0, answer)
#             check_model_answer()
#             return("", 204)
#         elif request.form.get("b") == "True":
#             check_player_answer(choices, 1, answer)
#             check_model_answer()
#             return("", 204)
#         elif request.form.get("c") == "True":
#             check_player_answer(choices, 2, answer)
#             check_model_answer()
#             return("", 204) 
#         elif request.form.get("d") == "True":
#             check_player_answer(choices, 3, answer)
#             check_model_answer()
#             return("", 204)

# def check_player_answer(choices, answer, index):
#     global player_score
#     if choices[index] == answer:
#         player_score += 1

# def check_model_answer():
#     global model_score
#     model_score += 1

if __name__ == "__main__":
    app.run(host = "127.0.0.1")