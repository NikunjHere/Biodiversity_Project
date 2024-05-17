from flask import Flask, jsonify, request, render_template
from sklearn import tree
import pandas as pd

app = Flask(__name__)
classifier = tree.DecisionTreeClassifier()
print('Biodiversity includes range of species that live in an area.')

data = {
    'Ear Size': [2.4, 0.13, 4.0],
    'Body Size': [27.0, 2.0, 4.2],
    'Eyes Size': [0.17, 0.02, 0.03],
    'Heart Length': [1.5, 0.12, 0.2],
    'Endangered Animals': ['Blue Whale', 'White Tailed Deer', 'Black Rhinos']
}

df = pd.DataFrame(data)

data_features = df[['Ear Size', 'Body Size', 'Eyes Size', 'Heart Length']]
target = df['Endangered Animals']

classifier.fit(data_features, target)


def predict_species(new_features):
    # The new_features should be a list of features
    prediction = classifier.predict([new_features])
    return prediction[0]


example_features = [2.4, 27.0, 0.17, 1.5]
print('The predicted Endangered Animals is: ', predict_species(example_features))


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("11111111111111111")
    feature1 = request.form.get("feature1")
    feature2 = request.form.get("feature2")
    feature3 = request.form.get("feature3")
    feature4 = request.form.get("feature4")

    print("feature1: ", feature1)
    userfeatures = [feature1, feature2, feature3, feature4]
    print("userfeatures:", userfeatures)

    # features = [0.6, 22.5, 10.2, 0.08]
    print("222222222222222222", userfeatures)
    prediction = classifier.predict([userfeatures])
    modelPrediction = prediction[0]
    # return jsonify({'Endangered Animals': prediction[0]})
    return render_template("index.html",
                           predictedAnimal=modelPrediction)


if __name__ == '__main__':
    app.run(debug=True)
