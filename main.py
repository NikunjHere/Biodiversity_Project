from flask import Flask, request, render_template
from sklearn import tree
import pandas as pd

app = Flask(__name__)
classifier = tree.DecisionTreeClassifier()
data = {
    'Ear Size (cm)': [2.4, 0.13, 4.0, 0.05, 1.2, 0.17],
    'Body Size (cm)': [27.0, 2.0, 4.2, 10.3, 20.1, 5.1],
    'Eyes Size (cm)': [0.17, 0.02, 0.03, 0.15, 0.8, 0.12],
    'Heart Length (cm)': [1.5, 0.12, 0.2, 2.1, 1.2, 2.4],
    'Endangered Animals': ['Blue Whale', 'White Tailed Deer', 'Black Rhinos', 'Hawksbill turtle',
                           'Sumatran elephant', 'Sunda tiger']
}

df = pd.DataFrame(data)

data_features = df[['Ear Size (cm)', 'Body Size (cm)', 'Eyes Size (cm)', 'Heart Length (cm)']]
target = df['Endangered Animals']

classifier.fit(data_features, target)


def predict_species(new_features):
    prediction = classifier.predict([new_features])
    return prediction[0]


example_features = [2.4, 27.0, 0.17, 1.5]
print('The predicted Endangered Animals is: ', predict_species(example_features))


@app.route("/")
def home():
    return render_template("index.html")

vari="The reason behind the extinction of this animal is hunting it in a way that negatively impacts the species of it, and hunted for their pelts, bones, teeth, and claws, which are then sold on the black market."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    feature1 = request.form.get("feature1")
    feature2 = request.form.get("feature2")
    feature3 = request.form.get("feature3")
    feature4 = request.form.get("feature4")

    print("feature1: ", feature1)
    userfeatures = [feature1, feature2, feature3, feature4]
    print("userfeatures:", userfeatures)

    prediction = classifier.predict([userfeatures])
    modelprediction = prediction[0], (vari)
    return render_template("index.html", predictedAnimal=modelprediction)


if __name__ == '__main__':
    app.run(debug=True)
