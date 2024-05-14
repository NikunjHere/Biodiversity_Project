# Import necessary libraries
from sklearn import tree
import pandas as pd

animal_input = float(input('Enter the Ear Size, Body Size, Eyes Size and Heart Length of the animal using \',\' in between each data entry: '))
plant_input = float(input('Enter the Stem Width, Stem Length, Root Length and Leaf Size of the plant using \',\' in between each data entry: '))

# Sample dataset: Features of different species (e.g., size, color, habitat)
# Replace this with your actual data

print('Biodiversity includes range of species that live in an area.')

data = {
    'Ear Size': [2.4, 0.13, 4.0],
    'Body Size': [27.0, 2.0, 4.2],
    'Eyes Size': [0.17, 0.02, 0.03],
    'Heart Length': [1.5, 0.12, 0.2],
    'Endangered Animals': ['Blue Whale', 'White Tailed Deer', 'Black Rhinos']
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Define the features and the target variable
features = df[['Ear Size', 'Body Size', 'Eyes Size', 'Heart Length']]
target = df['Endangered Animals']

# Initialize the decision tree classifier
classifier = tree.DecisionTreeClassifier()

# Train the classifier with the dataset
classifier.fit(features, target)

# Function to predict the species
def predict_species(new_features):
    # The new_features should be a list of features
    prediction = classifier.predict([new_features])
    return prediction[0]

# Example usage of the function
# Replace the list with actual features to get the species prediction
example_features = [animal_input]  # Example feature set
print('The predicted Endangered Animals is: ',predict_species(example_features))

plants_data = {
    'Stem Width': [0.6, 0.05, 0.7],
    'Stem Length': [22.5, 2.2, 15.5],
    'Root Length': [10.2, 1.5, 2.5],
    'Leaf Size': [0.08, 0.4, 0.1],
    'Endangered Plants': ['White Pine Tree', 'Sunflower', 'Ebony']
}

df = pd.DataFrame(plants_data)

# Define the features and the target variable
plant_feature = df[['Stem Width', 'Stem Length', 'Root Length', 'Leaf Size']]
plant_target = df['Endangered Plants']

# Initialize the decision tree classifier
classifier = tree.DecisionTreeClassifier()

# Train the classifier with the dataset
classifier.fit(plant_feature, plant_target)

# Function to predict the species
def predict_plant(new_plant):
    # The new_features should be a list of features
    plant_prediction = classifier.predict([new_plant])
    return plant_prediction[0]

# Example usage of the function
# Replace the list with actual features to get the species prediction
example_plants = [plant_input]  # Example feature set
print('The predicted Endangered Plant is: ',predict_plant(example_plants))
