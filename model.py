import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle  # lagring av modell
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


#laste datasettet fra csv fil
data = pd.read_csv('iris.data', header=None, 
                   names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])


#viser de første radene
print(data.head())


#visualisere med pairplot
sns.pairplot(data, hue='species', markers=["o", "s", "D"])
plt.show()


#splitter data til features og labels
X = data.drop("species", axis=1)  #features
y = data["species"]    #labels


#deler i training og test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#trener logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


#lagrer den trente modellen
with open('iri.pkl', 'wb') as file:
    pickle.dump(model, file)

#prediksjoner
y_pred = model.predict(X_test)

# evaluerer modellen
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)


#laster modellen fra .pkl filen
with open('iri.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
    #bruk den lagrede modellen til å gjøre prediksjoner
    loaded_pred = loaded_model.predict(X_test)
    print(f"Loaded Model Accuracy: {accuracy_score(y_test, loaded_pred):.2f}")