import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import os

file_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_path)

# Ler arquivo csv
train_df = pd.read_csv("cancer_patient_data.csv")

tomography_result = ["0", "1", "2"]  # 0 = baixa; 1 = media; 2 = alta probabilidade de ter cancer de pulmao
X = train_df.drop("Level", axis=1)
y = train_df["Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
sel.fit(X_train, y_train)

print('Most relevant features: ')
print(list(X.columns[sel.get_support()]))

# Tratamento dos dados do arquivo lung_cancer.csv
# columns = ["Age", "Gender", "Air Pollution", "Alcohol use", "Dust Allergy", "Occupational Hazards", "Genetic Risk",
#           "Chronic Lung Disease", "Balanced Diet", "Obesity", "Smoking", "Passive Smoker", "Chest Pain",
#           "Coughing of Blood", "Fatigue", "Weight Loss", "Shortness of Breath", "Wheezing", "Swallowing Difficulty",
#           "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough", "Snoring"]
columns = list(X.columns)
features = train_df[columns].values

# Aprendizado árvore de decisão
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf = clf.fit(X_train, y_train)

# Plot tree figure
fig = plt.figure(figsize=(25, 20))
tree.plot_tree(clf, feature_names=columns, class_names=tomography_result, filled=True)

fig.savefig("decision_tree.png")

# Características do paciente: 0 para "Não" e 1 para "Sim"
age = 60
gender = 1
air_pollution = 3
alcohol_use = 4
dust_allergy = 5
occupational_hazards = 5
genetic_risk = 8
chronic_lung_disease = 2
balanced_diet = 2
obesity = 2
smoking = 1
passive_smoker = 7
chest_pain = 5
coughing_blood = 4
fatigue = 3
weight_loss = 3
shortness_breath = 7
wheezing = 2
swallowing_difficulty = 5
clubbing_finger_nails = 3
frequent_cold = 4
dry_cough = 6
snoring = 5

# Exibe se o paciente provavelmente tem ou não câncer com base no histórico clínico de outros pacientes
pacient = [age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards, genetic_risk,
           chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker, chest_pain,
           coughing_blood, fatigue, weight_loss, shortness_breath, wheezing, swallowing_difficulty,
           clubbing_finger_nails, frequent_cold, dry_cough, snoring]
print('Predict: ')
print(clf.predict([pacient]))

# Exibe a distribuição estatística das classes, isto é, onde o nosso paciente analisado se encaixaria entre os outros
# casos.
print('Probability: ')
print(clf.predict_proba([pacient]))

print('Confusion matrix: ')
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))

print('Report: ')
print(classification_report(y_test, y_pred))

# Análise de acurácia
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)

print('Final accuracy on test data: ')
print(accuracy)
