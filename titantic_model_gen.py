#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import math

#%%
def corrGenerator(corr):
    # colormap = sns.diverging_palette(220, 10, as_cmap = True)
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title("Feature Correlation", y=1.05, size=15)
    sns.heatmap(corr,linewidths=0.1,vmax=1.0, 
                square=True, cmap=colormap, linecolor="white", annot=True)

#%%
import re

def get_title(name):
    title_search = re.search(" ([A-Za-z]+)\.", name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

#%%
def preprocess_data(data):
    # Convert String to binary integer for sex
    data["_Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["_Cabin"] = data["Cabin"].apply(lambda val: 1 if str(val) != "nan" else 0)
    data["Age"].fillna(data['Age'].median(), inplace=True)
    data["Embarked"].fillna(data['Embarked'].mode()[0], inplace=True)
    data["Cabin"].fillna("Unknown", inplace=True)
    data["Fare"].fillna(data['Fare'].median(), inplace=True)

    # Verify that the "Cabin" column was modified
    embark = data.Embarked.unique()
    embark = dict((str(value), index) for index, value in enumerate(embark))
    data["_Embarked"] = data["Embarked"].map(embark)
    
    data["_IsAlone"] = [0 if data["SibSp"][index] + data["Parch"][index] == 0 else 1 for index in range(len(data))]
    data["_NameLength"] = data["Name"].apply(lambda name: len(name))
    data["_Fare"] = data["Fare"].apply(lambda value: math.log10(value) if value > 0 else 0)
    data["_Title"] = data["Name"].apply(lambda name: get_title(name))
    data["_Title"] = data["_Title"].replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    data["_Title"] = data["_Title"].replace("Mlle", "Miss")
    data["_Title"] = data["_Title"].replace("Ms", "Miss")
    data["_Title"] = data["_Title"].replace("Mme", "Mrs")
    data["_Title"] = data["_Title"].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    print(data.columns[data.isnull().any()])

#%%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

preprocess_data(train)
preprocess_data(test)
#%%
corrGenerator(train.corr())
#%%
filterCol = []
for col in train.columns:
    if train[col].dtype != object and abs(train["Survived"].corr(train[col])) > 0.1:
        filterCol.append(col)
filterCol.remove("Survived")
print(filterCol)
#%%
y_train = train["Survived"].ravel()
x_train = np.array(train[filterCol])
print("X_train shape:", x_train.shape)
print("Y_train shape:", y_train.shape)

#%%
x_test = np.array(test[filterCol])
print("X_test shape:", x_test.shape)

#%%
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
#%%
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=2019)

#%%
n_estimators = [20, 30, 40, 50, 60, 70, 80]
for num in n_estimators:
    clf = AdaBoostClassifier(n_estimators=num, random_state=2019)
    clf.fit(x_train, y_train)
    print("Num:", num, "Score:", clf.score(x_valid, y_valid))
#%%
learning_rate = [0.925, 0.95, 0.975, 1, 1.025, 1.1, 1.5]
for num in learning_rate:
    clf = AdaBoostClassifier(random_state=2019, learning_rate=num, n_estimators=40)
    clf.fit(x_train, y_train)
    print("LR:", num, "Score:", clf.score(x_valid, y_valid))
#%%
for algo in ["SAMME", "SAMME.R"]:
    clf = AdaBoostClassifier(random_state=2019, n_estimators=40, algorithm=algo)
    clf.fit(x_train, y_train)
    print("Algorithm:", algo, "Score:", clf.score(x_valid, y_valid))
#%%
