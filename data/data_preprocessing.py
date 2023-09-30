import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


pd.set_option("display.max_columns", 10)

train_data = pd.read_csv(r"C:\Users\User\Desktop\datasets\titanic\train.csv")


def fill_age(obs):
    right_ages = {3: {"male": 25, "female": 21}, 2: {"male": 30, "female": 28}, 1: {"male": 40, "female": 35}}
    if np.isnan(obs.Age):
        return right_ages[obs.Pclass][obs.Sex]
    return obs.Age


train_data.Age = train_data.apply(fill_age, axis=1)
train_data.Sex = train_data.Sex.replace({"male": 1, "female": 0})

FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

X = train_data[FEATURES]
y = train_data['Survived'].values

X = pd.get_dummies(X, columns=['Pclass', 'SibSp', 'Parch'], dtype=int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#print(f"X_scaled.shape: {X_scaled.shape}, y.shape: {y.shape}")
#print(X_scaled[:10, :])
