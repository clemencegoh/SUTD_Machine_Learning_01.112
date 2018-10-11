import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LogisticRegression


X_data = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")
X_valid = X_data.sample(frac=0.2, random_state=200)
X_train = X_data.drop(X_valid.index)
Y_data = X_data["Survived"]
Y_valid = X_valid["Survived"]
Y_train = X_train["Survived"]
ID_test = X_test["PassengerId"]


def displayData():
    display(X_data.head())
    display(X_data.describe())
    display(X_test.head())
    display(X_test.describe())


def preproces(df):
    df = df.copy()
    df.drop(["PassengerId", "Survived"], axis=1, inplace=True, errors="ignore")
    df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # todo: fill in the blanks
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Age"].fillna(df["Age"].mean(), inplace=True)

    #print(df)

    df = df.join(pd.get_dummies(df["Embarked"]))
    df.drop(["Embarked"], axis=1, inplace=True)
    df = df.join(pd.get_dummies(df["Sex"]))
    df.drop(["Sex"], axis=1, inplace=True)
    df = df.join(pd.get_dummies(df["Pclass"]))
    df.drop(["Pclass"], axis=1, inplace=True)

    df.loc[:, "Family"] = (df["Parch"] + df["SibSp"] > 0) * 1
    df.loc[:, "Child"] = (df["Age"] < 16) * 1

    return df


def automatedPreprocess3a():
    global X_train, X_valid, X_data, X_test

    X_data = preproces(X_data)
    X_test = preproces(X_test)
    X_valid = preproces(X_valid)
    X_train = preproces(X_train)

    # display(X_data.head())
    # display(X_test.head())
    # display(X_valid.head())
    display(X_train.head())


def logisticRegression3b():
    global X_valid, Y_valid, X_train, Y_train
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, Y_train)

    print(logistic_reg.score(X_valid, Y_valid))


def findParameter3c():
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, Y_train)

    Y_test = logistic_reg.predict(X_test)

    coeff_df = pd.DataFrame(X_data.columns.delete(0))
    coeff_df.columns = ['Features']
    coeff_df['Coefficient Estimate'] = pd.Series(logistic_reg.coef_[0])

    return Y_test


def predictLabel3d():
    Y_test = findParameter3c()

    ans = pd.DataFrame({"PassengerId": ID_test, "Survived": Y_test})
    ans = pd.DataFrame({"PassengerId": ID_test, "Survived": Y_test})
    ans.to_csv("submit.csv", index=False)



if __name__ == '__main__':
    # automatedPreprocess3a()
    # logisticRegression3b()
    # findParameter3c()
    predictLabel3d()
