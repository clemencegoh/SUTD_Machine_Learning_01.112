import numpy as np
import matplotlib.pyplot as plt


def showGraph(_plt):
    _plt.show()


def plotGraphData():
    csv = 'kernel.csv'
    data = np.genfromtxt(csv, delimiter=',')

    X = data[:, 1:]
    Y = data[:, 0]
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    return X, Y, plt


def part1():
    _, _, plt = plotGraphData()
    showGraph(plt)


from sklearn import svm


# this is for part b
def decision(_x1, _x2, _clf):
    x = np.array([[_x1, _x2]])
    return _clf.decision_function(x)[0]


# function to train.csv SVM based on scikit package
# set gamma to 0.5, kernel to rbf
def trainSVM():
    # part a
    clf = svm.SVC(gamma=0.5, kernel='rbf')
    X, Y, plt = plotGraphData()
    clf.fit(X, Y)
    return clf, X, Y


# part c
def visualiseClassifier():
    clf, X, Y = trainSVM()
    vdecision = np.vectorize(decision, excluded=[2])
    x1list = np.linspace(-8.0, 8.0, 100)
    x2list = np.linspace(-8.0, 8.0, 100)
    X1, X2 = np.meshgrid(x1list, x2list)
    Z = vdecision(X1, X2, clf)
    cp = plt.contourf(X1, X2, Z)
    plt.colorbar(cp)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='gray')
    plt.show()


if __name__ == '__main__':
    clf, X, Y = trainSVM()
    visualiseClassifier()
