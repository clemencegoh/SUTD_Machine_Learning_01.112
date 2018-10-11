import numpy as np
import theano
import theano.tensor as Tensor
from scipy.optimize import fmin_l_bfgs_b as minimize
import matplotlib.pyplot as plt


csv = "D:\desktopStorage\school\SUTD_Machine_Learning_01.112\Homework\Week2\www.dropbox.com\s\oqoyy9p849ewzt2\linear.csv"
data = np.genfromtxt(csv, delimiter=',')

# all data into X and Y
X = data[:, 1:]
Y = data[:, 0]


def printShapes1a():
    # create validation set matrices
    vX = X[0:10, :]  # feature
    vY = Y[0:10]  # response

    # create training set matrices
    tX = X[10:, :]
    tY = Y[10:]

    print("Shapes:")
    print('vX:', vX.shape)
    print('tX:', tX.shape)
    print('vY:', vY.shape)
    print('tY:', tY.shape)


def ridgeRegression1b():
    # regularization penalty
    reg_penalty = 0.15

    # using values from homework 1...
    d = X.shape[1]
    n = X.shape[0]
    learn_rate = 0.5

    # create feature matrix
    x = Tensor.matrix(name='x')

    # create response vector
    y = Tensor.vector(name='y')

    # placeholder for w
    w = theano.shared(np.zeros((d, 1)), name='w')

    # print(w.get_value())

    main_regression = (Tensor.dot(x, w).T - y)**2 / (2 * n)
    regulizer = reg_penalty * (w[0, 0]**2 + w[1, 0]**2 + w[2, 0]**2) / (2 * n)
    reg_loss = Tensor.sum(main_regression + regulizer)

    gradient_loss = Tensor.grad(reg_loss, wrt=w)
    train_model = theano.function(inputs=[],
                                  outputs=reg_loss,
                                  updates=[(w, w - learn_rate * gradient_loss)],
                                  givens={x: X, y: Y})

    # similar to homework 1...
    n_steps = 50
    for i in range(n_steps):
        train_model()

    print(w.get_value())


def costgrad(w, x, y):
    reg_penalty = 0.15

    n = x.shape[0]
    cost = np.sum((np.dot(x, w).T - y)**2)/2/n + \
           reg_penalty * (w[0]**2 + w[1]**2 + w[2]**2) / 2

    a = np.asarray([w[0], w[1], w[2], 0])
    grad = reg_penalty * a + np.dot(np.dot(x.T, x), w)/n - \
           np.dot(x.T, y)/n

    return cost, grad


def bfgsOptimizer1c():
    global X, Y

    # data from global as above
    X = data[:, 1:]
    Y = data[:, 0]

    d = X.shape[1]
    w = np.zeros((d, 1))

    optx, cost, messages = minimize(costgrad, w, args=[X, Y])
    print(optx)


def ridge_regression(tX, tY, l):
    training_data_shape = tX.shape[0]

    # I matrix with diagonal 1 and 0 elsewhere
    i_matrix = np.eye(4)
    i_matrix[3, 3] = 0

    result = np.dot(np.dot(
        np.linalg.inv(
            training_data_shape * l * i_matrix + np.dot(tX.T, tX)),
            tX.T),
        tY)

    return result


def ridgeRegression1d():
    global X, Y
    X = data[:, 1:]
    Y = data[:, 0]
    w = ridge_regression(X, Y, 0.15)
    print(w)


def plotValidationGraph1e():
    global X, Y

    tX = X[10:, :]
    tY = Y[10:]
    vX = X[0:10, :]
    vY = Y[0:10]

    tn = tX.shape[0]
    vn = vX.shape[0]
    tloss = []
    vloss = []
    index = -np.arange(0, 5, 0.1)
    for i in index:
        w = ridge_regression(tX, tY, 10 ** i)
        tloss = tloss + [np.sum((np.dot(tX, w) - tY) ** 2) / tn / 2]
        vloss = vloss + [np.sum((np.dot(vX, w) - vY) ** 2) / vn / 2]

    plt.plot(index, np.log(tloss), 'r')
    plt.plot(index, np.log(vloss), 'b')
    plt.show()


if __name__ == '__main__':
    # printShapes1a()
    # ridgeRegression1b()
    plotValidationGraph1e()