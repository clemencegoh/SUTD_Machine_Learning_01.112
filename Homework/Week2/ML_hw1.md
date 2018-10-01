# Machine Learning Homework Week 1

Name: Clemence Goh (1002075) <br>
Talked to: Cheryl Goh
--- 
    
2. a) 
```
Python version: 3.6.2  
Theano version: 1.0.2
```
      
3. a)
```ipnbpython
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


csv = 'https://www.dropbox.com/s/oqoyy9p849ewzt2/linear.csv?dl=1'
data = np.genfromtxt(csv, delimiter=',')
X = data[:, 1:]
Y = data[:, 0]

d = X.shape[1]
n = X.shape[0]
learn_rate = 0.5

x = T.matrix(name='x')
y = T.matrix(name='y')
w = theano.shared(np.zeros((d, 1)), name='w')

risk = T.sum((T.dot(x, w).T - y)**2)/2/n
grad_risk = T.grad(risk, wrt=w)

train_model = theano.function(inputs=[],
                              outputs=risk,
                              updates=[(w, w-learn_rate*grad_risk)],
                              givens={x: X, y: Y})

n_steps = 50
for i in range(n_steps):
    print(train_model())
print(w.get_value())


"output = [[−0.57392068] \
 [ 1.35757059] \
 [ 0.01527565] \
 [−1.88288076]]"     
```
3. b) The answer is the same as 3 a)  
```python
import numpy as np


# data...
csv = 'https://www.dropbox.com/s/oqoyy9p849ewzt2/linear.csv?dl=1'
data = np.genfromtxt(csv, delimiter=',')
X = data[:, 1:]
Y = data[:, 0]

np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

# output for exact solution
"output = array([−0.57392068,\
                1.35757059,\
                0.01527565,\
                −1.88288076])" 
```
3. c) The answer is the same as the answer in 3 a) and 3 b)
```ipnbpython
from sklearn import linear_model 
import numpy as np


# data...
csv = 'https://www.dropbox.com/s/oqoyy9p849ewzt2/linear.csv?dl=1'
data = np.genfromtxt(csv, delimiter=',')
X = data[:, 1:]
Y = data[:, 0]

# create linear regression
regr = linear_model.LinearRegression(fit_intercept=False) 
regr.fit(X, Y) 

# print output
print(regr.coef_) 

# output
"output = (array([−0.57392068,\
                    1.35757059,\
                    0.01527565,\
                    −1.88288076]))"
```
3. d) Program output is very close to output from previous 3 parts.
```python
from theano import shared
from numpy.random import shuffle
import numpy as np
import theano
import theano.tensor as T


# data...
csv = 'https://www.dropbox.com/s/oqoyy9p849ewzt2/linear.csv?dl=1'
data = np.genfromtxt(csv, delimiter=',')
X = data[:, 1:]
Y = data[:, 0]

d = X.shape[1]
n = X.shape[0]
learn_rate = 0.5

x = T.matrix(name='x')
y = T.matrix(name='y')
w = theano.shared(np.zeros((d, 1)), name='w')

risk = T.sum((T.dot(x, w).T - y)**2)/2/n
grad_risk = T.grad(risk, wrt=w)

n_steps = 50

# create minibatch
minibatch_size = 5 
n_batches = n/minibatch_size 
index = T.lscalar () 
train_x = shared(np.array(X)) 
train_y = shared(np.array(Y)) 

train_model = theano.function(inputs=[index], outputs=risk, 
    updates=[(w, w - learn_rate * grad_risk)], 
    givens={
        x: train_x[index * minibatch_size: (index + 1) * minibatch_size], 
        y: train_y[index * minibatch_size: (index + 1) * minibatch_size]})
         
# loop through n_steps to parse data
for i in range(n_steps): 
    learn_rate = learn_rate / (i + 1) 
    arranged_n = np.arange(n) 
    shuffle(arranged_n)
    
    # create list for X and Y 
    X = [X[j] for j in arranged_n] 
    Y = [Y[j] for j in arranged_n]
     
    train_x = shared(np.array(X)) 
    train_y = shared(np.array(Y)) 
    
    
    for k in range(n_batches): 
        train_model(k) 
        
print(w.get_value()) 
    
# output    
output = "[[−0.57205036]\
        [ 1.35861364]\
        [ 0.01744707]\
        [−1.88281916]]"
```