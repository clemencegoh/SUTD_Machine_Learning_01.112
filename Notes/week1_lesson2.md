# Lecture 2: Regression

**Main areas of Machine Learning**
- Task
- Performance
- Experience

---
**Regression** <br>
Regression belongs to Supervised Learning. <br>
*Task:* Find a function such that y = f(x;theta) <br>
i.e. find the best fit line or curve (linear or non-linear) <br>
*Experience:* Require n pairs of training data, with coordinates of all dimensions required (x,y,z,...)
*Performance:* Prediction error y - f(x;theta) on test data. For this, the smaller the error, the better. <br>

**Features** <br>
In machine learning and pattern recognition, a feature is an individual measurable property or characteristic of a phenomenon being observed. <br>
*i.e. A feature is a n dimensional vector used to represent an object, such that it contains chosen properties of the object* <br>

**Predictor** <br>
The test loss R(f hat;S*) determines the performance. (S* is the test data ) <br>
Therefore given a predictor, we use the test loss to measure how well f hat generalizes to new data. <br>

For training loss, use "f" instead, since it is unoptimized. The ground truth has not been taken into account yet. During training, objective is to minimize training loss. <br>
The aim is to find a predictor f hat which minimizes the loss function, L(f;*S*n). For better performance, large training set is needed to reduce biasedness. <br>

Since the aim is generalization, there is a possiblity of underfitting and *overfitting* earlier without optimization. <br>
Therefore, **Model Selection** is critical.

---
*Overfitting:* If model H is too big, then model performs:
- well on training data, but poorly on test data

*Underfitting:* If model H is too small, then it performs:
- Poorly on training data, and poorly on test data.

---
**Optimization** <br>
! Important !
- Loss function: *typically (1/2)Z^2*. However, the power can be tweaked and experimented with.
- Empirical Risk/ Training Loss: The training loss is the average of the point losses.

**Gradient** <br>
The training gradient is the average of the point gradients. The point gradients are taken in the form of partial derivatives. Note here that the *L* represents the loss. Since the inverted triangle is the *nabla*, or the partial derivative, nabla L represents the partial derivative of loss (?). <br>
The objective of finding this nabla L is to reach the bottom of the gradient curve, such that loss is minimized. <br>

**Steps to take:** <br>
*Execute Gradient Descent* 
- Initialise theta, for [theta -  nk nabla L(theta)], where nk is the learning rate and k it the iteration number.
- Update theta accordingly until the the model converges. (When improvement in L(theta) is small enough.)
- Typical method is to gradually reduce step size.

---
**Gradient Descent** <br>
![alt text](https://cdn-images-1.medium.com/max/1600/1*f9a162GhpMbiTVTAua_lLQ.png " Local minimum Gradient Descent")

Due to being a high-dimensional data, problems do not usually have a case whereby local minimum = Global minimum.

**Stochastic Gradient Descent (Bm)** <br>
Select Bm from Sn. This involves a "mini-batch", which takes a smaller set of data (usually not more than 100), and trains the data. <br>
Steps of this are done following the same steps as the normal gradient descent, just replacing Sn with Bm. <br>

A function for learning rate which starts big and ends small: <br>
*N*k = 1/(k + 1)

<h2> ! Need to revise ! </h2>

---
- Point Gradient
    - 
- Training Gradient
    - 

- Exact Solution
    - Since the optimization problem is convex, minimum attained when gradient is zero.
    - Must have more data than features

---
**Regularization** <br>

*Ridge Regression:*

Add a penalty to simplify model. Lambda is used to reduce the effect of unimportant variables on the overall model. <br>

This is in comparisson with the least squares model, since that tends to lead to overfitting with mutliple lowest point. <br>

