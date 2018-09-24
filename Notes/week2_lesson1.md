# Regression
More on Regression and classification

---
Decision Boundaries:
- For linear classifiers, the decision boundary is a hyperplane of 
dimension d-1.
- vector theta is orthogonal to the decision boundary
- vector theta points in direction of region indicating +1

Linearly Separable:
- Training data Sn is linearly separable if there exists param theta and
theta not such that for all (x,y) within Sn, 
`y(theta^t * X + theta not) > 0`

Perceptron Algorithm
- Mistake driven
- initialise theta = 0
- for each data, update as per previous notes.


Loss function is the loss
eg.
- adjusts the graph line until it minimises loss
- minimising the function reduces the loss

Stochaistic and Perceptron algorithm very similar.
Differences are:
For Stochaic:
- check z <= 1 rather than z <= 0
- decrease nk rather than keep n = 1
- Select data at random rather than in sequence

! Sigmoid functions are S shaped curved


