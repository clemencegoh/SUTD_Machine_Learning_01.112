# Feature mapping

Mapping a 2-dim in 6-dim space.

Bad news:
---
In a high dimensional space, it makes the feature long
- High dimensional features represented by *phi()*

What is a Kernel?
---
- Measure of similarity
- Linear Kernel:
    - K(x,x') = x.x'
- poly kernel:
    - K(x, x') = (x.x' + 1)^k
- Radial Basis Kernel (Gaussian Kernel) 
**(Most common/ widely used)**
    - K(x, x') = exp(-(1/2*sig*^2) *abs*(x - x')^2)

**https://en.wikipedia.org/wiki/Radial_basis_function_kernel**    

Kernel Trick in SVM
---
- Refers to the strategy of converting a learning algorithm
and the resulting predictor into ones that involve only computation 
of the kernel K(x, x'), but not of the feature map *phi*(x).
- For any given dimensional data, there exists a dimensional 
space **high enough** which converts the non-linear regression line
into a linear regression (hyperplane)

Convolutional Neural Network
---
- Convolutional layer composed of many convolutional filters
- They make use of "frames" to detect pixels to determine
how similar each image is to guess based on probability
- A convolutional layer has a number of filters that does 
convolutional operation. 
    - This acts as weights into the next layer
    - Backpropogation
- Convolution is the same thing as dot product (essentially AND)
    - The output of the dot product is a number which outputs
    to the next layer
    - The number goes through activation function to determine if
    present or not.
    - Weighted sum is the input of the next layer
    - Activation functions could be **ReLU** (Rectified Linear Unit), 
    **sigmoid**, etc.
- Usually each filter detects a small pattern (3x3).
- **STRIDE** - step size. 
    - Process:
        - Use filter, start from top left. Dot product => next layer
        - Stride, repeat.
- The leftover matrix determines if there has been a pattern detected
in the image. This is a **Feature Map**
- This process can be repeated using multiple filters
- This maps from a higher dimension into a lower dimension.
- The dimensions of the final map depends onL
    - number of filters used
    - step size
    - original size
    - filter size
- A classic example of this is the RGB scheme, whereby 3 filters
are put together and formed into a single colour image.

Convolution VS Fully Connected:
---
- Convolution:
image => **convolution()** => matrices
    - Output is a feature map
        - In a feature map, the *nodes/weights* are not connected
        to every node before that. (Whereas in the fully connected,
        it is.)
        - This means that we have fewer parameters

- Fully Connected => 
**transpose/concat into 1xN matrix()** => connected network
    - Output is a vector

Pooling Layer
---
- Max pooling refers to grouping regions, and getting the max 
value of each window. **Essentially, compression.**
- This is done to reduce number of parameters.
- Also affected by step size.

Flatten
---
- Flattening refers to converting resulting matrix into 1xN vector. 
- This allows the vector to go through the fully-connected neural 
network. (Old Machine Learning, **feedforward**)



