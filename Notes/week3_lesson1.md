# Recommendation


Examples:
- Missing sensor data
- **Input incomplete matrix => output complete matrix**
- Collaborative filtering
    - Example comes in when we have 400k users, 17k movies
    - 1% give ratings
    - Therefore given such data, a training data set can be formed. 
    - Incomplete matrix Y is used as the base to predict unobserved ratings.
- Matrix/Tensor completion problems
    - Collaborative: Cross-users
    - Filtering: prediction
    - Dimensionality reduction
    - **A Tensor is a multi-dimensional array**
        - Example: nxm is a 2-tensor
        - pxqxr is a 3-tensor
- Use regression => the numbers have a relationship with each other
    - regression is used for collaborative filtering due to relationship
    - Key assumption of collaborative filtering: <br>
    **Users who share similar interests on an item in the past are more likely to 
    hold similar opinions on other items compared with a randomly chosen user.**

---
Types of recommendations
- Model based
    - Given training data of the form ((a, i), Yai), find a function
    f:{1,...,n} x {1,...,m} -> R.
- Memory based
    - Given incomplete user ratings Ya in R^m, find structure in
    the data to predict missing values.
   
**Both give a recommendation**

---
K-Nearest Neighbours
    - Memory based
    - Main idea is to find a few users that are similar to user a
    - use information from users b1,...,bk to predict the ratings
    of user a

---
Correlation Coefficient
- use corr for finding similarities between users
- Slides 10-11
- Important formula in images

---
K-Nearest-Neighbour
- Look in images for formula
- Formula not sensitive to bias for each user
    - i.e during testing it is intensive, since there is no training loss and
    hence no training algorithm for kNN
    - testing is very slow since model is not trained. 

---
**Matrix Factorization**

---
Subspace learning
- Completed rating vectors lie in some k-dimensional space
- Find complete matrix Y hat such that it is the closest complete matrix
to incomplete matrix Y. [Close to observed members]
-  Finding low rank approx of Y
- Y hat,ai = U,a V,i
- k factors
- 



