# Steps for Clustering Algorithms

1. K-Means
- Given hard labels, compute centroids
- Given centroids, compute hard labels
2. Expectation-Maximisation
- Given soft labels, compute gaussians
- Given Gaussians, compute soft labels

Soft labels:
- Instead of being 100% sure of which point belongs to which cluster,
we make it an estimate.
    - Example: point A: 95% cluster A, 5% cluster B
    
**Recap:**
- In a spherical gaussian, covariance is n*I*
(diagonal all the same)


EM:
- Expectation Maximisation

Working with hidden labels: (Mixture model)
- Previously, when the labels are known, just use
each point to calculate gaussian
- With this, there are 2 steps:
    - E: for i=1,...,n, fill in missing data y(i) - 
    the label according to what is most likely given 
    the current model *miu*
    - M: run ML for completed data, which gives the 
    new model *miu*
---
Numerical Algorithm:
1. Initialise params *theta*
2. Repeat till convergence:
    - E: given param *theta*, compute soft labels p(y|x)
    - M: Given soft labels p(y|x), compute params *theta*
---

Baynesian Information Criterion:
- Determine number of gaussians 
- Dangerous to consider only log likelihood due to 
**overfitting**
- BIC is used instead to account for number of params














