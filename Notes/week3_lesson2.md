# Support Vector Machines

---
- Support Vectors are the boundary points supporting the 
decision function
SVM maximises decision boundary such that it gets
the fattest points.
- SVM looks for the fattest separator 
**(unique solution)**
- Vectors are points
- decision function are lines drawn using support vectors
- 

---
Primal problem (original problem)
- Dual problem
    - Solving dual optimization problem where constraints 
    are nicer, easier to implement gradient descent
- Exact solution
    - Solve Lagrangian system of equations
    
- **PRIMAL VS DUAL**
    - Primal: min (max( L(x,lambda)))
    - Dual: max(min(L(x,lambda)))
    - if p* == d*, can solve primal (harder) by solving the dual (easier)
    - In a saddle problem, they are similar
    
- Slide 12:
    - The way to solve this is to look at all instances where
    the inner holds true first, then work outwards.
    - The first example looks at a max lambda of (4, 3) if it is (1, 2).
    This is followed by looking at the min x from these, whcih brings the solution 
    to 3.
    - The seconds example looks at a min x of (1, 2).
    Maximising lambda from here will give solution of 2.
    
    
    
    
    
            
    