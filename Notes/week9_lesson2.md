Difference:
- In discriminative model interested in P(Y|X)
    - Do not care how X is produced
- In generative model, interested in both X and Y.
    - Therefore, P(X,Y)
- Rewriting probability: P(a, b, c, y=+1)
    - = P(y=+1).P(a|y=+1).P(b|y=+1).P(c|y=+1)
- Revision: *Theta*a- = (Count(a,y=-1))/(Sum(Count(words, y=-1))) 
- Revision: *Pi*- = (Count(y=-1)*Theta*a-)/(Count(y=+1)+Count(y=-1))


---
Hidden Markov Model: (Analysis)
- Y label, and X words
- Insert Y0 and Y(N+1) for "Start" and "Stop"
- Each Prob of Y is dependent on the prev Y
- Each Prob of X is dependent on corresponding Y


- Generative:
    - Generate y1 from y0 (Prob of 1)
        - Generate x1 from y1
    - Generate y2 from y1
        - Generate x2 from y2
    - etc.
    - Total probability => P(x1,...|y0,y1...,yn+1)
    - Equals to: 
    Product(P(yi|yi-1), from i=1;i<n+1;i++) *dot*
    Product(P(xi|yi), from i=1;i<n;i++)
    - This is: Transition *AND* Emission
    - Also simplified to: A(yi+1,yi).B(yi(xi))
        - eg. A(y0y1) & Ba("the")
        
- Note that Y* = Argmax*y*(P(Y|X)) 
= argmax P(x,y)/P(x)
= argmax P(x,y), since P(x) is not 
involved in the argmax of Y
     
    
    