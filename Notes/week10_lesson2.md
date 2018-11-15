# Brief Revision - HMM

1. Supervised Learning: Given X,Y, find *theta*={a,b}
2. Decoding: Given *theta*={a,b}, find Y.
3. Unsupervised Learning: Given X, find: Y/*theta*={a,b}
    - Hard EM
    - Soft EM
        - E step: find membership
        - M step: update params
        
        
Overall: 
1. Count(u,v) = SUM(n, j=0, P(Yj=u,Yj+1=v|X)) = Sum(n, j=0,*alpha*u(j).Bu(Xj).Au,v.*beta*(j+1)/
                                                    Sum(v, *alpha*u,v.*beta*v(k)))
2. Count(u) = SUM(n, j=1, P(yj=u|X)) = Sum(n, j=1, *alpha*u(j).*beta*u(j)/Sum(*alpha*v(k).*beta*v(k)))
3. Count(START) = 1
4. Count(STOP) = 1
5. Au,v = Count(u,v)/Count(u)           
    - **a**       => transition state u => v
6. Bu(o) = Count(u->o)/Count(u)         
    - **b**       => emission x from current state
7. *alpha*u(j) = P(x1,....xj-1,yj=u)    
    - **alpha**   => "past"
8. *beta*u(j) = P(xj,...xn | yj=u)      
    - **beta**    => "future"
9. *alpha*.*beta* gives most optimal path
    - Sum(*alpha*.*beta*) gives every possible path


For both counts Count(u,v) and Count(u), can be further decomposed down
as long as there exists up till Xj-1,Yj, and Yj+1=v till Xn.



**Backward:**
1. Base case: 
- *beta*u(n) = Au,STOP.Nu(Xn)
2. Recursive case:
- *beta*u(j) = Sum(*beta*v(j+1). Bu(Xj).Au,v)


1. E-step: q^(t+1) = argmax,q F(q,*theta*^t) ==> set q(y) to P(Y|X)
2. M-Step: *theta*^(t+1) 
    - = argmax,*theta* F(q^(t+1),*theta*)


Alternative way of decoding:
- y1*,....yn* = argmax,y1...yn P(y0,y1,....,yn,yn+1 | x1,....xn)
- for i = 1,...,n: yi* = argmax,yi P(yi | x1,...xn)

