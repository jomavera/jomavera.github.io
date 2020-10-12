# Interior point method example
### We must solve the following linearization of the conditions derived in the barrier problem.


$$\begin{aligned}
\mu X^{-2} \mathbf{d}_x +  \mathbf{d}_s = \mu X^{-1} \mathbf{1}- \mathbf{c} \qquad (1) \\
A\mathbf{d}_x = 0  \qquad  (2) \\
-A^T\mathbf{d}_y-\mathbf{d}_s=0 \qquad (3) \\
(AX^2A^T)\mathbf{d}_y = - \mu AX\mathbf{1}+AX^2\mathbf{c} \qquad (4) \\
\end{aligned}$$


### were $X$ is the diagonal matrix where the entries are the components $\mathbf{x}>0$.

### Given the following LP

$$
\begin{aligned}
\min  & -x_1 - 4x_2 & \\
s.a. & \quad 2x_1 - x_2 &\geq  0\\ 
&\quad x_1 - 3x_2  &\leq  0 \\
&\quad x_1 +  x_2  &\leq  4 \\
&\quad x_1, x_2  & \geq  0 \\
\end{aligned}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

x  = np.linspace(0, 4, 100)
y1 = 2*x
y2 = x/3
y3 = 4 - x
plt.figure(figsize=(8, 6))
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.xlim((0, 3.5))
plt.ylim((0, 4))
plt.xlabel('x1')
plt.ylabel('x2')
y5 = np.minimum(y1, y3)
plt.fill_between(x[:-25], y2[:-25], y5[:-25], color='red', alpha=0.5)
```

![problem_fig](images/problem.png)

### If we express the LP in standard form we have five variables.

$$
\begin{aligned}
\min  & -x_1 - 4x_2 & \\
s.a. & \quad 2x_1 - x_2 -x_3  &=  0\\
& \quad x_1 - 3x_2 +x_4  &=  0 \\
& \quad x_1 +  x_2+x_5  &= 4 \\ 
& \quad x_1, x_2, x_3, x_4, x_5  & \geq  0 \\
\end{aligned}
$$

### We see that $(x_1,x_2)=(1,1)$ is an interior point thus, we choose it as initial point $\mathbf{x}_0$. The vector $\mathbf{x}_0$ and matrix $A$ will be as follows

$$
\mathbf{x}_0=\begin{bmatrix}
1\\
1\\
1\\
2\\ 
2
\end{bmatrix}
A= \begin{bmatrix} 2 & -1 & -1 & 0 & 0\\
1& -3 & 0 & 1 & 0\\
1 & 1 & 0 & 0 & 1
\end{bmatrix}
$$
### The initial vector will have an initial solution $f(\mathbf{x}_0)=-5$

```python
x = np.linspace(0, 4, 100)
y1 = 2*x
y2 = x/3
y3 = 4 - x
plt.figure(figsize=(8, 6))
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.xlim((0, 3.5))
plt.ylim((0, 4))
plt.xlabel('x1')
plt.ylabel('x2')
y5 = np.minimum(y1, y3)
plt.fill_between(x[:-25], y2[:-25], y5[:-25], color='red', alpha=0.5)
plt.scatter([1],[1],color='black')
plt.annotate('x_0',(1.05,1.05))
```

![problem_fig](images/init_problem.png)

### The function to run $n$ iterations of interior point method is as follows,

```python
def interior_point_2D(A,c, x_init,mu, gamma,n_iter):
    x = x_init
    x1s      = [] #Empty list to save x_1's
    x2s      = [] #Empty list to save x_2's 
    x1s.append(x[0,0])
    x2s.append(x[1,0])
    n = A.shape[1]
    X = np.zeros((n,n))
    
    for iteracion in range(n_iter):
        for i in range(n):
            X[i,i] = x[i,0]
            
        #SOLVE EQUATION 4
        left_ec_4 = np.matmul( A, np.matmul( np.power(X,2),A.T ) )

        #                     -mu*A*X*1                       +       AX^2c
        right_ec_4 = -mu*np.matmul( A,np.matmul( X,vector_1 ) ) + np.matmul( A,np.matmul( np.power(X,2),c ) )
        dy = np.linalg.solve(left_ec_4, right_ec_4)


        #SOLVE EQUATION 3
        ds = np.matmul(-1*A.T,dy) #ds=-A^T*dy


        #SOLVE EQUATION 1
        left_ec_1  = mu*np.power(np.linalg.inv(X),2) #mu*X^-2
        right_ec_1 = mu*np.matmul(np.linalg.inv(X),vector_1)-c-ds #mu*X^-1*1-c-ds
        dx       = np.linalg.solve(left_ec_1,right_ec_1)


        #UPDATE vector x
        x  = x + dx
        mu = mu*gamma
        x1s.append( x[0,0] )
        x2s.append( x[1,0] )

        return x1s, x2s
```

### Running function for 100 iterations on problem

```python
mu       = 100
gamma    = 0.8
A        = np.array([[2,-1,-1,0,0],[1,-3,0,1,0],[1,1,0,0,1]])
vector_1 = np.ones((5,1))
c        = np.array([[-1],[-4],[0],[0],[0]])
x        = np.array([[1],[1],[1],[2],[2]]) #Initial point
x1s, x2s = interior_point(A,c, x,mu, gamma,100)
```

### Plotting all the iterations

```python
x  = np.linspace(0, 4, 100)
y1 = 2*x
y2 = x/3
y3 = 4 - x
plt.figure(figsize=(8, 6))
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.xlim((0, 3.5))
plt.ylim((0, 4))
plt.xlabel('x1')
plt.ylabel('x2')
y5 = np.minimum(y1, y3)
plt.fill_between(x[:-25], y2[:-25], y5[:-25], color='red', alpha=0.5)

for iteracion in range(100):
    plt.scatter(x1s[iteracion],x2s[iteracion],color='black')
    if iteracion % 10 == 0:
        nombre = 'x_'+str(iteracion)
        plt.annotate(nombre,(x1s[iteracion]+0.05,x2s[iteracion]+0.05))
```

![problem_fig](images/results.png)

---

### References:
#### David G. Luenberger, Yinyu Ye (2016). *Linear and Non Linear Programming*. (4th edition)