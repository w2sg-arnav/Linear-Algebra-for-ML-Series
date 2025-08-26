# Practice Problems: Linear Algebra for Machine Learning

## Module 1: Foundations

### Problem Set 1: Vectors and Vector Operations

1. Given vectors $\mathbf{v} = [2, 3]$ and $\mathbf{w} = [1, -1]$:
   a) Calculate $\mathbf{v} + \mathbf{w}$
   b) Calculate $2\mathbf{v} - 3\mathbf{w}$
   c) Find the dot product $\mathbf{v} \cdot \mathbf{w}$
   d) Calculate the angle between $\mathbf{v}$ and $\mathbf{w}$

2. For vectors $\mathbf{a} = [1, 2, 3]$ and $\mathbf{b} = [0, 1, -1]$:
   a) Find their cross product
   b) Show that the cross product is perpendicular to both vectors
   c) Calculate the area of the parallelogram formed by these vectors

### Problem Set 2: Linear Combinations and Spans

1. Determine if the vector $[2, 1]$ is in the span of vectors $[1, 0]$ and $[0, 1]$

2. Find a linear combination of $[1, 1]$ and $[1, -1]$ that equals $[3, 1]$

3. For vectors $\mathbf{v}_1 = [1, 0, 1]$, $\mathbf{v}_2 = [0, 1, 1]$, and $\mathbf{v}_3 = [1, 1, 2]$:
   a) Are these vectors linearly independent?
   b) Find their span
   c) Can they span $\mathbb{R}^3$?

### Problem Set 3: Basis and Coordinates

1. Given the basis vectors $\mathbf{b}_1 = [1, 1]$ and $\mathbf{b}_2 = [1, -1]$:
   a) Express $[2, 0]$ in this basis
   b) Express $[1, 3]$ in this basis
   c) Is this a valid basis? Why?

2. Find the coordinates of $[3, 2, 1]$ in the basis:
   $\mathbf{b}_1 = [1, 0, 0]$, $\mathbf{b}_2 = [1, 1, 0]$, $\mathbf{b}_3 = [1, 1, 1]$

### Problem Set 4: Norms and Distances

1. Calculate the following norms for $\mathbf{v} = [1, -2, 2]$:
   a) L1 norm
   b) L2 norm
   c) L∞ norm

2. Find the distance between points $\mathbf{p} = [1, 2, 3]$ and $\mathbf{q} = [4, 0, -1]$ using:
   a) Euclidean distance
   b) Manhattan distance
   c) Cosine similarity

### Problem Set 5: Matrices and Linear Transformations

1. Given matrix $A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$:
   a) Find its eigenvalues
   b) Find its eigenvectors
   c) Verify that $A\mathbf{v} = \lambda\mathbf{v}$ for each eigenpair

2. For the transformation matrix $T = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$:
   a) What type of transformation is this?
   b) Apply it to vector $[1, 0]$
   c) Apply it twice to $[1, 0]$

## Module 2: Matrices as Transformations

### Problem Set 6: Matrix Operations

1. Given matrices:
   $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and
   $B = \begin{bmatrix} 0 & 1 \\ -1 & 2 \end{bmatrix}$
   Calculate:
   a) $A + B$
   b) $AB$
   c) $BA$
   d) $A^T B$

2. Find the inverse of $\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$

### Problem Set 7: Matrix Decompositions

1. Given matrix $A = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$:
   a) Find its SVD
   b) Find its eigendecomposition
   c) Compare the results

2. For matrix $B = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}$:
   a) Find its QR decomposition
   b) Verify that $Q$ is orthogonal
   c) Use the decomposition to solve $Bx = [1, 2, 3]^T$

### Problem Set 8: Applications

1. Given a dataset with features $X$ and labels $y$:
   a) Implement linear regression using normal equations
   b) Implement ridge regression with regularization parameter $\lambda = 0.1$
   c) Compare the results

2. For a correlation matrix $C$:
   a) Find its eigenvalues and eigenvectors
   b) Use PCA to reduce dimensionality
   c) Calculate the explained variance ratio

## Solutions

Detailed solutions to these problems can be found in the solutions directory. Try to solve them yourself first!
