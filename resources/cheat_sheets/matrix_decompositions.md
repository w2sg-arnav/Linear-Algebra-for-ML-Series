# Matrix Decompositions Cheat Sheet

## Overview of Matrix Decompositions

Different matrix decompositions reveal different aspects of a matrix and are useful for different applications in machine learning.

## 1. Eigendecomposition

### Form
$A = PDP^{-1}$

### Components
- $P$: Matrix of eigenvectors
- $D$: Diagonal matrix of eigenvalues
- $P^{-1}$: Inverse of eigenvector matrix

### When to Use
- Understanding linear transformations
- Diagonalizing matrices
- PCA and dimensionality reduction
- Analyzing dynamical systems

### Limitations
- Only for square matrices
- Not all matrices have real eigendecomposition
- Requires linearly independent eigenvectors

## 2. Singular Value Decomposition (SVD)

### Form
$A = U\Sigma V^T$

### Components
- $U$: Left singular vectors (orthogonal)
- $\Sigma$: Diagonal matrix of singular values
- $V^T$: Right singular vectors (orthogonal)

### When to Use
- Data compression
- Dimensionality reduction
- Matrix approximation
- Collaborative filtering
- Image processing

### Advantages
- Always exists for any matrix
- Stable numerical computation
- Reveals rank and null spaces
- Works for non-square matrices

## 3. QR Decomposition

### Form
$A = QR$

### Components
- $Q$: Orthogonal matrix
- $R$: Upper triangular matrix

### When to Use
- Solving linear systems
- Least squares problems
- Computing eigenvectors
- Gram-Schmidt process

### Advantages
- Computationally efficient
- Useful for iterative methods
- Good numerical stability

## 4. LU Decomposition

### Form
$A = LU$ or $A = PLU$

### Components
- $L$: Lower triangular matrix
- $U$: Upper triangular matrix
- $P$: Permutation matrix (optional)

### When to Use
- Solving linear systems
- Matrix inversion
- Computing determinants

### Advantages
- Efficient for solving systems
- Easy to implement
- Good for repeated solutions

## 5. Cholesky Decomposition

### Form
$A = LL^T$

### Components
- $L$: Lower triangular matrix
- $L^T$: Transpose of L

### When to Use
- Positive definite matrices
- Solving normal equations
- Monte Carlo simulations
- Matrix inversion

### Advantages
- Most efficient for symmetric positive definite matrices
- Numerically stable
- Unique decomposition

## Applications in Machine Learning

### 1. Principal Component Analysis
```python
# Using SVD
U, S, VT = np.linalg.svd(X)
principal_components = VT[:n_components]
```

### 2. Linear Regression
```python
# Using QR decomposition
Q, R = np.linalg.qr(X)
beta = np.linalg.solve(R, Q.T @ y)
```

### 3. Matrix Factorization for Recommendations
```python
# Using SVD
U, S, VT = np.linalg.svd(ratings_matrix)
reduced_ratings = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
```

### 4. Solving Linear Systems
```python
# Using LU decomposition
L, U = scipy.linalg.lu(A)
y = scipy.linalg.solve_triangular(L, b, lower=True)
x = scipy.linalg.solve_triangular(U, y)
```

## Comparison of Decompositions

### Computational Complexity
1. SVD: O(mn²)
2. QR: O(mn²)
3. LU: O(n³)
4. Cholesky: O(n³/3)

### Memory Requirements
1. SVD: O(mn)
2. QR: O(mn)
3. LU: O(n²)
4. Cholesky: O(n²/2)

### Numerical Stability
1. SVD: Most stable
2. QR: Very stable
3. LU: Moderately stable
4. Cholesky: Very stable for SPD matrices

## Best Practices

1. Choose decomposition based on:
   - Matrix properties
   - Computational requirements
   - Stability needs
   - Application requirements

2. Consider preprocessing:
   - Scaling
   - Centering
   - Condition number

3. Use specialized libraries:
   - NumPy
   - SciPy
   - LAPACK
   - BLAS
