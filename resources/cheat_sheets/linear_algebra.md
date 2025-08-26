# Linear Algebra Cheat Sheet

## Vector Operations

### Basic Operations
- Addition: $\mathbf{v} + \mathbf{w} = [v_1 + w_1, v_2 + w_2, ..., v_n + w_n]$
- Scalar multiplication: $c\mathbf{v} = [cv_1, cv_2, ..., cv_n]$
- Dot product: $\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^n v_iw_i$
- Cross product (3D): $\mathbf{v} \times \mathbf{w} = [v_2w_3-v_3w_2, v_3w_1-v_1w_3, v_1w_2-v_2w_1]$

### Properties
- Commutative: $\mathbf{v} + \mathbf{w} = \mathbf{w} + \mathbf{v}$
- Associative: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- Distributive: $c(\mathbf{v} + \mathbf{w}) = c\mathbf{v} + c\mathbf{w}$

### Vector Norms
- L1 norm: $\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$
- L2 norm: $\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$
- L∞ norm: $\|\mathbf{v}\|_\infty = \max_i |v_i|$

## Matrix Operations

### Basic Operations
- Addition: $(A + B)_{ij} = A_{ij} + B_{ij}$
- Scalar multiplication: $(cA)_{ij} = cA_{ij}$
- Matrix multiplication: $(AB)_{ij} = \sum_k A_{ik}B_{kj}$
- Transpose: $(A^T)_{ij} = A_{ji}$

### Special Matrices
- Identity matrix: $I_n$
- Zero matrix: $0$
- Diagonal matrix: Non-zero elements only on diagonal
- Symmetric matrix: $A = A^T$
- Orthogonal matrix: $A^TA = AA^T = I$

### Matrix Properties
- $(AB)^T = B^TA^T$
- $(A^T)^T = A$
- $\det(AB) = \det(A)\det(B)$
- $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$

## Linear Transformations

### Properties
1. Preserves vector addition: $T(\mathbf{v} + \mathbf{w}) = T(\mathbf{v}) + T(\mathbf{w})$
2. Preserves scalar multiplication: $T(c\mathbf{v}) = cT(\mathbf{v})$

### Common Transformations
- Rotation matrix (2D): $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$
- Scaling matrix: $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$
- Shear matrix: $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$

## Matrix Decompositions

### Eigendecomposition
$A = PDP^{-1}$ where:
- $D$ is diagonal matrix of eigenvalues
- $P$ columns are eigenvectors

### SVD (Singular Value Decomposition)
$A = U\Sigma V^T$ where:
- $U$ and $V$ are orthogonal
- $\Sigma$ contains singular values

### QR Decomposition
$A = QR$ where:
- $Q$ is orthogonal
- $R$ is upper triangular

## Common Applications in ML

### Principal Component Analysis (PCA)
1. Center the data
2. Compute covariance matrix
3. Find eigenvectors and eigenvalues
4. Project data onto principal components

### Linear Regression
- Normal equation: $\hat{\beta} = (X^TX)^{-1}X^Ty$
- Ridge regression: $\hat{\beta} = (X^TX + \lambda I)^{-1}X^Ty$

### Neural Networks
- Weight matrices for linear layers
- Backpropagation using chain rule
- Gradient descent optimization
