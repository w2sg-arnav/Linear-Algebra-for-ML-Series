# Linear Algebra Fundamentals Cheat Sheet

## Core Concepts

### 1. Vectors
- A vector is an ordered list of numbers
- Properties:
  * Direction
  * Magnitude (length)
  * Dimension (number of components)

#### Key Operations
```python
# Vector Addition
v + w = [v₁ + w₁, v₂ + w₂, ..., vₙ + wₙ]

# Scalar Multiplication
cv = [cv₁, cv₂, ..., cvₙ]

# Dot Product
v · w = v₁w₁ + v₂w₂ + ... + vₙwₙ
```

### 2. Linear Transformations
A transformation T is linear if:
1. T(v + w) = T(v) + T(w)    [Additivity]
2. T(cv) = cT(v)             [Scalar Multiplication]

### 3. Key Properties
- **Magnitude**: ||v|| = √(v₁² + v₂² + ... + vₙ²)
- **Unit Vector**: û = v/||v||
- **Angle**: cos(θ) = (v·w)/(||v||||w||)
- **Projection**: proj_u(v) = (v·û)û

## Quick NumPy Reference

```python
import numpy as np

# Create vectors
v = np.array([1, 2, 3])

# Vector operations
w = np.array([4, 5, 6])
v + w           # Addition
v * 2          # Scalar multiplication
np.dot(v, w)   # Dot product
np.linalg.norm(v)  # Magnitude

# Unit vector
u_hat = v / np.linalg.norm(v)

# Angle between vectors
cos_theta = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
theta = np.arccos(cos_theta)
```

## Common Applications in ML

1. **Feature Vectors**
   - Each data point is a vector
   - Components are features/attributes

2. **Linear Regression**
   - y = w·x + b
   - w is weight vector
   - x is input vector

3. **Neural Networks**
   - Input layer: feature vector
   - Weights: transformation matrices
   - Activations: non-linear functions

4. **Distance Metrics**
   - Euclidean: ||v - w||
   - Cosine similarity: (v·w)/(||v||||w||)

## Visual Intuition

1. **Vector Addition**
   - Parallelogram method
   - Tip-to-tail method

2. **Dot Product**
   - Projection * magnitude
   - cos(θ) * ||v|| * ||w||

3. **Linear Transformation**
   - Matrix multiplication
   - Preserves grid lines
   - Origin remains fixed

## Common Pitfalls

1. **Zero Vector**
   - Has no direction
   - Magnitude = 0
   - Projects to zero

2. **Linear Independence**
   - Check if one vector is a scalar multiple of another
   - For n vectors: check if any is a linear combination of others

3. **Numerical Issues**
   - Watch for division by zero
   - Check for numerical stability in calculations
   - Use np.allclose() for float comparisons

## Best Practices

1. **Code Implementation**
   - Use NumPy for efficiency
   - Validate input dimensions
   - Handle edge cases (zero vectors, parallel vectors)

2. **Debugging**
   - Visualize when possible
   - Check dimensions match
   - Test with simple cases first

3. **Optimization**
   - Vectorize operations
   - Use built-in NumPy functions
   - Avoid unnecessary copies
