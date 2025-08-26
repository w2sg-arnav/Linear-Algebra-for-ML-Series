# Vector Operations Cheat Sheet

## Basic Vector Operations

### Creation
```python
# Using NumPy
import numpy as np

# From a list
v = np.array([1, 2, 3])

# Zeros vector
zeros = np.zeros(3)

# Ones vector
ones = np.ones(3)

# Range vector
range_vec = np.arange(5)  # [0, 1, 2, 3, 4]
```

### Addition and Subtraction
```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Addition
sum_vec = v1 + v2  # [5, 7, 9]

# Subtraction
diff_vec = v2 - v1  # [3, 3, 3]
```

### Scalar Multiplication
```python
v = np.array([1, 2, 3])
scalar = 2

# Multiplication
scaled = scalar * v  # [2, 4, 6]
```

### Vector Magnitude (Length)
```python
v = np.array([3, 4])

# L2 norm (Euclidean length)
magnitude = np.linalg.norm(v)  # 5.0

# Other norms
l1_norm = np.linalg.norm(v, ord=1)     # Manhattan
l2_norm = np.linalg.norm(v, ord=2)     # Euclidean
inf_norm = np.linalg.norm(v, ord=np.inf)  # Maximum
```

### Dot Product
```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Method 1
dot_product = np.dot(v1, v2)  # 32

# Method 2
dot_product = v1 @ v2  # 32
```

### Vector Normalization
```python
v = np.array([3, 4])

# Unit vector (length 1)
unit_v = v / np.linalg.norm(v)  # [0.6, 0.8]
```

## Common Operations in ML

### Feature Scaling
```python
# Min-Max Scaling
def min_max_scale(v):
    return (v - v.min()) / (v.max() - v.min())

# Z-Score Normalization
def standardize(v):
    return (v - v.mean()) / v.std()
```

### Cosine Similarity
```python
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

### Distance Metrics
```python
# Euclidean Distance
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

# Manhattan Distance
def manhattan_distance(v1, v2):
    return np.linalg.norm(v1 - v2, ord=1)

# Cosine Distance
def cosine_distance(v1, v2):
    return 1 - cosine_similarity(v1, v2)
```

## Common Mistakes to Avoid

1. **Shape Mismatch**
   - Always check vector dimensions match before operations
   - Use `v.shape` to verify

2. **Broadcasting Issues**
   - Be careful with operations between vectors of different sizes
   - Understand NumPy broadcasting rules

3. **Numerical Stability**
   - Watch out for division by zero in normalization
   - Use `np.finfo(float).eps` for small numbers

4. **Memory Efficiency**
   - For large vectors, use appropriate dtypes
   - Consider sparse vectors when appropriate

## Best Practices

1. **Type Checking**
```python
def is_vector(v):
    return isinstance(v, np.ndarray) and len(v.shape) == 1
```

2. **Safe Division**
```python
def safe_normalize(v):
    norm = np.linalg.norm(v)
    if norm < np.finfo(float).eps:
        return v
    return v / norm
```

3. **Vectorized Operations**
```python
# Good (vectorized)
result = np.sum(v1 * v2)

# Bad (loop)
result = 0
for i in range(len(v1)):
    result += v1[i] * v2[i]
```

## Common Applications

1. **Text Processing**
   - Document vectors (TF-IDF)
   - Word embeddings

2. **Image Processing**
   - Pixel intensity vectors
   - Feature vectors

3. **Machine Learning**
   - Feature vectors
   - Weight vectors
   - Gradient vectors
