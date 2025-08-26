# Deep Learning Mathematics Cheat Sheet

## Tensor Operations

### Basic Operations
```python
# Element-wise operations
C = A + B
C = A * B
C = np.maximum(A, 0)  # ReLU

# Reduction
mean = np.mean(A, axis=0)
sum = np.sum(A, axis=1)

# Matrix multiplication
C = np.matmul(A, B)
C = A @ B

# Transpose
C = np.transpose(A)
```

### Broadcasting Rules
1. Arrays are compatible when:
   - Same shape, or
   - One array has shape 1 in the broadcasted dimension

Example:
```python
A = np.random.randn(64, 32, 1)   # batch_size x seq_length x 1
B = np.random.randn(1, 1, 100)   # 1 x 1 x embedding_dim
C = A * B  # Shape: (64, 32, 100)
```

## Neural Network Operations

### Linear Layer
```python
Y = XW + b
# X: (batch_size, input_dim)
# W: (input_dim, output_dim)
# b: (output_dim,)
# Y: (batch_size, output_dim)
```

### Convolutional Layer
```python
# Conv2D
Y = conv2d(X, W)
# X: (batch_size, in_channels, height, width)
# W: (out_channels, in_channels, kernel_h, kernel_w)
# Y: (batch_size, out_channels, out_height, out_width)
```

### Batch Normalization
```python
# During training
mean = np.mean(X, axis=0)
var = np.var(X, axis=0)
X_norm = (X - mean) / np.sqrt(var + eps)
Y = gamma * X_norm + beta

# During inference
Y = gamma * (X - running_mean) / np.sqrt(running_var + eps) + beta
```

### Attention Mechanism
```python
# Scaled Dot-Product Attention
scores = (Q @ K.T) / np.sqrt(d_k)
attention_weights = softmax(scores)
output = attention_weights @ V
```

## Activation Functions

### Common Functions
1. ReLU: `f(x) = max(0, x)`
2. Sigmoid: `f(x) = 1 / (1 + exp(-x))`
3. Tanh: `f(x) = tanh(x)`
4. LeakyReLU: `f(x) = max(αx, x)`
5. Softmax: `f(x_i) = exp(x_i) / Σ exp(x_j)`

### Derivatives
1. ReLU: `f'(x) = 1 if x > 0 else 0`
2. Sigmoid: `f'(x) = f(x)(1 - f(x))`
3. Tanh: `f'(x) = 1 - tanh²(x)`
4. LeakyReLU: `f'(x) = 1 if x > 0 else α`

## Loss Functions

### Cross Entropy Loss
```python
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))
```

### MSE Loss
```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
```

## Optimization

### Gradient Descent
```python
# Basic gradient descent
w = w - learning_rate * gradient

# Momentum
v = beta * v - learning_rate * gradient
w = w + v

# Adam
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
w = w - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
```

## Regularization

### L1 Regularization
```python
loss = original_loss + lambda_l1 * np.sum(np.abs(weights))
```

### L2 Regularization
```python
loss = original_loss + lambda_l2 * np.sum(weights**2)
```

### Dropout
```python
# Training
mask = np.random.binomial(1, p, size=X.shape) / p
Y = X * mask

# Inference
Y = X  # No scaling needed due to training phase scaling
```

## Initialization

### Xavier/Glorot
```python
W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
```

### He/Kaiming
```python
W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
```

## Common Tensor Shapes

### CNN
- Input: `(batch_size, channels, height, width)`
- Convolution weights: `(out_channels, in_channels, kernel_h, kernel_w)`
- Feature maps: `(batch_size, channels, height, width)`

### RNN
- Input: `(batch_size, seq_length, input_dim)`
- Hidden state: `(batch_size, hidden_dim)`
- LSTM cell state: `(batch_size, cell_dim)`

### Transformer
- Input embeddings: `(batch_size, seq_length, embed_dim)`
- Attention scores: `(batch_size, num_heads, seq_length, seq_length)`
- Multi-head output: `(batch_size, seq_length, num_heads * head_dim)`

## Useful NumPy Operations

```python
# Reshaping
X = X.reshape(batch_size, -1)
X = np.expand_dims(X, axis=1)
X = np.squeeze(X, axis=1)

# Concatenation
C = np.concatenate([A, B], axis=1)

# Splitting
X1, X2 = np.split(X, 2, axis=1)

# Advanced indexing
mask = X > 0
X[mask] = 0

# Einstein summation
C = np.einsum('bij,bjk->bik', A, B)  # Batch matrix multiply
```

## Performance Tips

1. Use vectorized operations over loops
2. Prefer in-place operations when possible
3. Use appropriate data types (float32 vs float64)
4. Leverage broadcasting for memory efficiency
5. Profile memory usage with large tensors
6. Use appropriate device placement (CPU/GPU)
7. Batch operations for parallel processing
