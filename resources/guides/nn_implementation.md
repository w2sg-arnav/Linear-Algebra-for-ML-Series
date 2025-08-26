# Neural Network Implementation Guide

This guide provides step-by-step instructions for implementing neural networks from scratch, focusing on the linear algebra and mathematical operations involved.

## Table of Contents
1. [Basic Building Blocks](#basic-building-blocks)
2. [Layer Implementation](#layer-implementation)
3. [Activation Functions](#activation-functions)
4. [Loss Functions](#loss-functions)
5. [Optimization](#optimization)
6. [Advanced Architectures](#advanced-architectures)

## Basic Building Blocks

### Vector and Matrix Operations
```python
import numpy as np

def dot_product(v1, v2):
    """Compute dot product between two vectors"""
    return np.sum(v1 * v2)

def matrix_multiply(A, B):
    """Implement matrix multiplication from scratch"""
    assert A.shape[1] == B.shape[0], "Matrix dimensions must match"
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i,j] = dot_product(A[i,:], B[:,j])
    return result
```

### Tensor Operations
```python
def tensor_dot(A, B, axes):
    """Implement tensor dot product"""
    return np.tensordot(A, B, axes=axes)

def batch_matmul(A, B):
    """Implement batch matrix multiplication"""
    return np.matmul(A, B)
```

## Layer Implementation

### Dense Layer
```python
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # Xavier initialization
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/(input_dim + output_dim))
        self.bias = np.zeros(output_dim)
        
    def forward(self, X):
        """Forward pass"""
        self.input = X
        return np.dot(X, self.weights) + self.bias
    
    def backward(self, grad_output):
        """Backward pass"""
        grad_input = np.dot(grad_output, self.weights.T)
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_input
```

### Convolutional Layer
```python
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels)
        
    def im2col(self, X):
        """Convert input image to column matrix"""
        # Implementation details...
        pass
        
    def col2im(self, cols, output_shape):
        """Convert column matrix back to image"""
        # Implementation details...
        pass
        
    def forward(self, X):
        self.input = X
        batch_size, _, height, width = X.shape
        
        # Calculate output dimensions
        out_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Im2col transformation
        X_col = self.im2col(X)
        W_col = self.weights.reshape(self.out_channels, -1)
        
        # Convolution as matrix multiplication
        out = np.dot(W_col, X_col) + self.bias.reshape(-1, 1)
        
        # Reshape output
        out = out.reshape(self.out_channels, out_height, out_width, batch_size)
        out = out.transpose(3, 0, 1, 2)
        
        return out
```

## Activation Functions

### Implementation
```python
class ReLU:
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class Sigmoid:
    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class Softmax:
    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        return self.output * (grad_output - np.sum(grad_output * self.output, axis=1, keepdims=True))
```

## Loss Functions

### Implementation
```python
class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        log_prob = -np.log(y_pred[range(len(y_true)), y_true])
        return np.mean(log_prob)
    
    def backward(self):
        m = len(self.y_true)
        grad = self.y_pred.copy()
        grad[range(m), self.y_true] -= 1
        return grad / m

class MSELoss:
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        return np.mean(self.diff ** 2)
    
    def backward(self):
        return 2 * self.diff / len(self.diff)
```

## Optimization

### Implementation
```python
class SGD:
    def __init__(self, parameters, learning_rate=0.01, momentum=0):
        self.parameters = parameters
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {k: np.zeros_like(v) for k, v in parameters.items()}
    
    def step(self, gradients):
        for key in self.parameters:
            self.velocity[key] = (self.momentum * self.velocity[key] - 
                                self.lr * gradients[key])
            self.parameters[key] += self.velocity[key]

class Adam:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in parameters.items()}
        self.v = {k: np.zeros_like(v) for k, v in parameters.items()}
        self.t = 0
    
    def step(self, gradients):
        self.t += 1
        for key in self.parameters:
            self.m[key] = (self.beta1 * self.m[key] + 
                          (1 - self.beta1) * gradients[key])
            self.v[key] = (self.beta2 * self.v[key] + 
                          (1 - self.beta2) * gradients[key]**2)
            
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            self.parameters[key] -= (self.lr * m_hat / 
                                   (np.sqrt(v_hat) + self.eps))
```

## Advanced Architectures

### Attention Mechanism
```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
    
    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # Linear projections
        Q = np.dot(Q, self.W_q)
        K = np.dot(K, self.W_k)
        V = np.dot(V, self.W_v)
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = np.softmax(scores, axis=-1)
        
        # Apply attention weights
        context = np.matmul(weights, V)
        
        # Concatenate heads and apply final linear projection
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = np.dot(context, self.W_o)
        
        return output, weights
```

### Residual Connection
```python
class ResidualBlock:
    def __init__(self, layer):
        self.layer = layer
    
    def forward(self, x):
        self.input = x
        return x + self.layer.forward(x)
    
    def backward(self, grad_output):
        layer_grad = self.layer.backward(grad_output)
        return grad_output + layer_grad
```

### Layer Normalization
```python
class LayerNorm:
    def __init__(self, features, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
    
    def forward(self, x):
        self.input = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.normalized + self.beta
    
    def backward(self, grad_output):
        # Implementation of backward pass...
        pass
```

## Tips and Best Practices

1. **Initialization**
   - Use appropriate weight initialization (Xavier/He)
   - Initialize biases to zero or small constants

2. **Numerical Stability**
   - Use stable implementations of softmax and log-softmax
   - Add small epsilon to denominators
   - Normalize inputs and gradients

3. **Performance Optimization**
   - Vectorize operations when possible
   - Use appropriate batch sizes
   - Implement gradient clipping
   - Monitor gradient norms

4. **Debugging Tips**
   - Check gradient computations with numerical gradients
   - Monitor loss and gradient values
   - Implement gradient checks
   - Use proper testing procedures

5. **Memory Management**
   - Clear unnecessary intermediate results
   - Use appropriate data types
   - Implement memory-efficient backpropagation
