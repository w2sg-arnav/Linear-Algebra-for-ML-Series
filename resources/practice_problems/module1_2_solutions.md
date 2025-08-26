# Solutions: Linear Algebra for Machine Learning

## Module 1: Foundations

### Problem Set 1: Vectors and Vector Operations

1. Given vectors $\mathbf{v} = [2, 3]$ and $\mathbf{w} = [1, -1]$:
   
   a) $\mathbf{v} + \mathbf{w} = [2+1, 3+(-1)] = [3, 2]$
   
   b) $2\mathbf{v} - 3\mathbf{w} = [2(2)-3(1), 2(3)-3(-1)] = [4-3, 6+3] = [1, 9]$
   
   c) $\mathbf{v} \cdot \mathbf{w} = 2(1) + 3(-1) = 2 - 3 = -1$
   
   d) $\cos \theta = \frac{\mathbf{v} \cdot \mathbf{w}}{|\mathbf{v}||\mathbf{w}|}$
      
      $|\mathbf{v}| = \sqrt{4 + 9} = \sqrt{13}$
      
      $|\mathbf{w}| = \sqrt{1 + 1} = \sqrt{2}$
      
      $\theta = \arccos(\frac{-1}{\sqrt{26}}) \approx 2.034$ radians or 116.6 degrees

2. For vectors $\mathbf{a} = [1, 2, 3]$ and $\mathbf{b} = [0, 1, -1]$:
   
   a) $\mathbf{a} \times \mathbf{b} = [2(-1)-3(1), 3(0)-1(-1), 1(1)-2(0)]$
      $= [-5, 3, 1]$
   
   b) Check perpendicularity:
      $(\mathbf{a} \times \mathbf{b}) \cdot \mathbf{a} = -5(1) + 3(2) + 1(3) = 0$
      $(\mathbf{a} \times \mathbf{b}) \cdot \mathbf{b} = -5(0) + 3(1) + 1(-1) = 2$
   
   c) Area = $|\mathbf{a} \times \mathbf{b}| = \sqrt{25 + 9 + 1} = \sqrt{35}$

### Problem Set 2: Linear Combinations and Spans

1. $[2, 1] = c_1[1, 0] + c_2[0, 1]$
   
   This gives system of equations:
   $c_1 = 2$
   $c_2 = 1$
   
   Therefore $[2, 1]$ is in the span.

2. Need to find $c_1$ and $c_2$ such that:
   $c_1[1, 1] + c_2[1, -1] = [3, 1]$
   
   This gives system:
   $c_1 + c_2 = 3$
   $c_1 - c_2 = 1$
   
   Solving: $c_1 = 2, c_2 = 1$

3. For vectors $\mathbf{v}_1 = [1, 0, 1]$, $\mathbf{v}_2 = [0, 1, 1]$, $\mathbf{v}_3 = [1, 1, 2]$:
   
   a) Check if $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + c_3\mathbf{v}_3 = \mathbf{0}$
      $[c_1 + c_3, c_2 + c_3, c_1 + c_2 + 2c_3] = [0, 0, 0]$
      Only solution is $c_1 = c_2 = c_3 = 0$, so they are linearly independent.
   
   b) Their span is $\mathbb{R}^3$ since they are linearly independent and there are 3 of them.
   
   c) Yes, they span $\mathbb{R}^3$ as shown in (b).

### Problem Set 3: Basis and Coordinates

1. Given basis vectors $\mathbf{b}_1 = [1, 1]$ and $\mathbf{b}_2 = [1, -1]$:
   
   a) $[2, 0] = c_1[1, 1] + c_2[1, -1]$
      Solving: $c_1 = c_2 = 1$
   
   b) $[1, 3] = c_1[1, 1] + c_2[1, -1]$
      Solving: $c_1 = 2, c_2 = -1$
   
   c) Yes, this is a valid basis because:
      - Vectors are linearly independent
      - They span $\mathbb{R}^2$

2. Find coordinates $[c_1, c_2, c_3]$ such that:
   $c_1[1, 0, 0] + c_2[1, 1, 0] + c_3[1, 1, 1] = [3, 2, 1]$
   
   Solving system:
   $c_1 + c_2 + c_3 = 3$
   $c_2 + c_3 = 2$
   $c_3 = 1$
   
   Therefore: $c_3 = 1, c_2 = 1, c_1 = 1$

### Problem Set 4: Norms and Distances

1. For $\mathbf{v} = [1, -2, 2]$:
   
   a) L1 norm: $|1| + |-2| + |2| = 5$
   
   b) L2 norm: $\sqrt{1^2 + (-2)^2 + 2^2} = \sqrt{9} = 3$
   
   c) L∞ norm: $\max(|1|, |-2|, |2|) = 2$

2. For $\mathbf{p} = [1, 2, 3]$ and $\mathbf{q} = [4, 0, -1]$:
   
   a) Euclidean: $\sqrt{(4-1)^2 + (0-2)^2 + (-1-3)^2} = \sqrt{9 + 4 + 16} = \sqrt{29}$
   
   b) Manhattan: $|4-1| + |0-2| + |-1-3| = 3 + 2 + 4 = 9$
   
   c) Cosine similarity: $\frac{4+0-3}{\sqrt{14}\sqrt{17}} \approx 0.0345$

### Problem Set 5: Matrices and Linear Transformations

1. For $A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$:
   
   a) Characteristic equation: $(2-\lambda)(3-\lambda) - 1 = 0$
      $\lambda^2 - 5\lambda + 5 = 0$
      $\lambda = \frac{5 \pm \sqrt{25-20}}{2} = \frac{5 \pm \sqrt{5}}{2}$
   
   b) For $\lambda_1 = \frac{5 + \sqrt{5}}{2}$:
      $[2-\lambda_1, 1; 1, 3-\lambda_1][x, y]^T = [0, 0]^T$
      $\mathbf{v}_1 = [1, \frac{\sqrt{5}-1}{2}]^T$
      
      For $\lambda_2 = \frac{5 - \sqrt{5}}{2}$:
      $\mathbf{v}_2 = [1, -\frac{\sqrt{5}+1}{2}]^T$
   
   c) Verify: $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$

2. For $T = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$:
   
   a) This is a 90-degree counterclockwise rotation matrix
   
   b) $T[1, 0]^T = [0, 1]^T$
   
   c) $T^2[1, 0]^T = T[0, 1]^T = [-1, 0]^T$

## Module 2: Matrices as Transformations

[Solutions for Module 2 would continue in the same detailed format...]
