# Lecture 2: The Dot Product - The Heart of Machine Learning

This lecture dives into the single most important operation in linear algebra for machine learning: the dot product. We move from representing data to understanding relationships *between* data.

[▶️ Watch the full one-hour lecture on YouTube](your_youtube_video_link_here)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/w2sg-arnav/Linear-Algebra-for-ML-Series/blob/main/lectures/02-Dot-Product-and-Similarity/lecture_02_notebook.ipynb)

## 🎯 Learning Objectives

By the end of this lecture, you will have a deep, multi-faceted understanding of the dot product:

1.  **Geometric Intuition:** Understand the dot product as a measure of projection and alignment between vectors.
2.  **Algebraic Fluency:** Confidently compute the dot product and use its properties.
3.  **Bridge Theory and Practice:** Understand the formal connection between the geometric and algebraic definitions.
4.  **Application 1: Similarity:** Implement cosine similarity from scratch to see how search engines and recommender systems rank items.
5.  **Application 2: Neural Networks:** See how the dot product is the fundamental computation within a single neuron, the building block of deep learning.

## 📚 Key Topics Covered

### 1. The Three Perspectives of the Dot Product
- **Geometric View:** `v · w = ||v|| ||w|| cos(θ)`. We explore what this means in terms of vector alignment, projection, and orthogonality.
- **Algebraic View:** `v · w = Σ(vi * wi)`. We cover the component-wise multiplication and summation, the computational workhorse.
- **Machine Learning View:** We see it as a tool for measuring similarity and as the core of a neuron's weighted sum.

### 2. Cosine Similarity
- **Derivation:** How to derive the cosine similarity formula from the geometric definition of the dot product.
- **Implementation:** We build a document similarity checker from scratch, demonstrating how to find relevant documents in a corpus.

### 3. The Neuron
- **Weighted Sum:** We frame the computation inside a single artificial neuron as a dot product between an input vector and a weight vector.
- **Learning as Alignment:** We build the intuition that a neuron "learns" by adjusting its weight vector to be more aligned with input vectors of a certain class.

### 4. "Math to Code" Mappings
- We explicitly connect the mathematical formulas for the dot product and cosine similarity to their efficient implementations in NumPy.

## 💻 Hands-On Practice

This lecture comes with two essential notebooks:
1.  `lecture_02_notebook.ipynb`: Contains all the theory, interactive visualizations, and code from the lecture, plus a new set of challenging exercises.
2.  `lecture_02_solutions.ipynb`: Provides the complete, commented solutions to all exercises.

To master the concepts in this lecture, you should:
1.  Engage with the interactive plots to build a strong geometric intuition.
2.  Follow the from-scratch implementations of cosine similarity and the neuron.
3.  Attempt all exercises, especially the ones that challenge you to apply the dot product to new scenarios.
