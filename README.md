[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jYmqdKSO)
# 📌 Autoencoder with MNIST Dataset

This project demonstrates how to build, train, and use an **autoencoder** for the **MNIST handwritten digit dataset**. It includes visualization of digits, dimensionality reduction, and an image retrieval based on **cosine similarity**.

---

## 🚀 Project Overview

1. **Data Loading**:

   * The MNIST dataset (`mnist_784`) is fetched using `fetch_openml`.
   * It contains 70,000 grayscale images of handwritten digits (0–9).
   * Each image is represented as a **28×28** pixel grid (flattened to 784 features).

2. **Autoencoder Architecture**:

   * **Encoder**: Compresses 784-dimensional input into an n-dimensional latent representation.
   * **Decoder**: Reconstructs the original 784-dimensional image from the latent space.
   * **Loss Function**
   * **Optimizer**

3. **Training**:

   * The autoencoder is trained to reconstruct the input images.

4. **Dimensionality Reduction & Visualization**:

   * Encoded representations of test images are further reduced to 2D.

5. **Image Retrieval**:

   * A sample image is encoded into the latent space using the encoder only.
   * Cosine similarity is used to compare it against the training set.
   * The top-5 most similar images are retrieved and displayed.
