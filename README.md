# Autoencoder with MNIST Dataset

This project builds and trains an autoencoder on the MNIST handwritten digit dataset. It covers dimensionality reduction and image retrieval using cosine similarity.

## Project Structure

The project is implemented in a single Jupyter notebook: `autoencoder_mnist.ipynb`

## Requirements

```
tensorflow
scikit-learn
matplotlib
numpy
```

Install with:

```
pip install tensorflow scikit-learn matplotlib numpy
```

## Steps

**1. Data Loading**
The MNIST dataset is loaded using `fetch_openml`. It contains 70,000 grayscale images of handwritten digits (0-9), each represented as a flattened 784-dimensional vector. Pixel values are normalized to the range [0, 1].

**2. Autoencoder Architecture**
The model consists of an encoder and a decoder. The encoder compresses the input from 784 dimensions down to a 32-dimensional latent vector through two hidden layers (256, 128). The decoder mirrors this structure and reconstructs the original image from the latent vector. Binary crossentropy is used as the loss function with the Adam optimizer.

**3. Training**
The autoencoder is trained with the input as both the input and target (`fit(X_train, X_train)`). Training runs for 20 epochs with a batch size of 256.

**4. Dimensionality Reduction**
The encoder is used to produce latent representations of the test set. t-SNE is then applied to reduce these 32-dimensional vectors to 2D for visualization. Each digit forms a distinct cluster, showing that the latent space captures meaningful structure.

**5. Image Retrieval**
A query image is encoded into the latent space. Cosine similarity is computed between the query vector and all training set encodings. The top 5 most similar images are returned and displayed.

## Key Design Decisions

- Latent dimension of 32 balances compression and reconstruction quality
- Sigmoid activation in the final decoder layer keeps outputs in [0, 1]
- Cosine similarity is used over Euclidean distance because it measures direction rather than magnitude, which is more meaningful in latent spaces
- t-SNE is preferred over PCA for visualization because it better captures non-linear cluster structure

## Experimentation

Try changing `LATENT_DIM` to 8, 64, or 128 to observe the effect on reconstruction quality, cluster separation, and retrieval accuracy.