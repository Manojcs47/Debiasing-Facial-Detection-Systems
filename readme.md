# Mitigating Algorithmic Bias in Facial Detection Systems

## 📌 Project Overview
This project addresses critical algorithmic bias in computer vision systems by implementing a **Debiasing Variational Autoencoder (DB-VAE)**. Standard facial detection models often struggle to recognize faces from underrepresented demographics due to dataset imbalance. This project builds a semi-supervised generative model that automatically identifies and upsamples rare features (e.g., skin tone, gender) during training to create a fairer, more robust classifier.

## 🧠 The Problem: Algorithmic Bias
Deep learning models trained on imbalanced datasets (like CelebA, which is predominantly light-skinned) learn to correlate "face" features with the majority demographic. This leads to:
* High accuracy for Light-Skinned Females/Males.
* Significantly lower accuracy for Dark-Skinned Females/Males.
* Ethical risks in deployment (e.g., automated hiring, security).

## 💡 The Solution: DB-VAE
The **Debiasing Variational Autoencoder (DB-VAE)** modifies the standard training loop to mitigate bias in an unsupervised manner:

1.  **Latent Structure Learning:** The model (VAE) learns a low-dimensional latent representation of the human face.
2.  **Adaptive Resampling:** During training, the system estimates the frequency of features in the latent space.
3.  **Debiasing:** Rare data points (those in low-density regions of the latent space) are sampled with **higher probability**. This forces the model to "practice" more on underrepresented faces.

### Architecture
* **Encoder:** CNN with `Conv2D` and `BatchNormalization` layers. Outputs both class logits (Face/Not Face) and latent distribution parameters ($\mu, \sigma$).
* **Decoder:** Deconvolutional network (`Conv2DTranspose`) that reconstructs images from the latent variables.
* **Loss Function:** A hybrid loss combining **Binary Cross-Entropy** (classification) and **VAE Loss** (reconstruction + KL divergence).

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib
* **Experiment Tracking:** Comet ML
* **Datasets:** CelebA (Faces), ImageNet (Non-faces), Fitzpatrick Scale (Test set)

## 📊 Performance Visualizations
*Comparison of the Standard CNN (Blue) vs. the Debiased DB-VAE (Orange) on the test set.*

> *(Insert your bar chart image here, e.g., `image_71610b.png`)*

## 💻 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/debiasing-face-detection.git](https://github.com/your-username/debiasing-face-detection.git)
    cd debiasing-face-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib comet_ml mitdeeplearning
    ```

3.  **Run the Notebook:**
    Open `DB_VAE_Face_Detection.ipynb` in Jupyter Notebook or Google Colab/Kaggle and execute the cells.
    * *Note: A GPU backend is highly recommended for training.*

## 📜 Credits
This project was developed as part of the curriculum for **MIT 6.S191: Introduction to Deep Learning**.
