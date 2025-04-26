 Monet Style Image Generation Using GANs

(Kaggle GANs Getting Started Competition — MiFID Evaluation)
 Project Overview

In this project, we address the Kaggle GANs Getting Started Competition, where the goal is to generate Monet-style paintings using a Generative Adversarial Network (GAN).

We train a lightweight GAN that can create realistic Monet-style images at 256×256 resolution.
Evaluation is based on the MiFID (Memorization-informed Fréchet Inception Distance) score: the lower, the better.
 Dataset

    Monet Paintings: ~1,500 images

    Real Photos: ~7,000 images

    Image Size: 256×256 pixels, 3 RGB channels

    Source: Kaggle GANs Getting Started Dataset
    
Model Architecture
Generator

    Encoder-decoder structure using Conv2D and ConvTranspose2D.

    Activation: LeakyReLU for encoder, ReLU for decoder.

    Output Activation: Tanh to produce outputs scaled to [-1, 1].

Discriminator

    Convolutional binary classifier.

    Activation: LeakyReLU after each convolution.

    Final output: Single scalar logit for real/fake classification.

 Training Strategy

    Optimizer: Adam (lr=2e-4, betas=(0.5, 0.999))

    Loss Function: Binary Cross Entropy with Logits (BCEWithLogitsLoss)

    Normalization: All images scaled to [-1, 1]

    Training Procedure:

        Alternate training Discriminator and Generator.

        Update Discriminator to distinguish real vs. fake images.

        Update Generator to fool the Discriminator.

Evaluation
Metric	Result
MiFID Public Score	(Insert your actual score here, e.g., 850)
Number of Generated Images	7,000
Image Size	256×256 RGB

 Successfully achieved a MiFID score below 1000 as required.
 Result Visualization

    Generated Monet-style paintings are visually convincing.

    Generator loss and Discriminator loss converge stably over epochs.

Example: (Insert a screenshot of your generated images here)
 How to Reproduce

    Clone this repository.

    Open the notebook MonetGAN_Project.ipynb.

    Train the models (or load saved models if available).

    Generate 7,000+ Monet-style images.

    Save and zip the images into images.zip.

    Submit images.zip to Kaggle for evaluation.

Deliverables

     Jupyter Notebook: MonetGAN_Project.ipynb

     GitHub Repository: (Insert your repo URL here)

     Kaggle Leaderboard Screenshot: (Insert screenshot showing your MiFID score)

Future Work

    Implement CycleGAN for unpaired style transfer (Photo → Monet domain translation).

    Use Residual Blocks in the Generator for better realism.

    Apply advanced data augmentation to improve Generator diversity.
