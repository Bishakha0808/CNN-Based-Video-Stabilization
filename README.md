# DeepStab-GAN: Video Stabilization with 3D CNNs & Adversarial Learning

Your goal is to build an AI model that takes in shaky aerial video (like what you’d get from a handheld drone or camera) and outputs a smooth, stabilized version — similar to what a physical gimbal does. But here’s the twist — you’ll be doing it entirely through deep learning, without using any hardware stabilizer.



  

A deep learning framework for video stabilization that combines **3D Convolutional Neural Networks (3D CNNs)** for spatiotemporal feature extraction with **Generative Adversarial Networks (GANs)** for realistic frame reconstruction.

This project uses a hybrid loss function (Perceptual + Adversarial + Smoothness) to transform shaky, unstable footage into smooth, stable video clips.

-----

##  Key Features

  * **Generator (StabGenerator):** A 3D CNN encoder-decoder architecture that predicts affine transformation matrices ($\theta$) for every frame in a clip.
  * **Differentiable Warping:** Uses PyTorch's `grid_sample` and `affine_grid` for end-to-end differentiable frame warping.
  * **Dual Input Stream:** Fuses raw RGB frames with **Optical Flow** (calculated via Farneback method) to better understand motion dynamics.
  * **Perceptual Loss:** Utilizes a pre-trained **VGG19** network to ensure the stabilized frames preserve high-level structural details, not just pixel accuracy.
  * **Smart Evaluation:** Includes a custom `SSIM (Cropped)` metric that evaluates structural similarity while ignoring black borders caused by stabilization warping.

-----
## Dataset 

```
https://www.kaggle.com/datasets/bishakhamondal/deepstab
```

-----

## ⚙️ Configuration

You can tweak the training hyperparameters in the `CONFIG` dictionary located in the main script:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `EPOCHS` | 200 |
| `BATCH_SIZE` | 2 | 
| `CLIP_LEN` | 32 | 
| `RESIZE` | (224, 224) | 
| `LR_G` | 1e-4 | 
| `KAGGLE_ROOT` | `...` |

-----

##  Model Architecture

### 1\. The Generator (StabGenerator)

  * **Input:** 5-Channel Tensor (3 RGB channels + 2 Optical Flow channels).
  * **Encoder:** Sequence of `Conv3d` layers to extract spatiotemporal features.
  * **Regressor:** A fully connected head that outputs a $2 \times 3$ affine matrix for each of the 32 frames.
  * **Warping:** The predicted matrices are applied to the original high-res RGB frames to produce the stabilized output.

### 2\. The Discriminator (StabDiscriminator)

  * A 3D CNN binary classifier that attempts to distinguish between the ground truth (stable videos) and the model's output (stabilized videos).

### 3\. Loss Function

The model optimizes a weighted sum of four specific losses:
$$L_{total} = \lambda_{gan}L_{BCE} + \lambda_{pixel}L_{1} + \lambda_{vgg}L_{perceptual} + \lambda_{smooth}L_{smooth}$$

  * **Pixel Loss ($L_1$):** Ensures color accuracy.
  * **VGG Loss:** Ensures structural/perceptual quality.
  * **Smoothness Loss:** Penalizes the second derivative of the affine parameters to ensure the camera path is smooth (minimizes "jerk").

-----

## 🛠️ Installation & Usage

### 1\. Prerequisites

Ensure you have Python installed with the necessary libraries:

```bash
pip install torch torchvision opencv-python numpy tqdm scikit-image
```

### 2\. Running the Training

1.  Set the `KAGGLE_ROOT` in the `CONFIG` dictionary to point to your dataset.
2.  Run the script:

<!-- end list -->

```bash
python train_full_pipeline.py
```

*Note: If no dataset is found, the script will automatically generate **dummy noise data** for debugging purposes.*

### 3\. Output

  * **Console:** Displays real-time training progress with `tqdm`.
  * **Metrics:** Logs **PSNR** and **Cropped SSIM** at the end of every epoch.
  * **Checkpoints:**
      * `best_stab_gan.pth`: Saved whenever PSNR improves.
      * `final_stab_gan.pth`: Saved at the end of training.

-----

## 📊 Evaluation Metric: Cropped SSIM

Standard SSIM is unreliable for video stabilization because the warping process introduces black borders (empty pixels) at the edges of the frame.

This project implements `calculate_ssim_cropped`:

1.  It automatically crops the **outer 10%** of the frame.
2.  It calculates SSIM only on the center content.
3.  This ensures the metric measures image quality, not border artifacts.

