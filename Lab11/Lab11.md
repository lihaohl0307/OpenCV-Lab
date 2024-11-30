## **Advanced Image Processing and Generative Models**

This project demonstrates three key areas in modern machine learning and computer vision:

1. **FGSM (Fast Gradient Sign Method)**: An adversarial attack technique.
2. **Diffusion Models**: Noise-based generative models and denoising methods.
3. **Generative Adversarial Networks (GANs)**: Deep learning models for generating synthetic data.

---

### **1. FGSM (Fast Gradient Sign Method)**

#### **Overview**
FGSM introduces adversarial noise to an image, which may cause a model to misclassify it. This demonstrates the vulnerability of neural networks to adversarial attacks.

#### **Key Steps**
- Load a grayscale image.
- Generate random noise.
- Add noise to the original image to create an adversarial image.

#### **Code Snippet**
```python
# Load original image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Create adversarial noise
noise = np.random.normal(0, 25, image.shape).astype('uint8')

# Add noise to create adversarial image
adversarial_image = cv2.addWeighted(image, 1.0, noise, 0.1, 0)
```

#### **Output**
Displays the adversarial image with subtle noise applied.

---

### **2. Diffusion Models**

#### **Overview**
Diffusion models generate data by incrementally adding noise to an image and then learning to denoise it. This section simulates noise addition and removes it using a Gaussian blur.

#### **Key Steps**
1. **Add Noise in Steps**:
   - Apply incremental Gaussian noise over multiple steps to simulate diffusion.
2. **Denoise the Image**:
   - Use a Gaussian blur to remove noise from the image.

#### **Code Snippet**
```python
# Add Gaussian noise in steps
for i in range(5): 
    noise = np.random.normal(0, 0.1 * (i + 1), image.shape)
    noisy_image = np.clip(image + noise, 0, 1)

# Denoise using Gaussian Blur
denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)
```

#### **Output**
- Displays noisy images after each step.
- Shows the denoised image after applying Gaussian blur.

---

### **3. Generative Adversarial Networks (GANs)**

#### **Overview**
GANs consist of two neural networks:
- **Generator**: Generates synthetic data.
- **Discriminator**: Distinguishes between real and fake data.
They compete in a zero-sum game, resulting in a generator that can create realistic data.

#### **Implementation**
- **Dataset**: Uses MNIST handwritten digits.
- **Training**:
  - The discriminator learns to classify real and fake images.
  - The generator learns to produce images that fool the discriminator.

#### **Key Components**
1. **Generator Architecture**:
   ```python
   self.model = nn.Sequential(
       nn.Linear(latent_dim, 128),
       nn.ReLU(),
       nn.Linear(128, 256),
       nn.ReLU(),
       nn.Linear(256, img_dim),
       nn.Tanh()
   )
   ```

2. **Discriminator Architecture**:
   ```python
   self.model = nn.Sequential(
       nn.Linear(img_dim, 256),
       nn.LeakyReLU(0.2),
       nn.Linear(256, 128),
       nn.LeakyReLU(0.2),
       nn.Linear(128, 1),
       nn.Sigmoid()
   )
   ```

3. **Training Loop**:
   - Train discriminator on real and fake data.
   - Train generator to produce data that fools the discriminator.

#### **Output**
- Logs discriminator and generator losses during training.
- Saves generated images every 5 epochs.

---

### **4. Execution**

#### **Requirements**
Install the required Python packages:
```bash
pip install torch torchvision opencv-python numpy
```

#### **Run the Script**
Execute the script to run all three sections:
```bash
python advanced_image_processing.py
```

---

### **5. Results**

1. **FGSM**:
   - Generates an adversarial image with added noise.

2. **Diffusion Models**:
   - Visualizes the process of adding noise incrementally.
   - Displays the denoised image.

3. **GANs**:
   - Produces synthetic MNIST images saved as PNG files.

---

### **6. Observations**
- **FGSM**: Highlights the need for robust models against adversarial attacks.
- **Diffusion Models**: Demonstrates noise addition and denoising as part of generative modeling.
- **GANs**: Illustrates the power of adversarial training for generating realistic data.

---

### **7. Conclusion**
This project showcases fundamental techniques in adversarial attacks, generative modeling, and denoising. Each method has significant applications in machine learning and image processing.

