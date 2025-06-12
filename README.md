# Image-Classification
# 🃏 Joker vs 💀 Thanos - Image Classifier using CNN

A fun yet powerful **Deep Learning project** built using **TensorFlow and Keras** to classify images of two iconic characters — **Joker** and **Thanos** — using a Convolutional Neural Network (CNN).

---

## 🧠 Overview

This project is a binary image classifier that distinguishes between Joker and Thanos images. It uses a simple CNN architecture with Conv2D, MaxPooling, and Dense layers and is trained on a custom dataset.

---

## 📂 Dataset Structure

Organized into training and validation folders:

Dataset/
├── train/
│ ├── joker/
│ └── thanos/
└── val/
├── joker/
└── thanos/

yaml
Copy
Edit

Ensure all images are in `.jpeg`, `.jpg`, or `.png` format.

---

## 🚀 Technologies Used

- Python 🐍  
- TensorFlow / Keras  
- NumPy  
- ImageDataGenerator (for data augmentation)  
- OpenCV (optional for visualization)

---

## 🏗️ Model Architecture

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
Loss Function: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy

📊 Training Summary
Epochs: 50

Batch Size: 8

Image Size: 64x64

Data Augmentation: Rescaling, Zoom, Shear, Horizontal Flip

💾 Model Saving
Architecture saved as model.json

Weights saved as model.weights.h5

You can reload the model using:

python
Copy
Edit
from keras.models import model_from_json

with open("model.json", "r") as file:
    model = model_from_json(file.read())

model.load_weights("model.weights.h5")
🔍 Image Prediction Script
python
Copy
Edit
def classify(img_file):
    test_image = image.load_img(img_file, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    if result[0][0] >= 0.5:
        print("Predicted: Thanos")
    else:
        print("Predicted: Joker")
📷 Sample Predictions
Add sample images or screenshots here

📌 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/your-username/joker-vs-thanos-cnn.git
Install requirements:

bash
Copy
Edit
pip install tensorflow keras numpy
Prepare the dataset in the structure shown above.

Train the model:

python
Copy
Edit
python train.py
Test predictions:

python
Copy
Edit
python predict.py
🙌 Credits
Project by Your Name
Inspired by the blend of AI and Pop Culture 🤖🎬

🏷️ Tags & Topics
#DeepLearning #CNN #MachineLearning #AI #Keras #TensorFlow #ImageClassification #Joker #Thanos #Python #ComputerVision #OpenSource #PopCultureInTech

⭐ Star this repo if you liked it!
yaml
Copy
Edit

---

Let me know if you'd like me to generate:
- A `requirements.txt`
- A starter `train.py` and `predict.py`
- A GitHub banner or badges for stars/forks/python version

Just say the word!
