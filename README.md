# Smart-Product-Catalog-System
📌 Objective

This project implements a mini computer vision pipeline for product recognition and generation using the Fruits 360 dataset.
The pipeline covers image classification, object detection, and synthetic image generation.


---

📂 Project Structure

classification/     # Transfer Learning models (EfficientNetB0, ResNet50, DenseNet121)
object_detection/   # YOLOv5 for fruit detection
gan/                # GAN-based synthetic fruit image generation
README.md           # This file
requirements.txt    # Required Python packages


---

📊 Tasks Implemented

1️⃣ Image Classification – Transfer Learning

Dataset: Fruits 360 – Kaggle

Models Implemented:

EfficientNetB0

ResNet50

DenseNet121


Steps:

Preprocessing and augmentation using ImageDataGenerator

Fine-tuning the final layers of pretrained CNNs

Evaluated with accuracy and confusion matrix




---

2️⃣ Object Detection – YOLO

Selected 20–30 images from Fruits 360

Annotated with LabelImg

Trained YOLO model on custom fruit dataset

Output: Bounding boxes and labels for detected fruits in test images



---

3️⃣ GAN-Based Image Generation

Dataset subset used for training

Implemented a simple GAN in PyTorch for synthetic fruit image generation

Generated 10 synthetic fruit images

Bonus: Added labels for generated images



---

📦 Requirements

Install all dependencies with:

pip install -r requirements.txt

Main packages:

tensorflow

torch, torchvision

opencv-python

matplotlib

numpy

pillow



---

🚀 How to Run

1. Classification

cd classification
python transfer_learning.py

Outputs accuracy, confusion matrix, and saved models.


2. YOLO Object Detection

cd object_detection
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt

Outputs detection results in runs/detect/.


3. GAN Image Generation

cd gan
python train_gan.py

Saves generated fruit images in gan/output_images/.



---

📌 Results Summary

Task	Best Model / Method	Key Metric

Classification	EfficientNetB0	Highest accuracy
Detection	YOLOv5	High mAP on fruits
Generation	Simple GAN	10 synthetic images generated



---

💡 Potential Applications

E-commerce: Automated fruit classification for product listings

Inventory systems: Detect and count fruits in storage

Data augmentation: GAN-generated images for training larger models
