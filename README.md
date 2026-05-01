# Intelligent Waste Detection & Classification System

## Overview
This project presents an intelligent system for **waste detection and classification** using **Deep Learning and Computer Vision**.  
It aims to assist users in **sorting waste efficiently** by identifying objects in images and providing **recycling recommendations**.

The system is built using the **TACO (Trash Annotations in Context) dataset**, which contains real-world images with multiple objects, complex backgrounds, and diverse conditions.

---

## Objectives
- Detect and classify different types of waste from images  
- Handle **multi-object scenarios** in real-world environments  
- Provide **recyclability information and sorting guidance**  
- Build an **interactive application** for end users  

---

## Approaches Explored
Several methods were tested and compared:

### 1. Single-Object Classification
- Custom CNN  
- Transfer Learning (ResNet18)  
❌ Limited performance due to small dataset size

### 2. Multi-Object CNN
- Classification + Bounding Box regression  
⚠️ Improved results but unable to detect multiple objects effectively

### 3. YOLO (Final Approach)
- YOLOv8n and YOLOv8s tested  
- Class grouping to reduce imbalance  
- Best model: **YOLOv8s (832px resolution)**  
✔️ Capable of real-time **multi-object detection and classification**

---

## Dataset
- **Name:** TACO (Trash Annotations in Context)  
- Real-world waste images  
- Multiple objects per image  
- Includes:
  - Categories (fine classes)
  - Supercategories (plastic, glass, metal, etc.)
  - Bounding boxes  

### Download 
- The dataset used in this project can be downloaded from Kaggle:  
https://www.kaggle.com/datasets/kneroma/tacotrashdataset/data
---

## Final Model Performance
- **Model:** YOLOv8s  
- **mAP50 (test):** 0.244  
- Handles:
  - Multi-object detection  
  - Complex scenes  
- Limitations mainly due to:
  - Class imbalance  
  - Limited data for some categories  

---

##  Features
-  Upload image(s)  
-  Automatic waste detection  
-  Classification of objects  
-  Recycling status (recyclable / non-recyclable)  
-  Sorting recommendations  
-  Confidence score visualization  
-  Session history & statistics  

---

## Application
Built with **Streamlit**, the app includes:

### Analysis Page
- Image upload  
- Detection results with bounding boxes  
- Object details:
  - Class  
  - Confidence score  
  - Recycling status  
  - Sorting advice  

### History Page
- Summary statistics  
- Distribution of detected classes  
- Recycling ratio  
- Export results (CSV)

---

## Impact
This project highlights how **AI can support environmental sustainability** by helping users make better recycling decisions and improving waste management practices.

## Authors

- **Mariem GAALICHE**
- **Omar ALHAJJ**

**Master 1 – Data Science**  
Université de Rouen Normandie, FRANCE  
2025–2026
