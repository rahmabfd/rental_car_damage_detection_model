# 🚗 Car Damage Detection using YOLOv11 & Web Integration  

## 📌 Project Overview  
This project focuses on building an **AI-powered system for automatic car damage detection**, designed to assist car rental companies in comparing a vehicle’s condition **before and after rental periods**. The solution leverages **computer vision** and **deep learning** to identify and classify different types of damages on cars with high accuracy.  

I followed the **Agile Scrum methodology** over 2 months, iteratively improving the dataset, model, and integration.  

---

## 🎯 Objectives  
- Automate the **damage inspection process** for rental cars.  
- Detect and classify **different categories of damages** (scratches, dents, bumper damage, etc.).  
- Provide an **easy-to-use web application** for users to upload images and get instant damage detection results.  
- Reduce **human error** and **time consumption** in manual car inspection.  

---

## 🧠 Dataset & Preprocessing  
- Collected and curated a dataset tailored to car damage detection.  
- **Data Cleaning & Augmentation** performed to ensure robustness.  
- Dataset annotated and managed via **Roboflow**.  

### Classes of Damage Detected (22 categories):  
- damaged-door  
- damaged-hood   
- damaged-rear-bumper   
- damaged-roof   
- damaged-running-board
- damaged-trunk 
- damaged-window   
- dent   
- fender-damage
- front-bumper-damage   
- front-windscreen-damage  
- headlight-damage   
- pillar-damage   
- quarter-panel-damage  
- rear-windscreen-damage   
- scratch   
- side-mirror-damage   
- taillight-damage
- tire-flat  
- damaged-windshield-wiper  
- dirty 
- missing-parts  

---

## ⚙️ Model Training  
- Model trained on **YOLOv11 Object Detection (Accurate)** via **Roboflow**.  
- Evaluation Metrics:  
  - **Precision**: 81%  
  - **Recall**: 75%  
  - **mAP@50**: 75%  

This ensures the model achieves a good balance between correctly detecting damages and minimizing false detections.  

---

## 🌐 Web Application  
To make the model practical and accessible, I developed a **web application** integrating the trained YOLOv11 model.  

### Tech Stack  
- **Frontend**: React  
- **Backend**: Flask (Python)  
- **Database**: MongoDB  
- **Model Deployment**: Integrated YOLOv11 for inference  

### Features  
✅ Upload car images for analysis  
✅ Detect and classify damages in real-time  
✅ User-friendly interface for rental companies  
✅ Database-backed system for managing inspection history  

---

## 🚀 Installation & Usage  

### 1. Clone Repository  
```bash
git clone https://github.com/your-username/car-damage-detection.git
cd car-damage-detection

