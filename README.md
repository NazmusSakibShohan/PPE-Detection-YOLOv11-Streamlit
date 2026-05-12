# 🛡️ PPE Detection System using YOLOv11 & Streamlit

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-green.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This is a real-time Personal Protective Equipment (PPE) detection system developed to enhance safety monitoring in construction sites and industrial environments. The project utilizes the state-of-the-art **YOLOv11** architecture for object detection and a **Streamlit** web interface for easy accessibility.

---

## 🚀 Key Features
- **Real-time Surveillance:** Detects safety gear via live webcam feed.
- **Image Processing:** Supports uploading static images (JPG, PNG, JPEG) for detailed compliance checks.
- **Dynamic Thresholding:** Adjust confidence scores in real-time to detect small or transparent objects.
- **Modern UI:** Clean and responsive dashboard built with Streamlit.

## 🛠️ Detection Classes
The model is trained/configured to identify:
- ✅ **Hardhat / Helmet**
- ✅ **Safety Vest**
- ✅ **Mask**
- ❌ **No-Hardhat / No-Vest (Safety Violations)**

---

## 📂 Project Structure
```text
Image-Processing/
├── app.py                 # Main Streamlit Web Application
├── best.pt                # Trained YOLOv11 Model Weights
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```
---

## ⚙️ Setup and Installation

 **1. Clone the repository:**
 ```bash
 git clone [https://github.com/NazmusSakibShohan/PPE-Detection-YOLOv11-Streamlit.git](https://github.com/NazmusSakibShohan/PPE-Detection-YOLOv11-Streamlit.git)
 cd PPE-Detection-YOLOv11-Streamlit
 ```

**2. Create a Virtual Environment (Recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the Web App:**
```bash
streamlit run app.py
```
---

## 📊 Technical Insights & Challenges
 During development and testing at Dhaka International University, several observations were made:

- **Accuracy:** YOLOv11 provides exceptional speed and accuracy for solid objects like Helmets and Vests.
- **The Goggles Challenge:** Small, transparent objects like Safety Goggles can be difficult to detect in low-contrast environments. This project addresses this by allowing users to lower the Confidence Threshold via the UI.
- **Real-world Application:** Designed for low-latency performance, making it suitable for edge devices.
---
## 👨‍💻 Author
 **Nazmus Sakib Shohan**   
 **Dhaka International University (DIU)** 
 
---
## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
