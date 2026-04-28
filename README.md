# 🩺 Skin Disease Detection using Images

A complete end-to-end deep learning project that classifies skin diseases from images using Transfer Learning (MobileNetV2). Built with **Python + TensorFlow/Keras** backend and a modern **HTML/CSS/JavaScript** frontend.

---

## 📁 Project Structure

```
Skin-Disease-Detection/
│
├── model/
│   ├── train_model.py          # Model training script
│   ├── skin_disease_model.h5   # Saved trained model (generated after training)
│   └── class_indices.npy       # Class label mappings (generated after training)
│
├── backend/
│   └── app.py                  # Flask REST API server
│
├── frontend/
│   ├── index.html              # Main web page
│   ├── style.css               # Modern responsive styling
│   └── script.js               # Frontend logic & API integration
│
├── dataset/                    # Training images (organized by class folders)
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ✨ Features

- ✅ **Transfer Learning** with MobileNetV2 for high accuracy
- ✅ **REST API** built with Flask for prediction endpoints
- ✅ **Modern Web UI** with drag-and-drop image upload
- ✅ **Real-time predictions** with confidence scores
- ✅ **Disease descriptions** and recommendations displayed
- ✅ **Loading animations** while analyzing
- ✅ **Probability bar charts** for all disease classes
- ✅ **Responsive design** - works on mobile and desktop
- ✅ **CORS enabled** for seamless frontend-backend communication

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone / Download the Project

```bash
cd "Skin-Disease-Detection"
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 **Note:** TensorFlow installation may take a few minutes.

---

## 📊 Dataset Preparation

### Option A: Use HAM10000 Dataset (Recommended)

1. Download from [Kaggle - Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Organize images into folders by class:

```
dataset/
├── train/
│   ├── acne/
│   ├── eczema/
│   ├── melanoma/
│   ├── psoriasis/
│   └── normal/
└── val/
    ├── acne/
    ├── eczema/
    ├── melanoma/
    ├── psoriasis/
    └── normal/
```

### Option B: Use Your Own Dataset

Organize your images similarly with class-named folders.

### Option C: Test Mode (Dummy Data)

If no dataset is provided, the training script automatically generates dummy images for testing the pipeline.

---

## 🧠 Training the Model

```bash
python model/train_model.py
```

**What happens:**
1. Loads and preprocesses images (224x224)
2. Applies data augmentation (rotation, zoom, flip, etc.)
3. Trains MobileNetV2 with custom top layers
4. Fine-tunes the last 30 layers of the base model
5. Saves the best model to `model/skin_disease_model.h5`
6. Saves class mappings to `model/class_indices.npy`
7. Generates training history plot

**Training tips:**
- More images = better accuracy
- Minimum 20-30 images per class recommended
- Training time: 10-60 minutes depending on dataset size

---

## 🔌 Running the Backend API

```bash
python backend/app.py
```

The Flask server will start at: **http://localhost:5000**

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check & model status |
| GET | `/classes` | List disease classes & info |
| POST | `/predict` | Upload image, get prediction |

### Example API Request (cURL)

```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

### Example Response

```json
{
  "success": true,
  "prediction": "melanoma",
  "confidence": 94.23,
  "all_probabilities": {
    "melanoma": 94.23,
    "psoriasis": 3.12,
    "eczema": 1.45,
    "normal": 0.80,
    "acne": 0.40
  },
  "info": {
    "description": "The most serious type of skin cancer...",
    "symptoms": "Asymmetrical moles, irregular borders...",
    "recommendation": "⚠️ Please consult a dermatologist immediately..."
  }
}
```

---

## 🖥️ Running the Frontend

### Option 1: Direct Open (Simplest)

Simply open `frontend/index.html` in your web browser.

> ⚠️ Some browsers may block CORS when opening files directly. Use Option 2 for best results.

### Option 2: Using Live Server (Recommended)

1. Install VS Code extension: **"Live Server"**
2. Right-click `frontend/index.html` → **"Open with Live Server"**

Or use Python's built-in server:

```bash
cd frontend
python -m http.server 5500
```

Then open: **http://localhost:5500**

### Option 3: Using Node.js (npx serve)

```bash
npx serve frontend -p 5500
```

---

## 🔄 Full Workflow

1. **Start Backend** (Terminal 1):
   ```bash
   python backend/app.py
   ```

2. **Start Frontend** (Terminal 2):
   ```bash
   cd frontend
   python -m http.server 5500
   ```

3. **Open Browser** and go to `http://localhost:5500`

4. **Upload an image** and click "Analyze Image"

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow 2.x / Keras |
| Transfer Learning | MobileNetV2 (ImageNet weights) |
| Backend API | Flask + Flask-CORS |
| Frontend | HTML5 + CSS3 + Vanilla JavaScript |
| Image Processing | Pillow, NumPy |

---

## 📸 Screenshots

The UI includes:
- 🎨 Modern gradient design
- 📤 Drag-and-drop upload zone
- 👁️ Image preview before analysis
- ⏳ Animated loading spinner
- 📊 Results with confidence badge
- 📋 Disease descriptions & recommendations
- 📈 Animated probability bar charts

---

## ⚠️ Disclaimer

> **This tool is for educational purposes only.** It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition. Never disregard professional medical advice because of something you have read or inferred from this application.

---

## 📝 Customization

### Adding New Disease Classes

1. Add new folder in `dataset/` with class name
2. Retrain the model: `python model/train_model.py`
3. Add disease info in `backend/app.py` → `DISEASE_INFO` dictionary
4. Add icon in `frontend/script.js` → `DISEASE_ICONS` object

### Changing Model Architecture

Edit `model/train_model.py`:
- Change `MobileNetV2` to `ResNet50` or `EfficientNetB0`
- Adjust dense layer sizes
- Modify dropout rates

### Deploying to Production

1. Use a production WSGI server (Gunicorn)
2. Deploy backend to Heroku, AWS, or GCP
3. Host frontend on Netlify or Vercel
4. Update `API_BASE_URL` in `frontend/script.js`

---

## 📚 References

- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [HAM10000 Dataset Paper](https://arxiv.org/abs/1803.10417)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

---

## 🤝 Contributing

Feel free to fork this project and submit pull requests for:
- Additional disease classes
- Improved model accuracy
- Better UI/UX
- Bug fixes

---

## 📄 License

This project is for educational purposes. Please ensure compliance with medical data privacy laws (HIPAA, GDPR) if using real patient data.

---

## 💡 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model not found" | Run `python model/train_model.py` first |
| CORS errors | Ensure backend is running on port 5000 |
| Low accuracy | Add more training images per class |
| Out of memory | Reduce `BATCH_SIZE` in `train_model.py` |
| Slow predictions | Ensure GPU is available or use smaller model |

---

**Made with ❤️ for AI/ML Education**

