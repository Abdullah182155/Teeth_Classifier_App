# Teeth Classification – Preprocessing, Visualization, and Model Training

This repository contains the first phase of our AI-driven dental imaging project. The objective is to develop an end-to-end deep learning pipeline that preprocesses, visualizes, and classifies dental images into **seven distinct categories** of teeth. This model will be integrated into our healthcare solutions to improve diagnostic accuracy and enhance patient outcomes.

## 🎯 Project Goals

- Build a robust **computer vision model** to classify dental images into 7 categories.
- Preprocess and normalize the data for better model performance.
- Apply data augmentation to improve model generalization.
- Visualize image distributions and transformations for data understanding.
- Track model experiments using **MLflow**.

---

## 🧠 Why This Matters

Accurate classification of teeth is critical for our dental AI tools. This project aligns with our broader strategic healthcare goals, enabling faster diagnostics, improved patient experiences, and data-driven dental care.

---

## 🧰 Tools & Technologies

- **Jupyter Notebook**
- **TensorFlow / Keras** – for CNN modeling
- **MLflow** – for experiment tracking
- **Pandas, NumPy** – for data manipulation
- **Matplotlib, Seaborn** – for visualization
- **Pillow (PIL)** – for image preprocessing

---

## 📁 Repository Structure

```
project-root/
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── Notebook.ipynb                              # Notebook For Project
├── .gitignore                                  # Files to exclude from Git
└── mlruns/                                     # MLflow experiment logs
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/teeth-classification.git
cd teeth-classification
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open the `.ipynb` file to explore preprocessing, visualization, and training steps.

---

## 🧪 Project Components

### 1. 📷 Image Preprocessing

- Normalization to scale pixel values
- Augmentation: rotation, flipping, zooming to improve generalization
- Displaying original and augmented images for visual inspection

### 2. 📊 Data Visualization

- Class distribution plots to check for class imbalance
- Sample image grids before and after augmentation

### 3. 🤖 Model Training

- CNN architecture using `tensorflow.keras`
- Compilation with appropriate loss and metrics
- Training on the dataset with validation split
- Logging training runs to MLflow

### 4. 🧾 Experiment Tracking

Launch MLflow UI locally:

```bash
mlflow ui
```

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore model runs, metrics, and artifacts.

---

## 🧱 Example Output

- Accuracy and loss curves
- Comparison of original vs augmented images
- MLflow logs including hyperparameters, training duration, and final metrics

---

## ✅ Requirements

Listed in `requirements.txt`:

- tensorflow
- mlflow
- numpy
- pandas
- matplotlib
- seaborn
- Pillow

Install them via:

```bash
pip install -r requirements.txt
```

---

## 🏁 Next Steps

This is the **Week 1 submission**. The repository will be continuously updated with:

- Model evaluation and confusion matrix
- Hyperparameter tuning
- Deployment integration (API / cloud)
- Documentation and testing

---

## 👨‍💻 Contributors

- [Abdullah Ashraf] – Machine Learning Engineer


---

📌 For questions, contact the ML Engineering team or open an issue on this repository.
