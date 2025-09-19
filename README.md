# 🏠 House Price Prediction

A machine learning web application that predicts house prices using the Boston Housing dataset. Built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

- **Machine Learning Pipeline**: Ridge regression with hyperparameter tuning using GridSearchCV
- **Interactive Web Interface**: User-friendly Streamlit app for real-time predictions
- **Data Preprocessing**: Automated handling of missing values and feature scaling
- **Model Evaluation**: Comprehensive metrics including RMSE and R² score
- **Cross-Validation**: 5-fold cross-validation for robust model assessment

## 📊 Model Performance

- **Test RMSE**: 5.0045
- **R² Score**: 0.6585
- **CV RMSE**: 5.7059 ± 1.8319
- **Best Parameters**: Ridge alpha = 10

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hashirshaikh23/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if not already trained)
   ```bash
   python -c "from src.train import train; train('data/HousingData.csv')"
   ```

## 🎯 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Making Predictions

1. Open the Streamlit app in your browser
2. Enter values for the following features:
   - **CRIM**: Per capita crime rate by town
   - **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft
   - **INDUS**: Proportion of non-retail business acres per town
   - **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
   - **NOX**: Nitric oxides concentration (parts per 10 million)
   - **RM**: Average number of rooms per dwelling
   - **AGE**: Proportion of owner-occupied units built prior to 1940
   - **DIS**: Weighted distances to five Boston employment centres
   - **RAD**: Index of accessibility to radial highways
   - **TAX**: Full-value property-tax rate per $10,000
   - **PTRATIO**: Pupil-teacher ratio by town
   - **B**: 1000(Bk - 0.63)² where Bk is the proportion of blacks by town
   - **LSTAT**: % lower status of the population

3. Click "Predict" to get the estimated house price in thousands of dollars

### Using the Python API

```python
from src.predict import load_model, predict_single

# Load the trained model
model_data = load_model('models/best_model.pkl')

# Make a prediction
sample_data = {
    'CRIM': 0.02731, 'ZN': 0.0, 'INDUS': 7.07, 'CHAS': 0.0,
    'NOX': 0.469, 'RM': 6.421, 'AGE': 78.9, 'DIS': 4.9671,
    'RAD': 2.0, 'TAX': 242.0, 'PTRATIO': 17.8, 'B': 396.90,
    'LSTAT': 9.14
}

prediction = predict_single(model_data, sample_data)
print(f"Predicted house price: ${prediction:.2f}k")
```

## 📁 Project Structure

```
house-price-prediction/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
├── src/                  # Source code
│   ├── __init__.py
│   ├── data_utills.py    # Data preprocessing utilities
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction utilities
│   └── evaluate.py       # Model evaluation functions
├── notebooks/            # Jupyter notebooks for EDA
│   ├── eda_and_experiments.ipynb
│   └── house-price-prediction.ipynb
├── models/               # Trained model files
│   └── best_model.pkl
└── data/                 # Dataset (not included in repo)
    └── HousingData.csv
```

## 🔧 Development

### Training a New Model

```bash
python -c "from src.train import train; train('path/to/your/data.csv')"
```

### Evaluating Model Performance

```python
from src.evaluate import evaluate
import joblib
from sklearn.model_selection import train_test_split

# Load your test data
X_test, y_test = # your test data

# Evaluate the model
rmse, r2 = evaluate('models/best_model.pkl', X_test, y_test)
```

## 📈 Dataset

This project uses the Boston Housing dataset, which contains information about housing prices in Boston suburbs. The dataset includes 13 features and 506 samples.

**Target Variable**: MEDV (Median value of owner-occupied homes in $1000's)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Hashir Shaikh**
- GitHub: [@Hashirshaikh23](https://github.com/Hashirshaikh23)

## 🙏 Acknowledgments

- Boston Housing dataset
- scikit-learn library
- Streamlit framework
- The open-source community

---

⭐ **Star this repository if you found it helpful!**
