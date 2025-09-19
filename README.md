# NASA Turbofan Predictive Maintenance 🚀  

![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)  
![pandas](https://img.shields.io/badge/pandas-2.2.3-green.svg)  
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)


This project demonstrates a **predictive maintenance pipeline** using the NASA C-MAPSS Turbofan Engine dataset.  
We train a machine learning model to estimate the **Remaining Useful Life (RUL)** of aircraft engines, evaluate it on test data, and visualize important features.

---

## 📂 Project Structure

```
NASA_Turbofan_Predictive_Maintenance/
├── data/                 # datasets (generated or NASA/Kaggle)
│   ├── train_FD001.txt           # full dataset (optional)
│   ├── train_FD001_demo.txt      # tiny demo dataset (included)
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── reports/              # outputs
│   ├── rf_fd001.joblib   # saved RandomForest model
│   ├── predictions_fd001.csv
│   └── figs/
│       └── feature_importances.png
├── src/                  # source code
│   ├── preprocess.py     # data loader + RUL labeler (auto demo fallback)
│   ├── train.py          # training pipeline
│   ├── predict.py        # evaluate on test set
│   └── inspect_model.py  # inspect saved model
├── tools/
│   └── simulate_cmapss_fd001.py  # synthetic dataset generator
└── README.md
```

---

## ⚙️ Setup

1. **Clone this repo**  
   ```bash
   git clone https://github.com/ArmandoSNHU/NASA_Turbofan_Predictive_Maintenance.git
   cd NASA_Turbofan_Predictive_Maintenance
   ```

2. **Create virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # mac/linux
   venv\Scripts\activate      # windows
   ```

3. **Install requirements**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare dataset**  
   - By default, the code will look for `data/train_FD001.txt`.  
   - If it is not found, it **automatically falls back** to the included **demo file** `train_FD001_demo.txt` (10 rows) so you can run everything instantly.  
   - To use the full dataset, download it from the official [NASA CMAPSS dataset](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/ff5v-kuh6) and place the files in `data/`.

---

## 🏋️ Training

Train a RandomForest model and save it:

```bash
python src/train.py
```

Output (with demo dataset):
```
Dataset shape: (10, 26)
✅ Validation RMSE: 0.21
💾 Saved model -> reports/rf_fd001.joblib
```

Output (with full dataset):
```
Dataset shape: (18242, 27)
✅ Validation RMSE: 72.12
💾 Saved model -> reports/rf_fd001.joblib
```

---

## 🔮 Prediction

Evaluate on test set:

```bash
python src/predict.py
```

Output (sample):
```
Predictions (first 10 engines):
 unit  pred_RUL  true_RUL   error
    1     120.5      112    8.5
    2      95.7      102   -6.3
    3     145.2      150   -4.8
...

Metrics on test last-cycle per engine:
  RMSE: 10.24
  MAE : 7.45
  MAPE: 6.2%
```

✅ Saved predictions -> `reports/predictions_fd001.csv`

---

## 🔍 Inspect Model

Check parameters and plot feature importances:

```bash
python src/inspect_model.py
```

Output:
```
✅ Loaded model
RandomForestRegressor(n_estimators=300, ...)
Number of trees: 300
📊 Saved feature importance chart -> reports/figs/feature_importances.png
```

Example chart:

![Feature Importances](reports/figs/feature_importances.png)

---

## 📊 Example Visualization

Sensor drift for one engine over cycles:

```python
import matplotlib.pyplot as plt
from preprocess import load_fd001, add_rul

df = add_rul(load_fd001())
engine1 = df[df["unit"] == 1]
engine1.plot(x="cycle", y="sensor_2", title="Engine 1 - Sensor 2 over cycles")
plt.show()
```

---

## 📌 Requirements

```txt
numpy==2.1.2
pandas==2.2.2
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.13.2
joblib==1.4.2
jupyter==1.0.0
```

---

## 🚀 Next Steps
- Try more advanced models (XGBoost, LightGBM, LSTMs for sequences).  
- Deploy a simple **Streamlit dashboard** to upload engine logs and return predicted RUL.  
- Add cross-dataset evaluation (FD002, FD003, FD004).  

---

👨‍💻 Built as a portfolio-ready AI/ML project for predictive maintenance.
