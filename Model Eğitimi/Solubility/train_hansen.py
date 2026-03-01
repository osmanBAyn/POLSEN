import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings("ignore")

df = pd.read_excel('Database_Hansen.xlsx')
df = df.dropna(subset=['SMILES', 'PARAMETRO'])
df['PARAMETRO'] = pd.to_numeric(df['PARAMETRO'], errors='coerce')
df = df.dropna(subset=['PARAMETRO'])
#df = df[df['PARAMETRO'] >= 10]
#df = df[df['PARAMETRO'] <= 40]
#plt.hist(df['PARAMETRO'], bins = 50)

def get_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_arr = np.zeros((1,))
        ConvertToNumpyArray(fp, fp_arr)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        num_rotatable = Descriptors.NumRotatableBonds(mol)
        return np.concatenate((fp_arr, [mw, logp, hbd, hba, tpsa, num_rotatable]))
    except:
        return None

df['Features'] = df['SMILES'].apply(get_features)
df = df.dropna(subset=['Features'])

X = np.stack(df['Features'].values)
y = df['PARAMETRO'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modeller = {
    "XGBoost": xgb.XGBRegressor(n_estimators=1155, learning_rate=0.06840304292333116, max_depth=10, subsample=0.9988910981196208, colsample_bytree=0.9594208739956043, min_child_weight=1, gamma=3.180817839721346e-07, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42, n_jobs=-1, verbose=-1),
    "MLP (Sinir Ağı)": MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1),
    "Ridge Regression": Ridge(alpha=1.0)
}

for model_adi, temel_model in modeller.items():
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', temel_model)
    ])

    log_model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    log_model.fit(X_train, y_train)
    
    y_pred_train = log_model.predict(X_train)
    y_pred_test = log_model.predict(X_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print(f"--- {model_adi} ---")
    print(f"MAE: {mae_test:.3f} | RMSE: {rmse_test:.3f} | R²: {r2_test:.3f}")

    plt.figure(figsize=(9, 8))
    plt.scatter(y_train, y_pred_train, alpha=0.5, c='#1f77b4', label=f'Eğitim Seti (n={len(y_train)})')
    plt.scatter(y_test, y_pred_test, alpha=0.8, c='#ff7f0e', edgecolor='k', s=60, label=f'Test Seti (n={len(y_test)})')

    min_val = min(min(y), min(y_pred_test), min(y_pred_train))
    max_val = max(max(y), max(y_pred_test), max(y_pred_train))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='İdeal Tahmin')

    plt.xlabel('Gerçek PARAMETRO Değerleri', fontsize=12, fontweight='bold')
    plt.ylabel(f'{model_adi} Tahminleri', fontsize=12, fontweight='bold')
    plt.title(f'{model_adi} Tahmin Performansı\nTest MAE: {mae_test:.2f} | R²: {r2_test:.2f}', fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()
