import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor

df = pd.read_csv('polymer_degradability.csv')
df = df.dropna(subset=['SMILES', 'score'])

def get_advanced_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None

        res = {
            'TPSA': Descriptors.TPSA(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'MolWt': Descriptors.MolWt(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
        }

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_arr = np.zeros((0,), dtype=np.int8)
        ConvertToNumpyArray(fp, fp_arr)

        for i, val in enumerate(fp_arr):
            res[f'FP_{i}'] = val
        return res
    except:
        return None

advanced_data = []
scores = []
for index, row in df.iterrows():
    feats = get_advanced_features(row['SMILES'])
    if feats:
        advanced_data.append(feats)
        scores.append(row['score'])

X = pd.DataFrame(advanced_data)
y = pd.Series(scores)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators = 1400, learning_rate = 0.03, subsample = 0.8, colsample_bytree = 0.7, random_state=42, verbose=-1), # Hist yerine LGBMRegressor kullanıldı, terminali loga boğmaması için verbose=-1 eklendi
    'MLP (Neural Net)': MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=500, random_state=42, early_stopping=True),
    'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    'Ridge': Ridge(alpha=1.0)
}

performance_results = []

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    performance_results.append({'Model': name, 'RMSE': rmse, 'R2 Score': r2})

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Gerçek Skor')
    plt.ylabel(f'{name} Tahmini')
    plt.title(f'{name} Performansı (R2: {r2:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(performance_results).sort_values(by='R2 Score', ascending=False)
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='R2 Score', y='Model', data=results_df, palette='viridis')
plt.title('Model Karşılaştırması')
plt.xlabel('R2 Skoru')
plt.xlim(0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
