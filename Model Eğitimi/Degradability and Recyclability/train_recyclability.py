import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Fragments
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

df = pd.read_csv('cleaned_enthalpy_data.csv')
df = df.dropna(subset=['roe_kj/mol', 'smiles_polymer'])
y = df['roe_kj/mol']
smiles_list = df['smiles_polymer'].tolist()

def get_multi_len_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        feats = {}
        
        n_total = mol.GetNumHeavyAtoms() # ~repeating unit size
        feats['inv_len_total'] = 1.0 / n_total if n_total > 0 else 0
        
        dummy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
        if len(dummy_indices) >= 2:
            path = Chem.GetShortestPath(mol, dummy_indices[0], dummy_indices[1])
            n_backbone = len(path) - 2
            feats['inv_len_backbone'] = 1.0 / n_backbone if n_backbone > 0 else 0
            feats['branching_ratio'] = n_total / n_backbone if n_backbone > 0 else 1
        else:
            feats['inv_len_backbone'] = feats['inv_len_total']
            feats['branching_ratio'] = 1.0

        feats['NumRings'] = mol.GetRingInfo().NumRings()
        feats['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
        feats['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
        feats['Num_Esters'] = Fragments.fr_ester(mol)
        feats['Num_Ethers'] = Fragments.fr_ether(mol)
        feats['MolWt'] = Descriptors.MolWt(mol)
        feats['TPSA'] = Descriptors.TPSA(mol)
        feats['FractionCSP3'] = Descriptors.FractionCSP3(mol)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
        for i, bit in enumerate(fp.ToBitString()):
            feats[f'FP_{i}'] = int(bit)
            
        return feats
    except:
        return None

data = [get_multi_len_features(s) for s in smiles_list]

valid_indices = [i for i, d in enumerate(data) if d is not None]
X = pd.DataFrame([data[i] for i in valid_indices])
y = y.iloc[valid_indices].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
    "MLP (Neural Net)": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, max_depth = 8, subsample = 0.8, colsample_bytree = 0.8, random_state=42, n_jobs=-1, verbose=-1)
}

results = []

print("\n" + "="*65)
print(f"{'Model Adı':<20} | {'R2 Skoru':<10} | {'RMSE (kJ/mol)':<15}")
print("="*65)

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({'Model': name, 'R2': r2, 'RMSE': rmse})
    print(f"{name:<20} | {r2:.4f}     | {rmse:.4f}")

results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='R2', y='Model', data=results_df, palette='viridis')
plt.title('Tüm Aralık Polimer Entalpi Tahmini - Model Karşılaştırması')
plt.xlabel('R2 Skoru')
plt.xlim(0, 1.0)
for i, v in enumerate(results_df['R2']):
    plt.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
