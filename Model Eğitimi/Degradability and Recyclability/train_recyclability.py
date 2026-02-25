import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, GraphDescriptors, Fragments
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

df = pd.read_csv('cleaned_enthalpy_data.csv')
df = df.dropna(subset=['roe_kj/mol', 'smiles_monomer'])

y = df['roe_kj/mol']
smiles_list = df['smiles_monomer'].tolist()
inv_length = df['1/length'].values

def get_advanced_features(smiles, inv_len_val):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        feats = {'inv_length': inv_len_val}
        
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
        if charges:
            feats['MaxAbsPartialCharge'] = max(abs(c) for c in charges)
            feats['MaxPartialCharge'] = max(charges)
            feats['MinPartialCharge'] = min(charges)
        else:
            feats['MaxAbsPartialCharge'] = 0

        ring_info = mol.GetRingInfo()
        feats['IsRing3'] = 1 if any(len(r) == 3 for r in ring_info.AtomRings()) else 0
        feats['IsRing4'] = 1 if any(len(r) == 4 for r in ring_info.AtomRings()) else 0
        feats['IsRing5'] = 1 if any(len(r) == 5 for r in ring_info.AtomRings()) else 0
        feats['IsRing6'] = 1 if any(len(r) == 6 for r in ring_info.AtomRings()) else 0
        feats['IsRing7plus'] = 1 if any(len(r) >= 7 for r in ring_info.AtomRings()) else 0
        feats['NumRings'] = ring_info.NumRings()
        feats['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
        feats['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
        feats['Num_Esters'] = Fragments.fr_ester(mol)
        feats['Num_Amides'] = Fragments.fr_amide(mol)
        feats['Num_Ethers'] = Fragments.fr_ether(mol)
        feats['MolWt'] = Descriptors.MolWt(mol)
        feats['TPSA'] = Descriptors.TPSA(mol)
        feats['RotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        feats['FractionCSP3'] = Descriptors.FractionCSP3(mol)
        feats['BalabanJ'] = GraphDescriptors.BalabanJ(mol)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
        fp_bits = list(fp.ToBitString())
        for i, bit in enumerate(fp_bits):
            feats[f'FP_{i}'] = int(bit)
        return feats
    except:
        return None

data = [get_advanced_features(s, inv_length[i]) for i, s in enumerate(smiles_list)]
X = pd.DataFrame([d for d in data if d is not None])
y = y.iloc[X.index].reset_index(drop=True)
X = X.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_estimators = [
    ('xgb', XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
    ('rf', RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)),
    ('lgbm', LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=1000, early_stopping=True, random_state=42)),
    ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance')),
    ('ridge', Ridge(alpha=0.6)),
    ('et', ExtraTreesRegressor(n_estimators=400, max_depth=25, random_state=42, n_jobs=-1))
]

stacking_model = StackingRegressor(
    estimators=base_estimators,
    final_estimator=RidgeCV(),
    cv=5,
    n_jobs=-1
)

pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', stacking_model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

comparison = []
for name, model in base_estimators:
    temp_pipe = Pipeline([('scaler', RobustScaler()), ('m', model)])
    temp_pipe.fit(X_train, y_train)
    p = temp_pipe.predict(X_test)
    comparison.append({'Model': name.upper(), 'R2': r2_score(y_test, p)})

comparison.append({'Model': 'STACKING', 'R2': r2})
comparison_df = pd.DataFrame(comparison).sort_values(by='R2', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='R2', y='Model', data=comparison_df, palette='viridis')
plt.title('R2 Comparison')
plt.show()