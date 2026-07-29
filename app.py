import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
import operator
import time
import math
from stmol import showmol
import py3Dmol
import pubchempy as pcp
import deap.base as base
import deap.creator as creator
import deap.tools as tools
from deap import algorithms
import lightgbm as lgbm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies as sf
import smart_ga as sga  # chemistry-aware GA operators: seeding, fragment mutation, structure preservation
import retro            # rule-based polymer retrosynthesis (reliable disconnection to real monomers)
from datasets import load_dataset
import rdkit.Chem.rdChemReactions as rdChemReactions
import sys
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors
import rdkit.Chem.rdChemReactions as rdChemReactions
from rdkit.Chem import rdmolops
import requests
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
from rdkit import DataStructs
from rdkit.Chem import Fragments
from rdkit.Chem import GraphDescriptors
import  warnings
warnings.filterwarnings("ignore")
# translations.py
# Eski Hali:
# from translations import LANGUAGES

# Yeni Hali:
from lang_dict import LANGUAGES

# =====================================================================================
# CONFIG -- toggle app features here (safe to edit before publishing)
# =====================================================================================
SHOW_RELIABILITY = False    # show per-property model-reliability badges + legend + warnings
SHOW_MANUAL_ANALYSIS = True  # show the "Manual Polymer Analysis" expander
SHOW_PARETO_TABLE = True   # show the Pareto-front table/plot in the Evolution tab (NSGA-II)
USE_T5_RETRO = True       # try the (heavy) T5 retro model; off = rule engine only (recommended for deploy)

# set_page_config MUST be the first Streamlit command that renders anything,
# so it comes before the sidebar language selector below.
st.set_page_config(
    page_title="POLSEN",
    page_icon="🧬",
)

# Session State'te dil ayarı yoksa TR olarak başlat
if "lang" not in st.session_state:
    st.session_state["lang"] = "TR"

# Çeviri 
def _(text_key):
    return LANGUAGES[st.session_state["lang"]].get(text_key, text_key)

st.sidebar.markdown("### 🌍 Dil / Language")
selected_lang = st.sidebar.selectbox(
    "Dil Seçimi / Select Language",
    ["TR", "EN"],
    index=0 if st.session_state["lang"] == "TR" else 1,
    label_visibility="collapsed"
)

if selected_lang != st.session_state["lang"]:
    st.session_state["lang"] = selected_lang
    st.rerun()

@st.cache_resource
def load_my_trained_model():
    model_path = "OsBaran/POLSEN_T5"
    try:
        # Lazy import: torch/transformers are only needed when USE_T5_RETRO is on.
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("--- [LOG] Model yükleme fonksiyonu başladı...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("--- [LOG] Tokenizer yüklendi. Model yükleniyor...", flush=True) 

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval() 
        print("--- [LOG] Model Quantization işlemi yapılıyor...", flush=True)
        model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        print("--- [LOG] Model başarıyla hafızaya alındı! Hazır.", flush=True) 
        return tokenizer, model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None, None

def predict_monomers_local(polymer_smiles):
    """
    Monomer prediction with the reliable RULE-BASED disconnection as the primary
    engine (retro.retro_decompose returns real monomer SMILES for any polymer with
    a recognisable backbone linkage). The synthetic-data T5 is used only as a
    secondary source AND only when its output is chemically valid -- otherwise its
    out-of-distribution guesses on GA molecules are discarded.
    """
    # 1. Rule-based disconnection (trustworthy, real monomer structures).
    routes = retro.retro_decompose(polymer_smiles)
    rule_monomers = routes[0]['monomers'] if routes else None
    rule_verified = routes[0].get('verified', False) if routes else False

    # 2. Optional T5, gated by a chemical-validity check. Skipped entirely when
    #    USE_T5_RETRO is off (avoids the heavy transformer download on the server;
    #    the rule engine is the reliable primary anyway).
    tokenizer, model = load_my_trained_model() if USE_T5_RETRO else (None, None)
    ai_prediction = ""
    if model:
        try:
            inputs = tokenizer("retrosynthesis: " + polymer_smiles, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], max_length=64, num_beams=5, early_stopping=True)
            ai_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception:
            ai_prediction = ""

    if ai_prediction and " . " in ai_prediction:
        parts = [p.strip() for p in ai_prediction.split(" . ")]
        if parts and all(retro.is_valid_monomer(p) for p in parts):
            return f"{ai_prediction} (T5 Model)"   # trusted only when every monomer parses

    # 3. Prefer the rule-based monomers; fall back to raw T5 text only if rules failed.
    if rule_monomers:
        tag = "Rule-based" if rule_verified else "Rule-based, tentative"
        return f"{' . '.join(rule_monomers)} ({tag})"
    if ai_prediction:
        return f"{ai_prediction} (T5, unverified)"
    return "Ayrıştırılamadı"
COMMON_SOLVENTS = {
    "n-Heksan (Apolar)": 7.3,
    "Dietil Eter": 7.4,
    "Toluen (Aromatik)": 8.9,
    "Etil Asetat": 9.1,
    "Kloroform": 9.3,
    "Aseton (Polar Aprotik)": 9.9,
    "Diklorometan (DCM)": 9.7,
    "THF (Tetrahidrofuran)": 9.1,
    "Etanol (Alkol)": 12.7,
    "Metanol": 14.5,
    "Su (Çok Polar)": 23.4
}
COMMON_SOLVENTS_HANSEN = {
    f"{_('hekzan')}": 14.9,
    f"{_('dietil_eter')}": 14.8,
    f"{_('toluen')}": 18.7,
    f"{_('etil asetat')}": 17.9,
    f"{_('kloroform')}": 23.2,
    f"{_('aseton')}": 18.6,
    f"{_('diklorometan')}": 20.9,
    f"{_('THF')}": 18.4,
    f"{_('etanol')}": 25.7,
    f"{_('metanol')}": 28.2,
    f"{_('water')}": 29.7
}
def get_soluble_solvents(pred_val):
    """Tahmin edilen Hansen değerine göre uygun çözücüleri bulur."""
    soluble_list = []
    swelling_list = [] 
    
    for solvent, s_val in COMMON_SOLVENTS_HANSEN.items():
        diff = abs(pred_val - s_val)
        
        if diff <= 3.5: 
            soluble_list.append(solvent)
        elif diff <= 5: 
            swelling_list.append(solvent)
            
    return soluble_list, swelling_list
def draw_2d_molecule(smiles):
    """SMILES kodundan yüksek kaliteli 2D resim oluşturur."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            dopts = Draw.MolDrawOptions()
            dopts.addAtomIndices = False
            dopts.bondLineWidth = 2
            return Draw.MolToImage(mol, size=(500, 400), options=dopts)
    except:
        return None
def inject_custom_css():
    st.markdown("""
    <style>
        /* Ana Başlık Stili */
        .main-title {
            font-size: 3rem;
            color: #4A90E2;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
        }
        /* Alt Başlık */
        .sub-title {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        /* Kart Tasarımı (Sonuçlar için) */
        .metric-card {
            background-color: #f9f9f9;
            border-left: 5px solid #4A90E2;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
        /* Dark Mode Uyumu için Kart Rengi */
        @media (prefers-color-scheme: dark) {
            .metric-card {
                background-color: #262730;
                border-left: 5px solid #4A90E2;
                color: white;
            }
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()
N_BITS = 2048 
@st.cache_data
def get_initial_population():
    """Verisetini sadece bir kez indirir ve önbelleğe alır."""
    repo_id = "OsBaran/Polimer-Ozellik-Tahmini"
    tg_data = load_dataset(repo_id, split="Tg")
    df = tg_data.to_pandas()
    col_name = 'p_smiles' if 'p_smiles' in df.columns else 'smiles'
    raw_smiles = df[col_name].tolist()
    valid_selfies = []
    for s in raw_smiles:
        sf_str = smiles_to_selfies_safe(s)
        if sf_str:
            valid_selfies.append(sf_str)
    return valid_selfies, raw_smiles 
def _force_single_thread(model):
    """Set n_jobs/n_threads=1 on a model (and pipeline steps) to avoid pool-startup
    overhead on the GA's single-row predictions."""
    def _one(obj):
        try:
            if hasattr(obj, 'n_jobs'):
                obj.n_jobs = 1
        except Exception:
            pass
        try:
            if hasattr(obj, 'set_params'):
                # xgboost/lightgbm expose n_jobs; harmless if absent
                obj.set_params(n_jobs=1)
        except Exception:
            pass
    _one(model)
    for attr in ('steps', 'named_steps'):
        steps = getattr(model, attr, None)
        if steps:
            for st_ in (steps.values() if hasattr(steps, 'values') else [s for _, s in steps]):
                _one(st_)


@st.cache_resource
def load_critic_models():
    """Tüm Eleştirmen (Critic) modellerini yükler."""
    models = {}
    try:
        models['Tg'] = joblib.load('xgb_tg.joblib')
        models['Td'] = joblib.load('xgb_td.joblib')
        models['EPS'] = joblib.load('xgb_eps_v2.joblib')  # retrained on clean dataset6 (R2~0.69)
        models['Tm'] = joblib.load('xgb_tm.joblib')
        models['BandgapBulk'] = joblib.load('xgb_band gap bulk.joblib')
        models['BandgapChain'] = joblib.load('xgb_band gap chain.joblib')
        models['BandgapCrystal'] = joblib.load('xgb_bandgap-crystal.joblib')
        models['GasPerma'] = joblib.load('lgbm_gas_pipeline.joblib')
        models['Refractive'] = joblib.load('rf_refractive_index_v2.joblib')  # retrained: MolMR feats + cleaned data
        models['LOI'] = joblib.load('xgb_loi.joblib')
        models['Solubility'] = joblib.load('xgb_solubility.joblib') 
        models['ThermalCond'] = joblib.load('xgb_thermal_cond.joblib') 
        models['CTE'] = joblib.load('xgb_cte.joblib')
        models['Recyclability'] = joblib.load('lgbm_recyclability.joblib')
        models['Degradability'] = joblib.load('xgb_degradability.joblib')
        models['Hansen'] = joblib.load('xgb_hansen.joblib')

        # PERF: force single-thread prediction. The GA predicts one molecule at a
        # time; with n_jobs=-1 (baked into the RandomForests) every predict() spins
        # up a multiprocessing pool -> ~40s/run of pure pool-startup overhead.
        for _m in models.values():
            _force_single_thread(_m)
        return models
    except Exception as e:
        st.error(f"⚠️ Model Yükleme Hatası! Lütfen 'tg_model.joblib', 'td_model.joblib' ve 'eps_model.joblib' dosyalarının mevcut olduğundan emin olun. Hata: {e}")
        return None

def run_ga_silent(models, generations, targets, active_props, initial_pop, ranges_dict):
    """
    GA'yı grafik çizmeden (sessizce) çalıştırır. Çoklu testler için optimize edilmiştir.
    """
    toolbox = base.Toolbox()
    toolbox.register("attr_selfies", random.choice, initial_pop)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_selfies, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual_optimized, models=models, targets=targets, active_props=active_props, ranges=ranges_dict)
    toolbox.register("mate", cxSmart)
    toolbox.register("select", tools.selTournament, tournsize=7)

    prop_bias = sga.goal_bias_from_targets(active_props, targets, ranges_dict)

    pop_size = 100
    pop = toolbox.population(n=pop_size)


    best_fitness_history = []
    
    cxpb, mutpb, extendpb, newpb, chempb = 0.8, 0.05, 0.05, 0.01, 0.05

    for gen in range(generations):
        offspring = toolbox.select(pop, pop_size)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for i in range(len(offspring)):
            if not offspring[i].fitness.valid: pass
            offspring[i] = generate_offspring(
                offspring[i], initial_pop, prop_bias=prop_bias,
                mutpb=mutpb, chempb=chempb, fragpb=0.5, newpb=newpb,
                preserve=True, use_residual=True,
            )
            del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = evaluate_population_batch(invalid_ind, models, targets, active_props, ranges_dict, multi=False)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = offspring
        
        fits = [ind.fitness.values[0] for ind in pop]
        best_fitness_history.append(min(fits))

    return best_fitness_history
def run_mass_random_test(models, generations, initial_pop, ranges_dict, num_trials=100):
    """
    Rastgele hedeflerle 100 kez stres testi yapar.
    """
    results = []
    all_props_list = list(ranges_dict.keys())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_trials):
        n_active = random.randint(2, 5)
        active_props = random.sample(all_props_list, n_active)
        
        current_targets = {}
        target_descriptions = []
        for prop in active_props:
            r = ranges_dict[prop]
            val = random.uniform(r['min'], r['max'])
            
            if r.get('is_int', False) or prop in ['Tg', 'Td', 'Tm', 'LOI']:
                val = round(val, 0)
            else:
                val = round(val, 2)
                
            current_targets[prop] = val
            target_descriptions.append(f"{prop}={val}")

        history = run_ga_silent(models, generations, current_targets, active_props, initial_pop, ranges_dict)
        
        final_score = history[-1] 
        
        results.append({
            "Deneme No": i + 1,
            "Hedef Sayısı": n_active,
            "Hedefler": ", ".join(target_descriptions),
            "Final Hata Skoru": final_score
        })
        
        progress_bar.progress((i + 1) / num_trials)
        status_text.text(f"Test {i+1}/{num_trials} | Son Hata: {final_score:.4f} | Hedefler: {', '.join(target_descriptions)[:50]}...")
    
    status_text.success(f"{num_trials} Farklı Senaryo Testi Tamamlandı!")
    return pd.DataFrame(results)
def smiles_to_selfies_safe(smiles):
    if not smiles: return None
    clean_smi = smiles.replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
    try:
        selfies_string = sf.encoder(clean_smi)
        return selfies_string.replace('[H]', '[*]')
    except:
        return None

def selfies_to_smiles_safe(selfes_string):
    if not selfes_string: return None
    try:
        temp_selfies = selfes_string.replace('[*]', '[H]')
        smiles = sf.decoder(temp_selfies)
        return smiles.replace('[H]', '*')
    except:
        return None

def get_morgan_fp(p_smiles, keep_star=False):
    # Some models were trained KEEPING '*' as dummy atoms, others with '*'->[H].
    # Feeding the wrong convention silently degrades predictions (worst for small
    # repeat units). keep_star=True matches the keep-'*' models (see compute_preds).
    if keep_star:
        smi_clean = str(p_smiles)
    else:
        smi_clean = str(p_smiles).replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, N_BITS)
    return np.array([fp])


def get_gas_features_combined(smiles):
    """
    Gaz geçirgenliği LGBM modeli için Morgan FP.
    NOTE: model was trained KEEPING '*' (save_model.py) -> keep '*' here too.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles))   # keep '*' (matches training)
        if mol is None: return None

        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
        
        desc = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HallKierAlpha(mol)
        ])
        
        return np.concatenate((fp, desc)).reshape(1, -1)
    except:
        return None

# Sütun isimlerini dışarıda (global) bir kere tanımlıyoruz ki GA döngüsünde zaman kaybetmesin

def get_degradability_features(smiles):
    """Bozunabilirlik modeli için özellik çıkarımı. Model '*' KORUYARAK eğitildi."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))   # keep '*' (matches training; R2 0.27->0.90)
        if mol is None: return None
        
        desc_vals = [
            Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumRotatableBonds(mol)
        ]
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
        final_arr = np.concatenate([desc_vals, np.array(fp)])
        
        return final_arr.reshape(1, -1)
    except:
        return None

def get_recyclability_features(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None: return None

        n_total = mol.GetNumHeavyAtoms()
        inv_len_total = 1.0 / n_total if n_total > 0 else 0

        dummy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
        if len(dummy_indices) >= 2:
            try:
                path = Chem.GetShortestPath(mol, dummy_indices[0], dummy_indices[1])
                n_backbone = len(path) - 2
                inv_len_backbone = 1.0 / n_backbone if n_backbone > 0 else 0
                branching_ratio = n_total / n_backbone if n_backbone > 0 else 1
            except:
                inv_len_backbone = inv_len_total
                branching_ratio = 1.0
        else:
            inv_len_backbone = inv_len_total
            branching_ratio = 1.0

        num_rings = mol.GetRingInfo().NumRings()
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        num_aliphatic_rings = Descriptors.NumAliphaticRings(mol)
        mol_wt = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        fraction_csp3 = Descriptors.FractionCSP3(mol)

        desc_vals = [
            inv_len_total,
            inv_len_backbone,
            branching_ratio,
            num_rings,
            num_aromatic_rings,
            num_aliphatic_rings,
            mol_wt,
            tpsa,
            logp,
            fraction_csp3
        ]

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)

        final_arr = np.concatenate([desc_vals, np.array(fp)])
        
        return final_arr.reshape(1, -1)
    except:
        return None
def get_hansen_features(smiles):
    """Hansen Çözünürlük modeli için özellik çıkarımı (1030 Sütun). '*' KORUNUR."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))   # keep '*' (matches training; R2 0.58->0.99)
        if mol is None: return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        
        # 2. Eğitim defterindeki 6 ekstra descriptor (Sırası birebir aynı)
        desc_vals = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol)
        ]
        
        # 3. Önce FP, Sonra Descriptors olacak şekilde birleştir
        final_arr = np.concatenate([np.array(fp), desc_vals])

        # XGBoost'un beklediği 2D Numpy formatı
        return final_arr.reshape(1, -1)
    except:
        return None


def get_refractive_features(smiles):
    """
    Feature vector for the RETRAINED refractive-index model (rf_refractive_index_v2):
    radius-2 Morgan(2048) + 6 descriptors (MolMR is a near-physical proxy for RI),
    keeping '*' as dummy atoms -- exactly matching the training recipe. -> 2054 cols.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        desc = [
            Descriptors.MolMR(mol),             # Molar Refractivity (key)
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.HeavyAtomCount(mol),
        ]
        return np.array(fp + desc, dtype=float).reshape(1, -1)
    except Exception:
        return None


def get_eps_features(smiles):
    """
    Feature vector for the RETRAINED dielectric-constant model (xgb_eps_v2):
    radius-2 Morgan(2048, keep '*') + 8 polarity descriptors -> 2056 cols. Trained on
    the CLEAN dataset6 EPS data (dielectric 2.6-9.1); the old model used a corrupt split.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles))   # keep '*'
        if mol is None:
            return None
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        desc = [
            Descriptors.TPSA(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.MolMR(mol), Descriptors.MolWt(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.FractionCSP3(mol),
        ]
        return np.array(fp + desc, dtype=float).reshape(1, -1)
    except Exception:
        return None


# Per-property model-reliability tiers, from validation against the HF dataset with
# the CORRECT '*'-handling per model (see scratchpad validate_all/star_test/gasperma_eval).
# Fixing the '*' convention lifted Td/Tm/band gaps/GasPerma/Hansen/Degradability to 'high'.
#   high = trustworthy | medium = usable | low = ranking only | unreliable = do not trust
MODEL_RELIABILITY = {
    'LOI': 'high', 'Solubility': 'high', 'Tg': 'high',
    'BandgapBulk': 'high', 'BandgapChain': 'high', 'BandgapCrystal': 'high',
    'Td': 'high', 'Tm': 'high', 'GasPerma': 'high', 'Hansen': 'high', 'Degradability': 'high',
    'ThermalCond': 'medium',
    'EPS': 'medium',   # retrained on the CLEAN dataset6 EPS data (R2~0.69); old model used a corrupt split
    # Refractive v2 (MolMR + cleaned data) scores R2~0.82 on the dataset's held-out test,
    # but generalises poorly to tiny canonical repeat units (data-source issue) -> medium.
    'Refractive': 'medium',
    'CTE': 'low', 'Recyclability': 'low',
}
RELIABILITY_BADGE = {'high': '🟢', 'medium': '🟡', 'low': '🟠', 'unreliable': '🔴'}


def cxSelfies(ind1, ind2):
    t1 = list(sf.split_selfies(ind1[0]))
    t2 = list(sf.split_selfies(ind2[0]))
    min_len = min(len(t1), len(t2))
    if min_len < 2: return ind1, ind2

    split1 = random.randint(1, min_len-1)
    split2 = random.randint(1, min_len-1)
    
    new1 = t1[:split1] + t2[split2:]
    new2 = t2[:split2] + t1[split1:]

    new1_str = "".join(new1)
    new2_str = "".join(new2)
    
    if is_valid_polymer(new1_str):
        ind1[0] = new1_str
    if is_valid_polymer(new2_str):
        ind2[0] = new2_str
    return ind1, ind2

def mutSelfies(individual):
    tokens = list(sf.split_selfies(individual[0]))
    if not tokens: return individual,
    if random.random() < 0.6 and len(tokens) > 1:
        idx = random.randint(0, len(tokens) - 1)
        del tokens[idx]
    if random.random() < 0.4:
        idx = random.randint(0, len(tokens))
        new_token = random.choice(['[C]', '[N]', '[O]', '[F]', '[Cl]', '[S]', '[*]'])
        tokens.insert(idx, new_token)
    individual[0] = "".join(tokens)
    return individual,
FITNESS_CACHE = {}
# Per-individual "which target is most off, and which way" -> lets mutation focus
# on closing THIS molecule's biggest property gap (residual-guided mutation).
RESIDUAL_BIAS_CACHE = {}
# Cache of the full prediction dict per SELFIES (reused for reporting / NSGA-II).
PRED_CACHE = {}


def compute_preds(s_smiles, models, props):
    """
    Run every requested property model for one molecule and return {prop: value}.
    Each property uses its own feature extractor (see appv2 feature helpers).
    Returns None if the base fingerprint can't be built.
    """
    # Two Morgan fingerprints: replace-'*' (default) and keep-'*'. Each property's
    # model is fed the convention it was TRAINED on (validated empirically -- feeding
    # the wrong one silently cripples Td/Tm/band gaps etc.). See scratchpad star_test.
    fp = get_morgan_fp(s_smiles, keep_star=False)
    if fp is None:
        return None
    fp_keep = get_morgan_fp(s_smiles, keep_star=True)

    gas_features = get_gas_features_combined(s_smiles) if 'GasPerma' in props else None
    rec_features = get_recyclability_features(s_smiles) if 'Recyclability' in props else None
    deg_features = get_degradability_features(s_smiles) if 'Degradability' in props else None
    han_features = get_hansen_features(s_smiles) if 'Hansen' in props else None
    ref_features = get_refractive_features(s_smiles) if 'Refractive' in props else None
    eps_features = get_eps_features(s_smiles) if 'EPS' in props else None

    # fingerprint-only models trained KEEPING '*'
    KEEP_STAR_FP = {'Td', 'Tm', 'BandgapBulk', 'BandgapChain', 'BandgapCrystal'}

    preds = {}
    for prop in props:
        if prop not in models:
            continue
        if prop == 'GasPerma':
            preds[prop] = 10 ** models[prop].predict(gas_features)[0] if gas_features is not None else 0.0
        elif prop == 'Recyclability':
            preds[prop] = models[prop].predict(rec_features)[0] if rec_features is not None else 0.0
        elif prop == 'Degradability':
            preds[prop] = models[prop].predict(deg_features)[0] if deg_features is not None else 0.0
        elif prop == 'Hansen':
            preds[prop] = models[prop].predict(han_features)[0] if han_features is not None else 0.0
        elif prop == 'Refractive':
            preds[prop] = models[prop].predict(ref_features)[0] if ref_features is not None else 0.0
        elif prop == 'EPS':
            preds[prop] = models[prop].predict(eps_features)[0] if eps_features is not None else 0.0
        elif prop in KEEP_STAR_FP:
            preds[prop] = models[prop].predict(fp_keep)[0] if fp_keep is not None else models[prop].predict(fp)[0]
        else:
            preds[prop] = models[prop].predict(fp)[0]
    return preds


# Each property's feature extractor + how to route '*' (keep vs replace) + post-transform.
# fp variants are '_fpk' (keep '*') and '_fp' (replace '*'); the rest are custom.
_PROP_FEATURE = {
    'Tg': '_fp', 'LOI': '_fp', 'Solubility': '_fp', 'ThermalCond': '_fp', 'CTE': '_fp',
    'Td': '_fpk', 'Tm': '_fpk', 'BandgapBulk': '_fpk', 'BandgapChain': '_fpk', 'BandgapCrystal': '_fpk',
    'GasPerma': 'gas', 'EPS': 'eps', 'Refractive': 'ref', 'Hansen': 'han',
    'Degradability': 'deg', 'Recyclability': 'rec',
}
_PROP_POST = {'GasPerma': lambda v: 10 ** v}   # model predicts log10(Barrer)


def compute_preds_batch(smiles_list, models, active_props):
    """
    Vectorised prediction for a whole list of molecules at once. Returns a list of
    prediction dicts aligned to smiles_list (None where SMILES is invalid).
    Batching is the key GA speedup: model.predict on a 100-row matrix costs ~the same
    as on 1 row, so this replaces ~100 predict() calls per generation with ~1 per model.
    """
    n = len(smiles_list)
    preds = [({} if s is not None else None) for s in smiles_list]

    # extractor per feature-key; each returns a flat vector or None
    def _ex(key, s):
        if key == '_fp':
            f = get_morgan_fp(s, keep_star=False)
        elif key == '_fpk':
            f = get_morgan_fp(s, keep_star=True)
        elif key == 'gas':
            f = get_gas_features_combined(s)
        elif key == 'eps':
            f = get_eps_features(s)
        elif key == 'ref':
            f = get_refractive_features(s)
        elif key == 'han':
            f = get_hansen_features(s)
        elif key == 'deg':
            f = get_degradability_features(s)
        elif key == 'rec':
            f = get_recyclability_features(s)
        else:
            return None
        return None if f is None else np.asarray(f).reshape(-1)

    # which feature-keys do we actually need?
    keys_needed = {_PROP_FEATURE[p] for p in active_props if p in _PROP_FEATURE and p in models}
    # compute each feature matrix once (shared across properties using the same key)
    feat_rows = {}   # key -> (list_of_row_indices, matrix)
    for key in keys_needed:
        rows, mats = [], []
        for i, s in enumerate(smiles_list):
            if s is None:
                continue
            v = _ex(key, s)
            if v is not None:
                rows.append(i); mats.append(v)
        feat_rows[key] = (rows, np.vstack(mats) if mats else None)

    for prop in active_props:
        if prop not in models or prop not in _PROP_FEATURE:
            continue
        key = _PROP_FEATURE[prop]
        rows, X = feat_rows.get(key, ([], None))
        yp = models[prop].predict(X) if X is not None else []
        post = _PROP_POST.get(prop)
        valid_rows = set(rows)
        for j, i in enumerate(rows):
            preds[i][prop] = post(yp[j]) if post else yp[j]
        for i in range(n):
            if preds[i] is not None and i not in valid_rows:
                preds[i][prop] = 0.0
    return preds


def evaluate_population_batch(individuals, models, targets, active_props, ranges, multi=False):
    """
    Batched fitness for a list of DEAP individuals (each [selfies]). Honours the
    caches, then predicts the whole batch at once. Returns a list of fitness tuples
    (single-objective weighted error, or the NSGA-II objective vector when multi=True).
    """
    n_obj = len(active_props) + 1
    fits = [None] * len(individuals)

    # cache lookups first
    todo = []
    for i, ind in enumerate(individuals):
        s = ind[0]
        if not multi and s in FITNESS_CACHE:
            fits[i] = FITNESS_CACHE[s]
        else:
            todo.append(i)
    if not todo:
        return fits

    selfies = [individuals[i][0] for i in todo]
    smis = [selfies_to_smiles_safe(s) for s in selfies]
    preds_list = compute_preds_batch(smis, models, active_props)

    for k, i in enumerate(todo):
        s, smi, preds = selfies[k], smis[k], preds_list[k]
        if smi is None or preds is None:
            fits[i] = tuple([1000.0] * n_obj) if multi else (1000.0,)
            if not multi:
                FITNESS_CACHE[s] = fits[i]
            continue
        PRED_CACHE[s] = preds

        worst_prop, worst_err, worst_dir = None, -1.0, None
        objectives, total_error = [], 0.0
        for prop in active_props:
            if prop in preds:
                ne = abs(preds[prop] - targets[prop]) / (ranges[prop]['max'] - ranges[prop]['min'])
            else:
                ne = 1000.0
            objectives.append(float(ne))
            total_error += np.exp(ne * 10) - 1
            if prop in preds and ne > worst_err:
                worst_err, worst_prop = ne, prop
                worst_dir = 'high' if preds[prop] < targets[prop] else 'low'
        if worst_prop is not None:
            RESIDUAL_BIAS_CACHE[s] = {worst_prop: worst_dir}

        sa = get_sa_score_local(smi)
        if multi:
            objectives.append(sa / 10.0)
            fits[i] = tuple(objectives)
        else:
            fits[i] = (total_error + sa * 2.0,)
            FITNESS_CACHE[s] = fits[i]
    return fits


def evaluate_individual_optimized(individual, models, targets, active_props, ranges):
    s_selfies = individual[0]

    if s_selfies in FITNESS_CACHE:
        return FITNESS_CACHE[s_selfies]

    s_smiles = selfies_to_smiles_safe(s_selfies)
    if s_smiles is None: return (1000.0,)

    preds = compute_preds(s_smiles, models, active_props)
    if preds is None: return (1000.0,)
    PRED_CACHE[s_selfies] = preds

    total_error = 0.0
    if not active_props: return (1000.0,)

    worst_prop, worst_err, worst_dir = None, -1.0, None
    for prop in active_props:
        if prop in preds:
            norm_error = abs(preds[prop] - targets[prop]) / (ranges[prop]['max'] - ranges[prop]['min'])
            total_error += np.exp(norm_error * 10) - 1
            if norm_error > worst_err:
                worst_err = norm_error
                worst_prop = prop
                # 'high' => need to INCREASE this property, 'low' => DECREASE it
                worst_dir = 'high' if preds[prop] < targets[prop] else 'low'

    # Remember the biggest gap so the next mutation can target it specifically.
    if worst_prop is not None:
        RESIDUAL_BIAS_CACHE[s_selfies] = {worst_prop: worst_dir}

    sa_score = get_sa_score_local(s_smiles)
    total_error += sa_score * 2.0

    result = (total_error,)
    FITNESS_CACHE[s_selfies] = result
    return result


def evaluate_individual_multi(individual, models, targets, active_props, ranges):
    """
    Multi-objective evaluation for NSGA-II. Returns a tuple of objectives to be
    MINIMISED: one normalised error per active property, plus a synthesizability
    objective (SA/10). This lets NSGA-II find the Pareto front of trade-offs
    instead of collapsing everything into one weighted number.
    """
    n_obj = len(active_props) + 1
    big = tuple([1000.0] * n_obj)

    s_selfies = individual[0]
    s_smiles = selfies_to_smiles_safe(s_selfies)
    if s_smiles is None:
        return big
    preds = compute_preds(s_smiles, models, active_props)
    if preds is None:
        return big
    PRED_CACHE[s_selfies] = preds

    objectives = []
    worst_prop, worst_err, worst_dir = None, -1.0, None
    for prop in active_props:
        if prop in preds:
            norm_error = abs(preds[prop] - targets[prop]) / (ranges[prop]['max'] - ranges[prop]['min'])
            if norm_error > worst_err:
                worst_err = norm_error
                worst_prop = prop
                worst_dir = 'high' if preds[prop] < targets[prop] else 'low'
        else:
            norm_error = 1000.0
        objectives.append(float(norm_error))

    if worst_prop is not None:
        RESIDUAL_BIAS_CACHE[s_selfies] = {worst_prop: worst_dir}

    objectives.append(get_sa_score_local(s_smiles) / 10.0)  # synthesizability objective
    return tuple(objectives)


def run_random_benchmark(models, targets, active_props, initial_pop, ranges_dict, total_budget, batch_size=100):
    """
    GA ile adil kıyaslama için Rastgele Arama (Random Search) yapar.
    total_budget: Toplam değerlendirme sayısı (GA'daki pop_size * generations)
    batch_size: Grafik çizimi için her kaç adımda bir kayıt alınacağı (GA'daki pop_size kadar olmalı)
    """
    history_random = []
    best_so_far = float('inf')
    
    progress_text = st.empty()
    bar = st.progress(0)
    
    for i in range(0, total_budget, batch_size):
        candidates = random.sample(initial_pop, batch_size) 
        
        scores = []
        for ind_selfies in candidates:
            fit = evaluate_individual_optimized([ind_selfies], models, targets, active_props, ranges_dict)
            
            scores.append(fit[0])
        
        current_batch_best = min(scores)
        
        if current_batch_best < best_so_far:
            best_so_far = current_batch_best
            
        history_random.append(best_so_far)
        
        progress = (i + batch_size) / total_budget
        if progress > 1.0: progress = 1.0
        bar.progress(progress)
        progress_text.text(f"Rastgele Arama: {i}/{total_budget} tamamlandı. En iyi skor: {best_so_far:.4f}")
        
    bar.empty()
    progress_text.empty()
    return history_random

def evaluate_individual_single_obj(individual, models, targets, active_props):
    """
    Seçilen hedeflere (active_props) olan toplam mesafeye (hata) göre değerlendirir.

    """
    s_selfies = individual[0]

    s_smiles = selfies_to_smiles_safe(s_selfies)
    if s_smiles is None:

        return (1000.0,)
    fp = get_morgan_fp(s_smiles)
    if fp is None:
        return (1000.0,)
    preds = {}

    for prop in active_props:
        if prop in models:
             preds[prop] = models[prop].predict(fp)[0]
    total_error = 0.0
    if not active_props:
        return (1000.0,)
    for prop in active_props:
        if prop in preds:
            norm_error = abs(preds[prop] - targets[prop]) / (ranges[prop]['max'] - ranges[prop]['min'])
            total_error += np.exp(norm_error * 10) - 1  

    if total_error == 0.0 and len(active_props) > 0:

         return (1000.0,)
    
    total_error += get_sa_score_local(s_smiles) / 10.0 
         

    return (total_error,)

if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) 
    creator.create("Individual", list, fitness=creator.FitnessMin)

def is_valid_polymer(selfies_str):
    """
    Hem kimyasal geçerliliği hem de polimer olma şartını (bağlantı noktaları) kontrol eder.
    """
    smiles = selfies_to_smiles_safe(selfies_str)
    if smiles is None: 
        return False

    star_count = smiles.count('*')
    if star_count < 2:
        return False  

    
    clean_smi = smiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(clean_smi)
    
    if mol is None:
        return False 
        
    if mol.GetNumHeavyAtoms() < 4:
        return False

    return True


MUTATION_TOKENS = ['[C]', '[N]', '[O]', '[F]', '[Cl]', '[S]', '[*]', 'c', 'n', 'o']
def mutSelfies(individual, max_attempts=5):
    tokens = list(sf.split_selfies(individual[0]))
    if not tokens: 
        return individual

    for _ in range(max_attempts):
        temp_tokens = tokens.copy()
        # Token silme
        if random.random() < 0.3 and len(temp_tokens) > 1:
            idx = random.randint(0, len(temp_tokens) - 1)
            del temp_tokens[idx]
        # Token ekleme
        if random.random() < 0.3:
            idx = random.randint(0, len(temp_tokens))
            new_token = random.choice(MUTATION_TOKENS)
            temp_tokens.insert(idx, new_token)
        # Token değiştirme
        if random.random() < 0.3:
            idx = random.randint(0, len(temp_tokens) - 1)
            temp_tokens[idx] = random.choice(MUTATION_TOKENS)
        
        candidate = "".join(temp_tokens)
        if is_valid_polymer(candidate):
            individual[0] = candidate
            return individual
    
    individual[0] = random.choice(initial_selfies)
    return individual


def extendPolymer(individual, max_add=3):
    tokens = list(sf.split_selfies(individual[0]))
    for _ in range(random.randint(1, max_add)):
        tokens.append(random.choice(['[C]', '[N]', '[O]', '[F]', '[Cl]', '[S]']))
    candidate = "".join(tokens)
    return candidate if is_valid_polymer(candidate) else individual[0]

REACTION_SMARTS = [
    "[C:1][H:2]>>[C:1]Cl",
    "[C:1][H:2]>>[C:1]O",
    "[C:1](=O)[O;H1].[O;H1][C:2]>>[C:1](=O)O[C:2]",
    "[C:1](=O)Cl.[N:2]>>[C:1](=O)N",
    "[O:1][H].[C:2]Br>>[O:1][C:2]",
    "c1ccccc1>>c1([N+](=O)[O-])ccccc1",
    "[C:1]=[C:2]>>[C:1]-[C:2]"
]

RDKit_REACTIONS = [rdChemReactions.ReactionFromSmarts(s) for s in REACTION_SMARTS]

def chemically_valid_mutate(p_smi: str, reactions=RDKit_REACTIONS, attempts=6):
    """Reaction tabanlı mutasyon uygular; başarısızsa fallback döner."""
    def sanitize_and_canonicalize(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            rdmolops.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None

    def replace_star_with_H(smi: str):
        return str(smi).replace('*', '[H]')

    def restore_H_to_star(smi: str):
        return str(smi).replace('[H]', '*')

    def is_reasonable_product(prod_smiles, max_atoms=120, min_atoms=4):
        if prod_smiles is None: return False
        try:
            m = Chem.MolFromSmiles(prod_smiles)
            if m is None: return False
            n = m.GetNumAtoms()
            if n > max_atoms or n < min_atoms: return False
            try: rdmolops.SanitizeMol(m)
            except: return False
            return True
        except: return False

    base = replace_star_with_H(p_smi)
    base_mol = Chem.MolFromSmiles(base)
    if base_mol is None: return p_smi

    candidate_products = []
    for _ in range(attempts):
        rxn = random.choice(reactions)
        try:
            ps = rxn.RunReactants((base_mol,))
        except:
            ps = ()
        for prod_tuple in ps:
            for prod_mol in prod_tuple:
                try:
                    prod_smiles = Chem.MolToSmiles(prod_mol, canonical=True)
                except: prod_smiles = None
                prod_restored = restore_H_to_star(prod_smiles) if prod_smiles else None
                if is_reasonable_product(prod_restored):
                    candidate_products.append(prod_restored)

    if candidate_products:
        out = random.choice(candidate_products)
        if out == p_smi or len(out) < max(4, len(p_smi)//2):
            return p_smi
        return out
    return p_smi

mutation_stats = {'SELFIES':0, 'REACTION':0, 'EXTEND':0, 'NEW':0}

def generate_offspring(individual, initial_selfies, prop_bias=None,
                       mutpb=0.05, chempb=0.05, removepb=0.15, fragpb=0.5,
                       newpb=0.02, preserve=True, use_residual=True, **kwargs):
    """
    Chemistry-aware mutation.

    Instead of blindly editing SELFIES atom-by-atom, this:
      * adds whole chemically-valid GROUPS / backbone spacers chosen to push the
        target properties in the right direction (prop_bias / residual bias);
      * only removes danglng leaf atoms (never ring / functional-group atoms);
      * enforces a STRUCTURE-PRESERVATION guard so a mutation can never remove an
        important motif -- any ring, any aromatic system, or a functional group
        (ester, amide, urethane, imide, sulfone, nitrile ...). Not just benzene;
      * on failure keeps the PARENT instead of reseeding randomly.

    prop_bias: {property: 'low'|'high'|'mid'} global target direction.
    use_residual: if True, prefer this individual's own biggest property gap
                  (RESIDUAL_BIAS_CACHE) so it mutates toward what it most lacks.
    preserve: if True, apply the important-motif guard.
    kwargs absorbs legacy params (e.g. extendpb).
    """
    # Rare full reseed keeps a little global diversity.
    if random.random() < newpb:
        individual[0] = random.choice(initial_selfies)
        mutation_stats['NEW'] += 1
        return individual

    parent_selfies = individual[0]
    parent_smi = selfies_to_smiles_safe(parent_selfies)

    # If the parent can't be read as a valid polymer, fall back to the old
    # SELFIES mutation / reseed path.
    if not parent_smi or not sga.is_valid_polymer_smiles(parent_smi):
        individual = mutSelfies(individual)
        if not is_valid_polymer(individual[0]):
            individual[0] = random.choice(initial_selfies)
        return individual

    # Choose the mutation direction: this molecule's own worst gap if we have it,
    # otherwise the global target direction.
    bias = prop_bias
    if use_residual:
        residual = RESIDUAL_BIAS_CACHE.get(parent_selfies)
        if residual:
            bias = residual

    new_smi = parent_smi

    # 1. Heuristic fragment / backbone-group edit (primary driver).
    if random.random() < fragpb:
        cand = sga.directional_fragment_mutate(new_smi, bias)
        if cand and sga.is_valid_polymer_smiles(cand):
            new_smi = cand
            mutation_stats['SELFIES'] += 1  # reuse existing counter for "fragment" edits

    # 2. Reaction-based valid mutation (existing SMARTS engine), low rate.
    if random.random() < chempb:
        cand = chemically_valid_mutate(new_smi)
        if cand and sga.is_valid_polymer_smiles(cand):
            new_smi = cand
            mutation_stats['REACTION'] += 1

    # 3. Structure-preserving pendant removal (trims danglng groups only).
    if random.random() < removepb:
        cand = sga.remove_leaf_group(new_smi)
        if cand and sga.is_valid_polymer_smiles(cand):
            new_smi = cand
            mutation_stats['EXTEND'] += 1  # reuse existing counter for "trim" edits

    # STRUCTURE GUARD: reject a candidate that dropped any important motif.
    if preserve and not sga.preserves_important(parent_smi, new_smi):
        new_smi = parent_smi

    # Commit only if the result round-trips to a valid SELFIES polymer AND the
    # SELFIES encode/decode step didn't silently drop an important motif (it can,
    # to force validity); otherwise keep the parent untouched.
    new_selfies = smiles_to_selfies_safe(new_smi)
    if new_selfies and is_valid_polymer(new_selfies):
        committed_smi = selfies_to_smiles_safe(new_selfies)
        ok = committed_smi is not None
        if ok and preserve:
            ok = sga.preserves_important(parent_smi, committed_smi)
        individual[0] = new_selfies if ok else parent_selfies
    else:
        individual[0] = parent_selfies
    return individual


def cxSmart(ind1, ind2):
    """
    Structure-preserving crossover: swap whole acyclic substituents between the
    two parents while keeping each backbone (and its rings) intact. Falls back to
    the SELFIES token crossover when no substituent can be exchanged.
    """
    s1 = selfies_to_smiles_safe(ind1[0])
    s2 = selfies_to_smiles_safe(ind2[0])
    if s1 and s2:
        res = sga.smart_crossover(s1, s2)
        if res:
            c1, c2 = res
            if c1 and sga.is_valid_polymer_smiles(c1):
                sf1 = smiles_to_selfies_safe(c1)
                if sf1 and is_valid_polymer(sf1):
                    ind1[0] = sf1
            if c2 and sga.is_valid_polymer_smiles(c2):
                sf2 = smiles_to_selfies_safe(c2)
                if sf2 and is_valid_polymer(sf2):
                    ind2[0] = sf2
            return ind1, ind2
    return cxSelfies(ind1, ind2)


def run_single_objective_flow(models, generations, targets, active_props, initial_pop, ranges_dict,
                              heuristic=True, preserve=True):
    toolbox = base.Toolbox()
    toolbox.register("attr_selfies", random.choice, initial_pop)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_selfies, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual_optimized, models=models, targets=targets, active_props=active_props, ranges=ranges_dict)

    toolbox.register("mate", cxSmart)
    toolbox.register("select", tools.selTournament, tournsize=7)

    # Heuristic direction for each active target (low / high / mid) -> guides mutation.
    # Disabled (None) when the user turns heuristic guidance off for an A/B run.
    prop_bias = sga.goal_bias_from_targets(active_props, targets, ranges_dict) if heuristic else None

    pop_size = 100
    pop = toolbox.population(n=pop_size)
    
    history = {
        "gen": [],
        "best_fitness": [],
        "avg_fitness": [],
        "diversity": [] 
    }

    fitnesses = evaluate_population_batch(pop, models, targets, active_props, ranges_dict, multi=False)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    st.markdown(f"{_('evolution_panel')}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.caption(f"{_('convergence_chart')}")
        chart_fitness_placeholder = st.empty()
    with col_chart2:
        st.caption(f"{_('diversity_chart')}")
        chart_diversity_placeholder = st.empty()

    log_expander = st.expander("GA Logs", expanded=False)
    with log_expander:
        log_placeholder = st.empty()
        mutation_placeholder = st.empty()

    log_data = [] 

    for gen in range(generations):
        scale = gen / generations
        cxpb = max(0.6, 0.8 - (0.2 * scale)) # 80den 60a düşen çaprazlama olasılığı
        mutpb = max(0.02, 0.1 - (0.08 * scale)) # 10dan 2ye düşen SELFIES mutasyon olasılığı
        chempb = max(0.01, 0.05 - (0.04 * scale)) # 5ten 1e düşen reaction mutasyon olasılığı
        extendpb = max(0.01, 0.05 - (0.04 * scale)) # 5ten 1e düşen zincir uzatma olasılığı
        newpb = 0.02 # Sabit kalan rastgele yeni birey ekleme olasılığı


        offspring = toolbox.select(pop, pop_size)
        offspring = list(map(toolbox.clone, offspring))

        # Çaprazlama
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for i in range(len(offspring)):
            if not offspring[i].fitness.valid:
                 pass
            offspring[i] = generate_offspring(
                offspring[i], initial_pop, prop_bias=prop_bias,
                mutpb=mutpb, chempb=chempb, removepb=0.15,
                fragpb=max(0.4, 0.7 - 0.3 * scale), newpb=newpb,
                preserve=preserve, use_residual=heuristic,
            )
            del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = evaluate_population_batch(invalid_ind, models, targets, active_props, ranges_dict, multi=False)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = offspring
        
        fits = [ind.fitness.values[0] for ind in pop]
        
        valid_fits = [f for f in fits if f < 999.0]
        
        if valid_fits:
            best_val = min(valid_fits) 
            mean_val = sum(valid_fits) / len(valid_fits) 
            std_val = np.std(valid_fits) 
        else:
            best_val = 1000.0
            mean_val = 1000.0
            std_val = 0.0

        survival_rate = (len(valid_fits) / len(pop)) * 100
        
        history["gen"].append(gen)
        history["best_fitness"].append(best_val)
        history["avg_fitness"].append(mean_val)
        history["diversity"].append(std_val)
        
        log_data.append({
            f"{_('gen')}": gen + 1,
            f"{_('best_fitness')}": round(best_val, 4),
            f"{_('avg_fitness')}": round(mean_val, 4),
            f"{_('survival_rate')}": round(survival_rate, 1)
        })

        if gen % 2 == 0 or gen == generations - 1:
            progress_bar.progress((gen + 1) / generations)
            status_text.markdown(f"**Nesil {gen+1}/{generations}** | {_('best_fitness')}: `{best_val:.4f}` | {_('diversity_chart')}: `{std_val:.4f}`")
            
            df_fit = pd.DataFrame({
                f"{_('best_fitness')}": history["best_fitness"],
                f"{_('avg_fitness')}": history["avg_fitness"]
            })
            chart_fitness_placeholder.line_chart(df_fit, height=250)
            
            df_div = pd.DataFrame({
                f"{_('diversity_chart')}": history["diversity"]
            })
            chart_diversity_placeholder.line_chart(df_div, height=250)
            
            df_log = pd.DataFrame(log_data)
            log_placeholder.dataframe(df_log.sort_values(by=f"{_('gen')}", ascending=False).head(5), width='stretch')
            mutation_placeholder.json(mutation_stats)

    best_ind = tools.selBest(pop, 5)[0]
    best_smiles = selfies_to_smiles_safe(best_ind[0])
    if best_smiles:
        preds = compute_preds(best_smiles, models, list(models.keys())) or {}
        return {'smiles': best_smiles, 'preds': preds, 'total_error': best_ind.fitness.values[0]}, history
    else:
        return None, history


def pick_knee_index(objective_rows):
    """
    Index of the most BALANCED Pareto solution: the point closest to the ideal
    (utopia) point in min-max-normalised objective space. This is the classic
    'knee' heuristic -- it avoids solutions that ace one target while sacrificing
    another, and works for any number of objectives.
    """
    if not objective_rows:
        return None
    M = np.asarray(objective_rows, dtype=float)
    lo = M.min(axis=0)
    hi = M.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    normed = (M - lo) / span          # each objective scaled to [0, 1] across the front
    dist = np.linalg.norm(normed, axis=1)  # distance to the normalised ideal (origin)
    return int(np.argmin(dist))


def run_nsga2_flow(models, generations, targets, active_props, initial_pop, ranges_dict,
                   heuristic=True, preserve=True):
    """
    Multi-objective (NSGA-II) optimisation.

    Each active property is its own objective (normalised error to minimise) plus a
    synthesizability objective. NSGA-II keeps a Pareto front of trade-off solutions
    instead of one weighted winner. Returns (best_poly_data, history, pareto_front)
    where best_poly_data has the same shape the single-objective flow returns, so the
    results UI is unchanged; pareto_front is a list of {smiles, preds, obj} dicts.
    """
    n_obj = len(active_props) + 1

    # Rebuild the multi-objective Individual with the right number of weights each run
    # (the objective count changes with how many properties the user selected).
    for cls in ("FitnessMulti", "IndividualMulti"):
        if cls in creator.__dict__:
            del creator.__dict__[cls]
    creator.create("FitnessMulti", base.Fitness, weights=tuple([-1.0] * n_obj))
    creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_selfies", random.choice, initial_pop)
    toolbox.register("individual", tools.initRepeat, creator.IndividualMulti, toolbox.attr_selfies, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual_multi, models=models, targets=targets,
                     active_props=active_props, ranges=ranges_dict)
    toolbox.register("mate", cxSmart)
    toolbox.register("select", tools.selNSGA2)

    prop_bias = sga.goal_bias_from_targets(active_props, targets, ranges_dict) if heuristic else None

    pop_size = 100  # multiple of 4 (required by selTournamentDCD)
    pop = toolbox.population(n=pop_size)
    for ind, fit in zip(pop, evaluate_population_batch(pop, models, targets, active_props, ranges_dict, multi=True)):
        ind.fitness.values = fit
    pop = toolbox.select(pop, pop_size)  # assigns crowding distance

    n_prop = len(active_props)

    def prop_error(ind):
        """Sum of the property-error objectives (excludes the SA objective)."""
        return sum(ind.fitness.values[:n_prop])

    history = {"gen": [], "best_fitness": [], "avg_fitness": [], "diversity": []}

    st.markdown(f"{_('evolution_panel')}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.caption(f"{_('convergence_chart')}")
        chart_fitness_placeholder = st.empty()
    with col_chart2:
        st.caption(f"{_('diversity_chart')}")
        chart_diversity_placeholder = st.empty()

    for gen in range(generations):
        scale = gen / generations
        cxpb = max(0.6, 0.8 - 0.2 * scale)
        mutpb = max(0.02, 0.1 - 0.08 * scale)
        chempb = max(0.01, 0.05 - 0.04 * scale)
        newpb = 0.02

        # NSGA-II parent selection by dominance + crowding distance.
        offspring = tools.selTournamentDCD(pop, pop_size)
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for i in range(len(offspring)):
            offspring[i] = generate_offspring(
                offspring[i], initial_pop, prop_bias=prop_bias,
                mutpb=mutpb, chempb=chempb, removepb=0.15,
                fragpb=max(0.4, 0.7 - 0.3 * scale), newpb=newpb,
                preserve=preserve, use_residual=heuristic,
            )
            del offspring[i].fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, evaluate_population_batch(invalid, models, targets, active_props, ranges_dict, multi=True)):
            ind.fitness.values = fit

        # mu+lambda environmental selection -> next generation.
        pop = toolbox.select(pop + offspring, pop_size)

        errs = [prop_error(ind) for ind in pop]
        valid = [e for e in errs if e < 999.0]
        if valid:
            best_val, mean_val, std_val = min(valid), sum(valid) / len(valid), float(np.std(valid))
        else:
            best_val, mean_val, std_val = 1000.0, 1000.0, 0.0

        history["gen"].append(gen)
        history["best_fitness"].append(best_val)
        history["avg_fitness"].append(mean_val)
        history["diversity"].append(std_val)

        if gen % 2 == 0 or gen == generations - 1:
            progress_bar.progress((gen + 1) / generations)
            front_size = len(tools.sortNondominated(pop, len(pop), first_front_only=True)[0])
            status_text.markdown(
                f"**Nesil {gen+1}/{generations}** | {_('best_fitness')}: `{best_val:.4f}` | "
                f"Pareto: `{front_size}` | {_('diversity_chart')}: `{std_val:.4f}`")
            chart_fitness_placeholder.line_chart(pd.DataFrame({
                f"{_('best_fitness')}": history["best_fitness"],
                f"{_('avg_fitness')}": history["avg_fitness"],
            }), height=250)
            chart_diversity_placeholder.line_chart(pd.DataFrame({
                f"{_('diversity_chart')}": history["diversity"],
            }), height=250)

    # Pareto front of the final population.
    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    seen, pareto = set(), []
    for ind in front:
        smi = selfies_to_smiles_safe(ind[0])
        if not smi or smi in seen:
            continue
        seen.add(smi)
        pareto.append({
            "smiles": smi,
            "preds": PRED_CACHE.get(ind[0], {}),
            "obj": list(ind.fitness.values),
            "prop_error": prop_error(ind),
        })
    pareto.sort(key=lambda d: d["prop_error"])
    for sol in pareto:
        sol["tag"] = ""

    if not pareto:
        return None, history, pareto

    # Tag the two reference solutions on the front:
    #   min-error = lowest total target error (may be lopsided across properties)
    #   knee      = most balanced trade-off (closest to the ideal point)
    pareto[0]["tag"] = "min-error"
    knee_idx = pick_knee_index([s["obj"][:n_prop] for s in pareto])
    if knee_idx is not None:
        pareto[knee_idx]["tag"] = ("knee + min-error" if knee_idx == 0 else "knee")

    # Headline "best" = the balanced knee-point solution.
    chosen = pareto[knee_idx] if knee_idx is not None else pareto[0]
    preds = compute_preds(chosen["smiles"], models, list(models.keys())) or {}
    best_poly_data = {
        'smiles': chosen["smiles"], 'preds': preds,
        'total_error': chosen["prop_error"], 'is_knee': True,
    }
    return best_poly_data, history, pareto


@st.cache_data
def check_pubchem_availability(smiles: str):
    """
    Verilen SMILES için PubChem'de kayıtlı mı kontrol eder.
    Yıldızları (*) temizleyerek arama yapar.
    """
    clean_smi = smiles.replace('*', '') 
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{clean_smi}/cids/JSON"
    
    try:
        response = requests.get(url, timeout=5)
        
        if response.status_code == 404:
            return False, None, None
            
        response.raise_for_status()
        data = response.json()
        
        if "IdentifierList" in data and "CID" in data["IdentifierList"]:
            cid = data["IdentifierList"]["CID"][0]
            
            name_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON"
            name_resp = requests.get(name_url, timeout=5)
            if name_resp.status_code == 200:
                name_data = name_resp.json()
                name = name_data["PropertyTable"]["Properties"][0].get("IUPACName", "Bilinmiyor")
            else:
                name = "Bilinmiyor"
                
            return True, cid, name
        else:
            return False, None, None
            
    except Exception:
        return False, None, None
def check_commercial_availability(query):
    """
    Verilen ismi veya SMILES'ı PubChem'de arar.
    Ticari olarak satılıp satılmadığını (Vendor sayısı) kontrol eder.
    """
    try:
        compounds = pcp.get_compounds(query, 'name')
        if not compounds:
            compounds = pcp.get_compounds(query, 'smiles')
            
        if compounds:
            cid = compounds[0].cid
            synonyms = compounds[0].synonyms
            common_name = synonyms[0] if synonyms else query
            return True, cid, common_name
        else:
            return False, None, None
    except:
        return False, None, None
def make_3d_view_with_reason(smiles):
    try:
        clean_smi = str(smiles).replace('*', '[H]')
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is None:
            return None, "SMILES geçersiz veya RDKit ile molekül oluşturulamadı."
        
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) != 0:
            return None, "3D koordinatlar hesaplanamadı (Embed başarısız)."
        
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            return None, "3D yapı enerji optimizasyonunda başarısız."
        
        mblock = Chem.MolToMolBlock(mol)
        view = py3Dmol.view(width=400, height=400)
        view.addModel(mblock, 'mol')
        view.setStyle({'stick':{'colorscheme':'Jmol'}})
        view.zoomTo()
        view.spin(True)
        return view, None
    except Exception as e:
        return None, f"Beklenmeyen bir hata: {e}"

def get_ai_interpretation(api_key, smiles, preds, targets, active_props):
    """Gemini API kullanarak polimer analizi yapar."""
    if not api_key:
        return "⚠️ Analiz için lütfen sol menüden geçerli bir Google Gemini API Anahtarı giriniz."

    try:
        import google.generativeai as genai   # lazy: only needed if the LLM feature is used
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Sen uzman bir Polimer Kimyagerisin ve Malzeme Bilimci'sin. 
        Aşağıda genetik algoritma ile üretilmiş yeni bir polimer adayı var.
        
        Molekül (SMILES): {smiles}
        
        Tahmin Edilen Özellikler:
        """
        
        for prop in active_props:
            target_val = targets.get(prop, "Belirtilmedi")
            pred_val = preds.get(prop, 0.0)
            prompt += f"- {prop}: Tahmin={pred_val:.2f} (Hedef={target_val})\n"
            
        prompt += """
        
        Lütfen bu polimeri şu başlıklar altında Türkçe olarak detaylıca analiz et:
        1. **Yapı-Özellik İlişkisi:** Bu yapısal özellikler (halkalar, fonksiyonel gruplar, zincir uzunluğu vb.) neden bu tahmin değerlerini (özellikle Tg ve Td) ortaya çıkarmış olabilir? Kimyasal mantığı nedir?
        2. **Potansiyel Uygulama Alanları:** Bu özelliklere sahip bir polimer endüstride nerede kullanılabilir? (Örn: Havacılık, paketleme, elektronik, membran vb.)
        3. **Sentezlenebilirlik Yorumu:** Yapıya bakarak sentez zorluğu veya stabilite hakkında kısa bir yorum yap.
        
        Yanıtın profesyonel, bilimsel ama anlaşılır olsun. Markdown formatı kullan.
        """
        
        with st.spinner('Yapay Zeka polimeri inceliyor...'):
            response = model.generate_content(prompt)
            return response.text
            
    except Exception as e:
        return f"❌ AI Bağlantı Hatası: {str(e)}"

def get_sa_score_local(p_smiles):
    """
    Eğer klasörde 'sascorer.py' varsa onu kullanır, yoksa hesaplama yapar.
    """
    try:
        import sascorer
        smi_clean = str(p_smiles).replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
        mol = Chem.MolFromSmiles(smi_clean)
        if mol is None:
            raise ValueError("Mol oluşturulamadı")
        score = sascorer.calculateScore(mol)
        if score is None:                 # sascorer returns None for empty/degenerate mols
            raise ValueError("SA score None")
        return float(score)
    except:
        length = len(str(p_smiles))
        score = 2.0 + (length * 0.05)
        if "c1" in str(p_smiles): 
            score += 0.5
        return min(score, 10.0)

def calculate_green_score(smiles, deg_val=None, reg_val=None):
    """
    Polimerin potansiyel biyo-bozunurluğunu ve çevresel etkisini puanlar.
    Puan: 1 (Çok Kötü/Kalıcı) - 10 (Mükemmel/Bozunabilir)
    """
    mol = Chem.MolFromSmiles(smiles.replace('*', '[H]'))
    if not mol: return 0, "Hesaplanamadı", "#7f8c8d"
    
    score = 5.0 # Nötr 
    notes = []
    
    # Ester 
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C;!R](=[O])[O;!R]")):
        score += 3.0
        notes.append(f"{_('ester')}")
        
    # Amid 
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C;!R](=[O])[N;!R]")):
        score += 2.0
        notes.append(f"{_('amide')}")
        
    # Eter
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C][O][C]")):
        score += 1.0
        notes.append(f"{_('ether')}")

    # Halojen
    halogens = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br']]
    if halogens:
        count = len(halogens)
        penalty = min(4.0, count * 1.0) 
        score -= penalty
        notes.append(f"{count} {_('halogen')}")
        
    # Aromatik
    aromatic_atoms = [atom for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    if len(aromatic_atoms) > 4: 
        score -= 2.0
        notes.append(f"{_('aromatic')}")

    rule_score = max(1.0, min(10.0, score)) 

    final_score = rule_score
    if deg_val is not None and reg_val is not None:
        
        # Deg Skoru Hesaplama (0 - 10 arası)
        if deg_val <= 0.716:
            deg_score = (deg_val / 0.716) * 9.0
        else:
            deg_score = 10.0
            
        # Reg Skoru Hesaplama (0 - 10 arası)
        if -20 <= reg_val <= -10:
            reg_score = 10.0
        elif -60 <= reg_val < -20:
            # -60'dan -20'ye doğru 0'dan 10'a çıkar
            reg_score = ((reg_val + 60) / 40.0) * 10.0
        elif -10 < reg_val <= 40:
            # -10'dan 40'a doğru 10'dan 0'a düşer
            reg_score = 10.0 - ((reg_val + 10) / 50.0) * 10.0
        else:
            reg_score = 0.0 # Sınır dışı acil durum
            
        # Her ikisini de sınırların içinde tuttuğumuzdan emin olalım
        deg_score = max(0.0, min(10.0, deg_score))
        reg_score = max(0.0, min(10.0, reg_score))
        
        # Ağırlıklı Toplam (%34 Yapısal, %33 Deg, %33 Reg)
        final_score = (rule_score * 0.34) + (deg_score * 0.33) + (reg_score * 0.33)
        
        # Kullanıcı arayüzü için notlara değerleri ekle
        notes.append(f"Deg: {deg_val:.3f}")
        notes.append(f"Reg: {reg_val:.1f}")

    # Nihai skoru 1-10 aralığında sınırla
    final_score = max(1.0, min(10.0, final_score))
    
    if final_score >= 7.0: color = "#2ecc71" 
    elif final_score >= 4.0: color = "#f1c40f" 
    else: color = "#e74c3c" 
    
    return round(final_score, 1), ", ".join(notes), color

def create_radar_chart(preds, targets, active_props, ranges):
    """
    Hedeflenen özellikler ile tahmin edilen özellikleri karşılaştıran
    havalı bir Radar (Spider) Grafiği çizer.
    """
    categories = []
    target_values = []
    pred_values = []
    
    for prop in active_props:
        if prop in preds and prop in targets:
            label = prop
            if prop == 'ThermalCond': label = 'Iletkenlik'
            if prop == 'Solubility': label = 'Cozunurluk'
            
            categories.append(label)
            
            t_val = targets[prop]
            p_val = preds[prop]
            
            min_v = ranges[prop]['min']
            max_v = ranges[prop]['max']
            
            if max_v - min_v == 0: denom = 1
            else: denom = max_v - min_v
            
            norm_t = (t_val - min_v) / denom
            norm_p = (p_val - min_v) / denom
            
            norm_t = max(0.0, min(1.0, norm_t))
            norm_p = max(0.0, min(1.0, norm_p))
            
            target_values.append(norm_t)
            pred_values.append(norm_p)
            
    categories = categories + [categories[0]]
    target_values = target_values + [target_values[0]]
    pred_values = pred_values + [pred_values[0]]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=target_values,
        theta=categories,
        fill='toself',
        name=f'{_("Hedeflenen")}',
        line=dict(color='#3498db', dash='dash')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=pred_values,
        theta=categories,
        fill='toself',
        name=f'{_("Uretilen Polimer")}',
        line=dict(color='#e74c3c'),
        opacity=0.7
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] 
            )),
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=20),
        height=300 
    )
    
    return fig
def decompose_polymer(smiles):
    """
    Polimeri parçalar. v3.0: Üre ve Üretan bağlarını da tanır.
    """
    clean_smi = smiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(clean_smi)
    if not mol: return None, "Geçersiz Molekül"
    
    breakdown_results = []
    
    # İMİD
    imide_pattern = Chem.MolFromSmarts("[CX3](=[OX1])[#7][CX3](=[OX1])")
    if mol.HasSubstructMatch(imide_pattern):
        return [{
            "type": f"{_('Poliimid Sentezi')}",
            "reaction": f"{_('Siklo-dehidrasyon')}",
            "monomers": [f"{_('Dianhidrit')}", f"{_('Diamin')}"],
            "mechanism": f"{_('Dianhidrit + Diamin -> Poliimid')}"
        }]

    # ÜRE 
    urea_pattern = Chem.MolFromSmarts("[N;!R][C;!R](=[O])[N;!R]")
    if mol.HasSubstructMatch(urea_pattern):
        breakdown_results.append({
            "type": f"{_('Poliüre (Polyurea) Sentezi')}",
            "reaction": f"{_('Basamaklı Polimerizasyon (Hızlı')}",
            "monomers": [f"{_('Diizosiyanat (Diisocyanate)')}", f"{_('Diamin (Diamine)')}"],
            "mechanism": f"{_('İzosiyanat + Amin -> Üre Bağı (Yan ürün yok)')}"
        })

    # ÜRETAN 
    urethane_pattern = Chem.MolFromSmarts("[N;!R][C;!R](=[O])[O;!R]")
    if mol.HasSubstructMatch(urethane_pattern):
        breakdown_results.append({
            "type": f"{_('Poliüretan (PU) Sentezi')}",
            "reaction": f"{_('Poliladisyon')}",
            "monomers": [f"{_('Diizosiyanat (Örn: TDI, MDI)')}", f"{_('Diol / Polyol')}"],
            "mechanism": f"{_('İzosiyanat + Alkol -> Üretan Bağı')}"
        })

    # ESTER 
    ester_pattern = Chem.MolFromSmarts("[C;!R](=[O])[O;!R]") 
    if mol.HasSubstructMatch(ester_pattern) and not breakdown_results:
        breakdown_results.append({
            "type": f"{_('Polyester Sentezi')}",
            "reaction": f"{_('Kademeli Polimerizasyon')}",
            "monomers": [f"{_('Dikarboksilik Asit')}", f"{_('Diol')}"],
            "mechanism": f"{_('Asit + Alkol -> Ester + Su')}"
        })

    # AMİD 
    amide_pattern = Chem.MolFromSmarts("[C;!R](=[O])[N;!R]")
    if mol.HasSubstructMatch(amide_pattern) and not breakdown_results: 
         breakdown_results.append({
            "type": f"{_('Poliamid (Nylon) Sentezi')}",
            "reaction": f"{_('Polikondenzasyon')}",
            "monomers": [f"{_('Dikarboksilik Asit')}", f"{_('Diamin')}"],
            "mechanism": f"{_('Asit + Amin -> Amid + Su')}"
        })

    #VARSAYILAN 
    if not breakdown_results:
        has_hetero = any(atom.GetSymbol() in ['N', 'O', 'S'] for atom in mol.GetAtoms())
        if has_hetero and "C=C" not in smiles:
             breakdown_results.append({
                "type": f"{_('Kompleks Kondenzasyon Polimeri')}",
                "reaction": f"{_('Özel Sentez (AI Analizi Önerilir)')}",
                "monomers": [f"{_('Fonksiyonel Grup A')}", f"{_('Fonksiyonel Grup B')}"],
                "mechanism": f"{_('Uç grupların reaksiyonu')}"
            })
        else:
            breakdown_results.append({
                "type": f"{_('Vinil Polimerizasyonu (Katılma)')}",
                "reaction": f"{_('Radikalik')}",
                "monomers": [smiles.replace('*', '')],
                "mechanism": f"{_('Çift bağ açılması')}"
            })
            
    return breakdown_results
def draw_retrosynthesis_grid(monomer_smiles_list):
    """Monomerlerin listesini alır ve yan yana çizer."""
    mols = [Chem.MolFromSmiles(s) for s in monomer_smiles_list]
    mols = [m for m in mols if m is not None] 
    if not mols: return None
    
    img = Draw.MolsToGridImage(
        mols, 
        molsPerRow=min(len(mols), 3), 
        subImgSize=(200, 200),
        legends=[f"Monomer {i+1}" for i in range(len(mols))]
    )
    return img

def get_ai_retrosynthesis_guide(api_key, polymer_smiles, monomer_info):
    """Gemini'den detaylı sentez rotası ister."""
    if not api_key: return "⚠️ Detaylı sentez planı için API Key gerekli."

    try:
        import google.generativeai as genai   # lazy: only needed if the LLM feature is used
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        Sen uzman bir Sentetik Polimer Kimyagerisin.
        Aşağıdaki polimer için endüstriyel veya laboratuvar ölçekli bir RETROSENTEZ (geriye dönük sentez) planı hazırla.
        
        Hedef Polimer (SMILES): {polymer_smiles}
        Algarlanan Olası Yöntem: {monomer_info}
        
        Lütfen şu formatta yanıtla:
        1. **Önerilen Monomerler:** Bu yapıyı oluşturmak için hangi ticari kimyasallar (IUPAC isimleri) gerekir?
        2. **Sentez Yöntemi:** Hangi reaksiyon türü uygundur? (Örn: Radikalik, Kondenzasyon, ROMP?)
        3. **Kritik Koşullar:** Sıcaklık, basınç veya spesifik katalizör (AIBN, Ziegler-Natta, H2SO4 vb.) önerisi.
        4. **Zorluk Analizi:** Bu sentezin pratik zorlukları nelerdir?
        
        Kısa, net ve bilimsel olsun.
        """
        with st.spinner('AI Sentez Rotasını Hesaplıyor...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"Hata: {str(e)}"

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Polsen - Ar-Ge Proje Raporu', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

def clean_text(text):
    """FPDF için Türkçe karakterleri ASCII'ye çevirir (Hızlı çözüm)"""
    replacements = {
        'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I', 'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf_report(poly_data, targets, active_props, ai_analysis_text, retro_info):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("1. Genel Degerlendirme ve Skorlar"), 0, 1)
    pdf.set_font("Arial", size=10)

    # Toplam Hata
    pdf.cell(50, 8, clean_text("Toplam Hata:"), 0, 0)
    pdf.cell(0, 8, f"{poly_data['total_error']:.4f}", 0, 1)

    # Sentez Zorluğu (SA Score)
    sa_score = get_sa_score_local(poly_data['smiles'])
    pdf.cell(50, 8, clean_text("Sentez Zorlugu (SA):"), 0, 0)
    pdf.cell(0, 8, f"{sa_score:.2f} / 10", 0, 1)

    # Yeşil Skor
    preds = poly_data['preds']
    de_val = preds.get('Degradability')
    reg_val = preds.get('Recyclability')
    g_score, g_note, _color = calculate_green_score(poly_data['smiles'], deg_val=de_val, reg_val=reg_val)
    pdf.cell(50, 8, clean_text("Yesil Skor:"), 0, 0)
    pdf.cell(0, 8, f"{g_score:.1f} / 10 ({clean_text(g_note)})", 0, 1)

    # Çözünürlük Analizi
    if 'Hansen' in preds:
        sol_val = preds['Hansen']
        solvents, partials = get_soluble_solvents(sol_val)
        pdf.cell(50, 8, clean_text("Hansen Cozunurluk:"), 0, 0)
        pdf.cell(0, 8, f"{sol_val:.2f}", 0, 1)
        if solvents:
            pdf.cell(50, 8, clean_text("Tam Cozundugu:"), 0, 0)
            pdf.multi_cell(0, 8, clean_text(", ".join(solvents)))
        if partials:
            pdf.cell(50, 8, clean_text("Sistigi/Kismen:"), 0, 0)
            pdf.multi_cell(0, 8, clean_text(", ".join(partials)))

    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("1. Polimer Özellik Tablosu"), 0, 1)
    pdf.set_font("Arial", size=10)
    
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(60, 8, "Ozellik", 1, 0, 'C', 1)
    pdf.cell(60, 8, "Hedef", 1, 0, 'C', 1)
    pdf.cell(60, 8, "Tahmin Degeri", 1, 1, 'C', 1)
    
    all_preds = poly_data['preds']
    
    for prop, val in all_preds.items():
        if prop in active_props:
            target_val = str(targets.get(prop, '-'))
        else:
            target_val = "-" 
        pred_val = f"{val:.2f}"
        
        pdf.cell(60, 8, clean_text(prop), 1)
        pdf.cell(60, 8, target_val, 1)
        pdf.cell(60, 8, pred_val, 1, 1)
    
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("2. Molekuler Yapi"), 0, 1)
    
    pdf.set_font("Courier", size=8)
    pdf.multi_cell(0, 5, poly_data['smiles'])
    pdf.ln(5)
    
    mol_img = draw_2d_molecule(poly_data['smiles'])
    if mol_img:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            mol_img.save(tmp_file.name)
            pdf.image(tmp_file.name, x=60, w=90)
    
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("3. Uretim Plani (Retrosentez)"), 0, 1)
    pdf.set_font("Arial", size=10)
    
    if not retro_info or len(retro_info) < 5:
        pdf.multi_cell(0, 6, clean_text("Retrosentez analizi yapilmadi veya veri yok."))
    else:
        clean_retro = clean_text(str(retro_info))
        pdf.multi_cell(0, 6, clean_retro)
    
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("4. Yapay Zeka Uzman Görüşü"), 0, 1)
    pdf.set_font("Arial", size=10)
    
    if not ai_analysis_text or len(ai_analysis_text) < 5:
        pdf.multi_cell(0, 6, clean_text("AI analizi talep edilmedi."))
    else:
        clean_ai = clean_text(ai_analysis_text).replace('**', '').replace('#', '')
        pdf.multi_cell(0, 6, clean_ai)
    
    return pdf.output(dest='S').encode('latin-1')

@st.cache_data
def get_reference_fingerprints(smiles_list):
    """
    Referans veri setindeki tüm SMILES'ların parmak izlerini önceden hesaplar ve önbelleğe alır.
    Bu işlem sadece bir kez yapılır, böylece uygulama hızlanır.
    """
    fps = []
    names = [] 
    
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(str(smi).replace('*', '[H]'))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
                fps.append(fp)
                names.append(f"Veri Seti Kaydı #{i+1}") 
        except:
            continue
    return fps, names

def calculate_novelty_optimized(generated_smiles, ref_smiles_list):
    """
    Toplu Tanimoto benzerliği hesaplar.
    """
    gen_mol = Chem.MolFromSmiles(generated_smiles.replace('*', '[H]'))
    if not gen_mol: return 0.0, "Hesaplanamadı"
    gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 3, 2048)
    
    ref_fps, ref_names = get_reference_fingerprints(ref_smiles_list)
    
    if not ref_fps: return 0.0, "Veri Seti Boş"
    
    sims = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
    
    max_sim = max(sims)
    max_idx = sims.index(max_sim)
    most_similar_name = ref_names[max_idx]
    most_similar_smiles = ref_smiles_list[max_idx] if max_idx < len(ref_smiles_list) else "Bilinmiyor"
    
    return max_sim, most_similar_smiles

st.markdown(f'<h1 class="main-title"> POLSEN <br><span style="font-size:1.5rem; color:#666; font-weight:400;">{_("app_subtitle")}</span></h1>', unsafe_allow_html=True)

models = load_critic_models()
ALL_PROPS = list(models.keys())

# --- Manual polymer analysis (predict properties + retrosynthesis for a user SMILES) ---
if models and SHOW_MANUAL_ANALYSIS:
    with st.expander(f"🧪 {_('manual_title')}", expanded=False):
        st.caption(f"{_('manual_desc')}")
        mcol_in, mcol_btn = st.columns([4, 1])
        with mcol_in:
            manual_smiles = st.text_input(f"{_('manual_input_label')}", key="manual_smiles_input",
                                          placeholder="*CC(c1ccccc1)*")
        with mcol_btn:
            st.write(""); st.write("")
            run_manual = st.button(f"{_('manual_btn')}", use_container_width=True, key="manual_run_btn")

        if run_manual and manual_smiles:
            _mmol = Chem.MolFromSmiles(str(manual_smiles).replace('*', '[H]'))
            if _mmol is None:
                st.error(f"❌ {_('manual_invalid')}")
            else:
                m_preds = compute_preds(manual_smiles, models, ALL_PROPS) or {}
                mc_struct, mc_props = st.columns([1, 2])
                with mc_struct:
                    _img = draw_2d_molecule(manual_smiles)
                    if _img is not None:
                        st.image(_img, use_container_width=True)
                    st.metric(f"{_('metric_sa_score')}", f"{get_sa_score_local(manual_smiles):.2f} / 10")
                with mc_props:
                    if SHOW_RELIABILITY:
                        st.caption(f"{_('reliability_legend')}")
                    _pcols = st.columns(3)
                    for _i, _p in enumerate(ALL_PROPS):
                        _badge = ""
                        if SHOW_RELIABILITY:
                            _tier = MODEL_RELIABILITY.get(_p, 'medium')
                            _badge = f'<span title="{_(f"reliability_{_tier}")}">{RELIABILITY_BADGE.get(_tier, "")}</span>'
                        with _pcols[_i % 3]:
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 5px solid #9b59b6;">
                                <small>{_p} {_badge}</small><br>
                                <h3 style="margin:0; padding:0;">{m_preds.get(_p, 0.0):.2f}</h3>
                            </div>""", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"**{_('retro_analysis_header')}**")
                _routes = retro.retro_decompose(manual_smiles)
                if _routes:
                    _r = _routes[0]
                    _vf = "🔬" if _r.get('verified', False) else "⚠️"
                    st.info(f"{_vf} **{_('yontem')}:** {_r['type']}  |  {_r['mechanism']}")
                    _img_r = draw_retrosynthesis_grid(_r['monomers'])
                    if _img_r is not None:
                        st.image(_img_r)
                    for _i, _m in enumerate(_r['monomers']):
                        st.code(f"Monomer {_i+1}: {_m}")
                else:
                    st.warning(f"{_('retro_auto_failed')}")

# --- MANUEL POLİMER TAHMİN BÖLÜMÜ BAŞLANGICI ---
# st.divider()
# with st.expander("🧪 Manuel Polimer SMILES Analizi", expanded=False):
#     st.markdown("Genetik algoritmayı çalıştırmadan, kendi polimerinizin özelliklerini hızlıca tahmin edin.")
    
#     col_input, col_btn = st.columns([4, 1])
#     with col_input:
#         manual_smiles = st.text_input("Polimer SMILES Kodu Giriniz (Örn: *CC(*) veya *CC(c1ccccc1)*):", key="manual_smiles_input")
#     with col_btn:
#         st.write("") # Düğmeyi hizalamak için boşluk
#         st.write("")
#         run_manual = st.button("Tahmin Et", type="primary", use_container_width=True)

#     if run_manual and manual_smiles:
#         # Molekülün geçerliliğini kontrol et
#         mol = Chem.MolFromSmiles(manual_smiles.replace('*', '[H]'))
#         if mol is None:
#             st.error("❌ Geçersiz SMILES Kodu! Lütfen kimyasal sözdizimini kontrol ediniz.")
#         else:
#             with st.spinner("Yapay Zeka Modelleri özellikleri hesaplıyor..."):
#                 # 1. Özellik Çıkarımı (Feature Extraction)
#                 fp = get_morgan_fp(manual_smiles)
#                 gas_features = get_gas_features_combined(manual_smiles)
#                 rec_features = get_recyclability_features(manual_smiles)
#                 deg_features = get_degradability_features(manual_smiles)
#                 han_features = get_hansen_features(manual_smiles)

#                 # 2. Modellerden Tahmin Alma (Prediction)
#                 manual_preds = {}
#                 for prop in ALL_PROPS:
#                     if prop == 'GasPerma' and gas_features is not None:
#                         manual_preds[prop] = 10 ** models[prop].predict(gas_features)[0]
#                     elif prop == 'Recyclability' and rec_features is not None:
#                         manual_preds[prop] = models[prop].predict(rec_features)[0]
#                     elif prop == 'Degradability' and deg_features is not None:
#                         manual_preds[prop] = models[prop].predict(deg_features)[0]
#                     elif prop == 'Hansen' and han_features is not None:
#                         manual_preds[prop] = models[prop].predict(han_features)[0]
#                     elif fp is not None:
#                         manual_preds[prop] = models[prop].predict(fp)[0]
#                     else:
#                         manual_preds[prop] = 0.0
                
#                 st.success("✅ Tahmin başarıyla tamamlandı!")

#                 # 3. Sonuçları Ekrana Çizdirme
#                 m_col1, m_col2 = st.columns([1, 2])
                
#                 with m_col1:
#                     st.markdown("#### 2D Molekül Yapısı")
#                     img = draw_2d_molecule(manual_smiles)
#                     if img:
#                         st.image(img, use_container_width=True)
                    
#                     # Sentez Zorluğu ve Yeşil Skor
#                     sa_score = get_sa_score_local(manual_smiles)
#                     de_val = manual_preds.get('Degradability')
#                     reg_val = manual_preds.get('Recyclability')
#                     g_score, g_note, g_color = calculate_green_score(manual_smiles, deg_val=de_val, reg_val=reg_val)
                    
#                     st.metric("Sentez Zorluğu (SA Score)", f"{sa_score:.2f} / 10")
#                     st.markdown(f"""
#                     <div style="background-color:{g_color}20; border: 1px solid {g_color}; border-radius: 5px; padding: 5px; text-align: center; margin-top: 10px;">
#                         <strong style="color:{g_color}; font-size: 0.8rem;">🌱 Yeşil Skor (Çevresel Etki)</strong><br>
#                         <span style="font-size: 1.5rem; font-weight: bold; color:{g_color};">{g_score:.1f}/10</span>
#                     </div>
#                     """, unsafe_allow_html=True)

#                 with m_col2:
#                     st.markdown("#### 📊 Makine Öğrenmesi Tahminleri")
#                     pred_cols = st.columns(3)
                    
#                     # Uygulamanızın kendi CSS Metric Card yapısını kullanarak özellikleri diziyoruz
#                     for idx, prop in enumerate(ALL_PROPS):
#                         pred_value = manual_preds.get(prop, 0.0)
#                         with pred_cols[idx % 3]:
#                             st.markdown(f"""
#                             <div class="metric-card" style="border-left: 5px solid #9b59b6;">
#                                 <small style="color: #FFF;">{prop}</small><br>
#                                 <h3 style="margin:0; padding:0; color: #FFF;">{pred_value:.2f}</h3>
#                             </div>
#                             """, unsafe_allow_html=True)
# --- MANUEL POLİMER TAHMİN BÖLÜMÜ BİTİŞİ ---

def add_synced_input(prop_key, label, min_val, max_val, default, step, is_int=False):
    """Sidebar üzerinde bir slider ve number_input oluşturur; ikisini session_state üzerinden senkronlar.
    Döndürülen değer her zaman current value (float/int) olur.
    """
    s_key = f"{prop_key}_val"
    slider_key = f"{prop_key}_slider"
    num_key = f"{prop_key}_num"

    if s_key not in st.session_state:
        st.session_state[s_key] = default
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[s_key]
    if num_key not in st.session_state:
        st.session_state[num_key] = st.session_state[s_key]

    def _on_slider_change():
        try:
            st.session_state[num_key] = st.session_state[slider_key]
            st.session_state[s_key] = st.session_state[slider_key]
        except Exception:
            pass

    def _on_num_change():
        try:
            st.session_state[slider_key] = st.session_state[num_key]
            st.session_state[s_key] = st.session_state[num_key]
        except Exception:
            pass

    if is_int:
        st.sidebar.slider(label + " (slider)", min_value=int(min_val), max_value=int(max_val), step=int(step), key=slider_key, on_change=_on_slider_change)
        st.sidebar.number_input(label + " (value)", min_value=int(min_val), max_value=int(max_val), step=int(step), key=num_key, on_change=_on_num_change)
    else:
        st.sidebar.slider(label + " (slider)", min_value=float(min_val), max_value=float(max_val), step=float(step), key=slider_key, on_change=_on_slider_change)
        st.sidebar.number_input(label + " (value)", min_value=float(min_val), max_value=float(max_val), step=float(step), format="%.4f", key=num_key, on_change=_on_num_change)

    return st.session_state[s_key]

if models:
    st.sidebar.header(f'{_("sidebar_target_selection")}')
    
    active_props = []
    
    st.sidebar.markdown(f'### {_("sidebar_included_props")}')
    if st.sidebar.checkbox(f'{_("prop_tg")}', value=True):
        active_props.append('Tg')
    if st.sidebar.checkbox(f'{_("prop_td")}'):
        active_props.append('Td')
    if st.sidebar.checkbox(f'{_("prop_eps")}'):
        active_props.append('EPS')
    if st.sidebar.checkbox(f'{_("prop_tm")}'):
        active_props.append('Tm')
    if st.sidebar.checkbox(f'{_("prop_bandgap_bulk")}'):
        active_props.append('BandgapBulk')
    if st.sidebar.checkbox(f'{_("prop_bandgap_chain")}'):
        active_props.append('BandgapChain')
    if st.sidebar.checkbox(f'{_("prop_bandgap_crystal")}'):
        active_props.append('BandgapCrystal')
    if st.sidebar.checkbox(f'{_("prop_gas_perma")}'):
        active_props.append('GasPerma')
    if st.sidebar.checkbox(f'{_("prop_refractive")}'):
        active_props.append('Refractive')
    if st.sidebar.checkbox(f'{_("prop_loi")}'): 
        active_props.append('LOI')
    if st.sidebar.checkbox(f'{_("prop_solubility")}'): 
        active_props.append('Solubility') 
    if st.sidebar.checkbox(f'{_("prop_thermal_cond")}'): 
        active_props.append('ThermalCond')  
    if st.sidebar.checkbox(f'{_("prop_cte")}'): 
        active_props.append('CTE')
    if st.sidebar.checkbox(f'{_("prop_recyclability")}'):
        active_props.append('Recyclability')
    if st.sidebar.checkbox(f'{_("prop_degradability")}'):
        active_props.append('Degradability')
    if st.sidebar.checkbox(f'{_("prop_hansen")}'):
        active_props.append('Hansen')
    if not active_props:
        st.sidebar.warning(_("warn_no_target"))
        st.stop()

    # Warn if the user is optimising toward a property whose model is unreliable.
    _weak = [p for p in active_props if MODEL_RELIABILITY.get(p) == 'unreliable'] if SHOW_RELIABILITY else []
    if _weak:
        st.sidebar.warning(f"🔴 {_('warn_unreliable')} {', '.join(_weak)}")

    st.sidebar.markdown(f'### {_("sidebar_target_values")}')
    targets = {}
    ranges = {
        'Tg': {'min': -150.0, 'max': 300.0, 'default': 200.0, 'step': 1.0, 'is_int': False},
        'Td': {'min': 150.0, 'max': 600.0, 'default': 350.0, 'step': 1.0, 'is_int': False},
        'Tm': {'min': 50.0, 'max': 450.0, 'default': 250.0, 'step': 1.0, 'is_int': False},
        'EPS': {'min': 1.5, 'max': 12.0, 'default': 2.5, 'step': 0.1, 'is_int': False},
        'BandgapBulk': {'min': 0.5, 'max': 6.0, 'default': 2.5, 'step': 0.01, 'is_int': False},
        'BandgapChain': {'min': 0.5, 'max': 6.0, 'default': 2.5, 'step': 0.01, 'is_int': False},
        'BandgapCrystal': {'min': 0.5, 'max': 7.0, 'default': 2.5, 'step': 0.01, 'is_int': False},
        'GasPerma': {'min': 0.0, 'max': 1000.0, 'default': 2.5, 'step': 0.1, 'is_int': False},
        'Refractive': {'min': 1.2, 'max': 1.8, 'default': 1.5, 'step': 0.01, 'is_int': False},
        'LOI': {'min': 15.0, 'max': 100.0, 'default': 28.0, 'step': 0.5, 'is_int': False},
        'Solubility': {'min': 5.0, 'max': 20.0, 'default': 9.5, 'step': 0.1, 'is_int': False},
        'ThermalCond': {'min': 0.0, 'max': 1.0, 'default': 0.2, 'step': 0.01, 'is_int': False},
        'CTE': {'min': 0.0, 'max': 300.0, 'default': 60.0, 'step': 5.0, 'is_int': False},
        'Recyclability': {'min': -60.0, 'max': 40.0, 'default': -15.0, 'step': 1.0, 'is_int': False},
        'Degradability': {'min': 0.0, 'max': 2.0, 'default': 0.5, 'step': 0.01, 'is_int': False},
        'Hansen': {'min': 10.0, 'max': 50.0, 'default': 20.0, 'step': 0.1, 'is_int': False},
    }

    for prop in active_props:
        if prop in ranges:
            r = ranges[prop]
            label = prop
            if prop == 'Tg': label = f'{_("target")} Tg (°C)'
            elif prop == 'Td': label = f'{_("target")} Td (°C)'
            elif prop == 'Tm': label = f'{_("target")} Tm (°C)'
            elif prop == 'EPS': label = f'{_("target")} EPS'
            elif prop == 'BandgapBulk': label = f'{_("target")} BandgapBulk (eV)'
            elif prop == 'BandgapChain': label = f'{_("target")} BandgapChain (eV)'
            elif prop == 'BandgapCrystal': label = f'{_("target")} BandgapCrystal (eV)'
            elif prop == 'GasPerma': label = f'{_("target")} GasPerma'
            elif prop == 'Refractive': label = f'{_("target")} Refractive Index'

            val = add_synced_input(prop, label, r['min'], r['max'], r['default'], r['step'], is_int=r['is_int'])
            targets[prop] = val
        else:
            targets[prop] = st.sidebar.number_input(f'{_("target")} {prop}:', value=0.0)

    # 3. GA Parametreleri
    generations = st.sidebar.slider(f'{_("sidebar_generations")}', 10, 300, 10)

    # GA strategy toggles (let the user A/B the heuristic engine vs. a blind run).
    st.sidebar.markdown(f'### {_("sidebar_strategy")}')
    opt_mode = st.sidebar.radio(
        f'{_("opt_mode")}', ["nsga2", "weighted"],
        index=0, format_func=lambda k: _(f"opt_{k}"), help=f'{_("opt_mode_help")}')
    use_heuristic = st.sidebar.checkbox(f'{_("use_heuristic")}', value=True,
                                        help=f'{_("use_heuristic_help")}')
    preserve_structure = st.sidebar.checkbox(f'{_("preserve_structure")}', value=True,
                                             help=f'{_("preserve_structure_help")}')
    seed_val = st.sidebar.number_input(f'{_("random_seed")}', min_value=0, max_value=999999,
                                       value=0, step=1, help=f'{_("random_seed_help")}')

    initial_selfies, reference_smiles = get_initial_population()
    st.sidebar.divider()
    # st.sidebar.markdown(f'{_("sidebar_llm_settings")}')
    # api_key = st.sidebar.text_input("API Key", type="password", help=f"{_('api_key_explanation')}")
    if st.sidebar.button(f'{_("sidebar_btn_search")}', type="primary"):

        if not initial_selfies:
            st.error(_("warn_empty_pop"))
            st.stop()

        # Reset per-run caches so a new search is not skewed by the previous one.
        FITNESS_CACHE.clear()
        RESIDUAL_BIAS_CACHE.clear()
        PRED_CACHE.clear()

        # Reproducible runs when a non-zero seed is given (GA uses global random + numpy).
        if seed_val:
            random.seed(int(seed_val))
            np.random.seed(int(seed_val))

        with st.spinner(f'{_("msg_optimizing")} {", ".join(active_props)}'):
            if use_heuristic:
                # Seed the initial population with base polymers that match the goal
                # (e.g. flexible backbones for a low-Tg target) instead of starting blind.
                seeded_pop = sga.build_seed_population(active_props, targets, ranges, initial_selfies)
            else:
                # Blind start: plain dataset population, no goal-directed seeding.
                seeded_pop = initial_selfies

            pareto = None
            if opt_mode == "nsga2":
                best_poly_data, history, pareto = run_nsga2_flow(
                    models, generations, targets, active_props, seeded_pop, ranges,
                    heuristic=use_heuristic, preserve=preserve_structure,
                )
            else:
                best_poly_data, history = run_single_objective_flow(
                    models, generations, targets, active_props, seeded_pop, ranges,
                    heuristic=use_heuristic, preserve=preserve_structure,
                )

        if best_poly_data:
            st.session_state['ga_results'] = best_poly_data
            st.session_state['ga_history'] = history
            st.session_state['ga_targets'] = targets
            st.session_state['ga_active_props'] = active_props
            st.session_state['ga_pareto'] = pareto
            
    
    if 'ga_results' in st.session_state:
        
        best_poly_data = st.session_state['ga_results']
        history = st.session_state['ga_history']
        saved_targets = st.session_state['ga_targets']
        saved_active_props = st.session_state['ga_active_props']
        
        preds = best_poly_data['preds']
        
        st.success(f"{_('msg_success')}")
        
        tab1, tab2, tab3, tab4, tab6 = st.tabs([f"{_('tab_general')}", f"{_('tab_structural')}", f"{_('tab_evolution')}", f"{_('tab_report')}", f"{_('tab_retro')}"])
        with tab1:
            col_main, col_score, col_green = st.columns([2, 1, 1])
            
            with col_main:
                st.markdown(f"### {_('metric_total_error')} **{best_poly_data['total_error']:.4f}**")
                
            with col_score:
                sa = get_sa_score_local(best_poly_data['smiles'])
                st.metric(f"{_('metric_sa_score')}", f"{sa:.2f}", help=f"1 ({_('easy')}) - 10 ({_('hard')})")
                
            with col_green:
                de_val, reg_val = None, None
                if 'Degradability' in preds and 'Recyclability' in preds:
                    de_val = preds['Degradability']
                    reg_val = preds['Recyclability']
                g_score, g_note, g_color = calculate_green_score(best_poly_data['smiles'], deg_val=de_val, reg_val=reg_val)
                
                st.markdown(f"""
                <div style="background-color:{g_color}20; border: 1px solid {g_color}; border-radius: 5px; padding: 5px; text-align: center;">
                    <strong style="color:{g_color}; font-size: 0.8rem;">🌱 {_('yeşil_score')}</strong><br>
                    <span style="font-size: 1.5rem; font-weight: bold; color:{g_color};">{g_score:.1f}/10</span>
                </div>
                """, unsafe_allow_html=True)
            
            if g_note:
                st.caption(f"**{_('env_analysis')}** {g_note}")

            st.divider()
            if 'Hansen' in preds:
                sol_val = preds['Hansen']
                solvents, partials = get_soluble_solvents(sol_val)
                
                st.markdown(f"### {_('solubility_analysis')}")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.info(f"**{_('soluble_in')}**")
                    if solvents:
                        for s in solvents:
                            st.markdown(f"- ✅ {s}")
                    else:
                        st.warning(f"{_('no_solubility')}")
                
                with c2:
                    st.warning(f"**{_('swelling_in')}**")
                    if partials:
                        for s in partials:
                            st.markdown(f"- ⚠️ {s}")
                    else:
                        st.write("-")
                
                st.caption(f"*{_('solubility_explanation_1')}(δ={sol_val:.1f}) {_('solubility_explanation_2')}")
            if SHOW_RELIABILITY:
                st.caption(f"{_('reliability_legend')}")
            cols = st.columns(3)
            for idx, prop in enumerate(ALL_PROPS):
                with cols[idx % 3]:
                    is_active = prop in saved_active_props
                    target_val = saved_targets.get(prop, '-')
                    target_text = f"{_('target')}: {target_val}" if is_active else f"{_('takip_disi')}"
                    border_color = "#2ecc71" if is_active else "#95a5a6"
                    pred_value = preds[prop]

                    badge_html = ""
                    if SHOW_RELIABILITY:
                        tier = MODEL_RELIABILITY.get(prop, 'medium')
                        badge_html = f'<span title="{_(f"reliability_{tier}")}">{RELIABILITY_BADGE.get(tier, "")}</span>'

                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid {border_color};">
                        <small>{prop} {badge_html}</small><br>
                        <h3 style="margin:0; padding:0;">{pred_value:.2f}</h3>
                        <small style="opacity:0.7">{target_text}</small>
                    </div>
                    """, unsafe_allow_html=True)
            st.divider()
            st.subheader(f"{_('radar_title')}")
            if len(saved_active_props) >= 3:
                    fig = create_radar_chart(preds, saved_targets, saved_active_props, ranges)
                    st.plotly_chart(fig, width='stretch')
            else:
                    st.info(f"{_('radar_warning')}")
                    st.progress(100) 
        with tab2:
            col_2d, col_3d = st.columns(2)
            with col_2d:
                st.subheader(f"{_('2d_structure')}")
                img = draw_2d_molecule(best_poly_data['smiles'])
                if img:
                    st.image(img, width=400)
                st.caption(f"SMILES : {_('code')}")
                st.code(best_poly_data['smiles'], language="text")
                st.caption(f"SELFIES : {_('code')}")
                selfies_str = smiles_to_selfies_safe(best_poly_data['smiles'])
                if selfies_str:
                    st.code(selfies_str, language="text")
                else:
                    st.warning("SELFIES formatına dönüştürülemedi.")
            with col_3d:
                st.subheader(f"{_('3d_structure')}")
                view, reason = make_3d_view_with_reason(best_poly_data["smiles"])
                if view:
                    showmol(view, height=400, width=400)
                else:
                    st.warning(f"3D Model oluşturulamadı: {reason}")
            
            is_avail, cid, name = check_pubchem_availability(best_poly_data['smiles'])
            if is_avail:
                 st.info(f"{_('kayitli')} **{name}** (CID: {cid})")
            st.divider()
            st.subheader(f"{_('novelty_search')}")
            
            similarity_score, similar_smi = calculate_novelty_optimized(best_poly_data['smiles'], reference_smiles)
            
            c1, c2 = st.columns([1, 3])
            
            with c1:
                st.metric(f"{_('similarity_to_train')}", f"%{similarity_score*100:.1f}")
                
            with c2:
                if similarity_score > 0.99:
                    st.error(f"{_('copy')}")
                    st.code(f"{_('benzer')} {similar_smi}")
                elif similarity_score > 0.85:
                    st.warning(f"{_('turev')}")
                    with st.expander(f"{_('benzer_yapi')}"):
                        st.code(similar_smi)
                else:
                    st.success(f"{_('novel')}")
                    st.caption(f" {_('closest_sim')}: %{similarity_score*100:.1f}.")
            
            st.progress(similarity_score)
            st.caption(f"{_('novelty_explanation')}")

        with tab3:
            st.subheader(f"{_('genetic_algorithm_report')}")
            
            if 'best_fitness' in history and len(history['best_fitness']) > 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                gens = range(len(history['best_fitness']))
                
                ax1.plot(gens, history['best_fitness'], label=f'{_("best_fitness")}', color='green', linewidth=2)
                ax1.plot(gens, history['avg_fitness'], label=f'{_("avg_fitness")}', color='blue', linestyle='--', alpha=0.7)
                ax1.set_ylabel(f'{_("metric_total_error")}')
                ax1.set_title(f'{_("convergence_chart")}', fontweight='bold')
                ax1.legend()
                ax1.grid(True, which='both', linestyle='--', alpha=0.5)
                
                ax2.plot(gens, history['diversity'], label=f'{_("diversity_chart")}', color='red', linewidth=2)
                ax2.fill_between(gens, history['diversity'], color='red', alpha=0.1)
                ax2.set_ylabel(f'{_("Çeşitlilik")} (Std Dev)')
                ax2.set_xlabel(f'{_("Generation")}')
                ax2.set_title(f'{_("diversity_chart")}', fontweight='bold')
                ax2.legend()
                ax2.grid(True, which='both', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info(f"""

                {_("how_to_read")}

                """)
            else:
                st.warning("Henüz grafik çizilecek veri yok.")

            # --- Pareto front (only when NSGA-II was used) ---
            pareto = st.session_state.get('ga_pareto') if SHOW_PARETO_TABLE else None
            if pareto:
                st.divider()
                st.subheader(f"{_('pareto_front')}")
                st.caption(f"{_('pareto_desc')}")
                st.info(f"⭐ {_('knee_note')}")

                rows = []
                for i, sol in enumerate(pareto):
                    mark = "⭐" if "knee" in sol.get("tag", "") else ("🎯" if "min-error" in sol.get("tag", "") else "")
                    row = {_("col_pick"): mark, "#": i + 1, "SMILES": sol["smiles"]}
                    for prop in saved_active_props:
                        row[prop] = round(sol["preds"].get(prop, float('nan')), 2)
                    row["SA"] = round(sol["obj"][-1] * 10, 2)
                    rows.append(row)
                df_pareto = pd.DataFrame(rows)
                st.dataframe(df_pareto, width='stretch', height=280)
                st.download_button(
                    label=f"{_('btn_download_csv')}",
                    data=df_pareto.to_csv(index=False).encode('utf-8'),
                    file_name="pareto_front.csv",
                    mime="text/csv",
                )

                # 2-objective scatter of the trade-off, when exactly two targets.
                if len(saved_active_props) == 2:
                    p1, p2 = saved_active_props
                    knee = next((s for s in pareto if "knee" in s.get("tag", "")), None)
                    fig_p, axp = plt.subplots(figsize=(7, 5))
                    xs = [s["preds"].get(p1, float('nan')) for s in pareto]
                    ys = [s["preds"].get(p2, float('nan')) for s in pareto]
                    axp.scatter(xs, ys, c='#e74c3c', s=60, zorder=3, label=f"{_('pareto_front')}")
                    axp.scatter([saved_targets.get(p1)], [saved_targets.get(p2)],
                                c='#2ecc71', marker='*', s=320, zorder=4, edgecolors='black', label=f"{_('target')}")
                    if knee:
                        axp.scatter([knee["preds"].get(p1)], [knee["preds"].get(p2)],
                                    c='#f1c40f', marker='D', s=160, zorder=5, edgecolors='black',
                                    label=f"⭐ {_('knee_point')}")
                    axp.set_xlabel(p1)
                    axp.set_ylabel(p2)
                    axp.set_title(f"{_('pareto_front')}: {p1} vs {p2}", fontweight='bold')
                    axp.grid(True, linestyle='--', alpha=0.4)
                    axp.legend()
                    st.pyplot(fig_p)
        # with tab3:
            # st.header(" Performans Kıyaslama (Benchmark)")
            # st.markdown("Modelin başarısını kanıtlamak için onu 'Rastgele Arama' ile yarıştırın.")
            
            # # Eğer GA sonuçları varsa
            # if 'ga_history' in st.session_state and 'best_fitness' in st.session_state['ga_history']:
            #     history = st.session_state['ga_history']
            #     ga_best_curve = history['best_fitness']
                
            #     
            #     if st.button(" Rastgele Arama ile Kıyasla (Benchmark Başlat)"):
            #         with st.spinner("Rastgele Arama yapılıyor... Bu işlem GA kadar sürebilir."):
            #             
            #             generations_run = len(ga_best_curve)
            #             pop_size = 100 # Kodunuzda sabit 100'dü
            #             total_evals = generations_run * pop_size
                        
            #            
            #             random_curve = run_random_benchmark(
            #                 models, saved_targets, saved_active_props, 
            #                 initial_selfies, ranges, 
            #                 total_budget=total_evals, 
            #                 batch_size=pop_size
            #             )
                        
            #            
            #             st.session_state['random_curve'] = random_curve
            #             st.success("Benchmark Tamamlandı!")

            #    
            #     fig, ax = plt.subplots(figsize=(10, 6))
                
            #     
            #     ax.plot(ga_best_curve, label='Genetik Algoritma (Sizin Modeliniz)', color='green', linewidth=2.5)
                
            #    
            #     if 'random_curve' in st.session_state:
            #      
            #         min_len = min(len(ga_best_curve), len(st.session_state['random_curve']))
            #         r_curve = st.session_state['random_curve'][:min_len]
            #         g_curve = ga_best_curve[:min_len]
                    
            #         ax.plot(r_curve, label='Rastgele Arama (Random Search)', color='gray', linestyle='--', linewidth=2)
                    
            #      
            #         diff = r_curve[-1] - g_curve[-1]
            #         st.caption(f"**Sonuç:** GA modeliniz, rastgele aramadan **{diff:.2f} puan** daha iyi performans gösterdi.")
                
            #     ax.set_title("Zeka Testi: GA vs Şans", fontweight='bold')
            #     ax.set_xlabel("Jenerasyon (Her adımda 100 yeni deneme)")
            #     ax.set_ylabel("Hata Skoru (Düşük İyidir)")
            #     ax.legend()
            #     ax.grid(True, linestyle='--', alpha=0.5)
                
            #     st.pyplot(fig)
                
            #     st.info("""
            #     **Grafik Nasıl Yorumlanır?**
            #     * **Yeşil Çizgi:** Hızlıca aşağı iniyorsa, modeliniz 'öğreniyor' demektir.
            #     * **Gri Çizgi:** Genelde daha yukarıda ve düz kalır.
            #     * **Fark:** İki çizgi arasındaki boşluk, Yapay Zekanızın kattığı değerdir.
            #     """)
                
            # else:
            #     st.warning("Önce 'Hedefi Ara' butonuna basarak GA'yı çalıştırın, sonra kıyaslama yapabilirsiniz.")
            # st.divider()
            # st.header("Stres Testi (Mass Random Testing)")
            # st.markdown("""
            # Modelin **genelleştirme yeteneğini** ölçmek için rastgele hedeflerle çoklu deneme yapın.
            # * Her denemede farklı özellikler ve farklı hedef değerler seçilir.
            # * Modelin "kolay" ve "zor" hedeflere tepkisi ölçülür.
            # """)
            
            # col_mass_input, col_mass_btn = st.columns([1, 2])
            # with col_mass_input:
            #     mass_trials = st.number_input("Test Sayısı", min_value=10, max_value=500, value=100, step=10)
            
            # if col_mass_btn.button("100+ Rastgele Testi Başlat"):
            #     with st.spinner("Model zorlu bir sınava giriyor... Kahvenizi alın, bu biraz sürebilir."):
            #         df_results = run_mass_random_test(models, generations, initial_selfies, ranges, num_trials=mass_trials)
                    
            #         st.subheader(" Test Sonuçları 📊")
                    
            #         avg_error = df_results["Final Hata Skoru"].mean()
            #         success_count = df_results[df_results["Final Hata Skoru"] < 5.0].shape[0]
            #         success_rate = (success_count / mass_trials) * 100
                    
            #         m1, m2, m3 = st.columns(3)
            #         m1.metric("Ortalama Hata", f"{avg_error:.2f}")
            #         m2.metric("Başarı Oranı (Hata < 5.0)", f"%{success_rate:.1f}")
            #         m3.metric("En Zorlu Senaryo Hatası", f"{df_results['Final Hata Skoru'].max():.2f}")
                    
            #         fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
            #         ax_hist.hist(df_results["Final Hata Skoru"], bins=20, color='#3498db', edgecolor='black', alpha=0.7)

            #         median_error = df_results["Final Hata Skoru"].median() 
            #         ax_hist.set_title("Hata Skorlarının Dağılımı (Histogram)")
            #         ax_hist.set_xlabel("Hata Skoru (Sola yığılma iyidir)")
            #         ax_hist.set_ylabel("Deneme Sayısı")
            #         ax_hist.axvline(avg_error, color='red', linestyle='dashed', linewidth=1, label=f'Ortalama: {avg_error:.2f}')
            #         ax_hist.axvline(median_error, color='green', linestyle='-.', linewidth=1.5, label=f'Medyan: {median_error:.2f}')
            #         ax_hist.legend()
            #         st.pyplot(fig_hist)
                    
            #         fig_sc, ax_sc = plt.subplots(figsize=(10, 5))
            #         ax_sc.scatter(df_results["Hedef Sayısı"], df_results["Final Hata Skoru"], alpha=0.6, c=df_results["Final Hata Skoru"], cmap='viridis')
            #         ax_sc.set_title("Hedef Sayısı vs. Başarı")
            #         ax_sc.set_xlabel("Aktif Hedef Sayısı (Zorluk)")
            #         ax_sc.set_ylabel("Hata Skoru")
            #         ax_sc.grid(True, alpha=0.3)
            #         st.pyplot(fig_sc)
                    
            #         with st.expander("📄 Tüm Test Verilerini Gör"):
            #             st.dataframe(df_results)

        with tab4:
            st.header(f"💾 {_('report_header')}")
            st.markdown(f"{_('report_desc')}")
            
            c1, c2 = st.columns(2)
            
            export_dict = {
                "SMILES": best_poly_data['smiles'],
                "Toplam Hata": best_poly_data['total_error'],
                "SA Score": get_sa_score_local(best_poly_data['smiles'])
            }
            export_dict.update(preds)
            df_best = pd.DataFrame([export_dict])
            csv_best = df_best.to_csv(index=False).encode('utf-8')

            with c1:
                st.download_button(
                    label=f"{_('btn_download_csv')}",
                    data=csv_best,
                    file_name="polimer_data.csv",
                    mime="text/csv"
                )
            
            st.divider()
            
            st.subheader(f"{_('pdf_report')}")
            st.info(f"{_('pdf_report_info')}")

            gen_ai_analysis = st.session_state.get('ai_analysis', "Genel AI analizi yapilmadi.")
            
            manual_retro = st.session_state.get('retro_manual_text', "Otomatik ayristirma verisi yok (Retrosentez sekmesini ziyaret edin).")
            ai_retro = st.session_state.get('ai_retro_text', "AI sentez recetesi olusturulmadi.")
            
            full_retro_info = manual_retro + "\n\n--- AI Sentez Notlari ---\n" + ai_retro

            if st.button(f"{_('btn_create_pdf')}", type="primary", width='stretch'):
                with st.spinner(f"{_('report_getting_ready')}..."):
                    pdf_data = create_pdf_report(
                        best_poly_data, 
                        saved_targets, 
                        saved_active_props, 
                        gen_ai_analysis, 
                        full_retro_info
                    )
                    
                    st.success(f"{_('report_ready')}")
                    st.download_button(
                        label=f"📥 {_('download')} {_('pdf_report')}",
                        data=pdf_data,
                        file_name="PolimerX_Final_Raporu.pdf",
                        mime="application/pdf",
                        width='stretch'
                    )
        # with tab5:
        #     st.subheader("ChatBot")
        #     st.markdown(f"{_('chatbot_desc')}")
        #     if not api_key:
        #         st.info(f"{_('chatbot_no_api')}")
        #         st.markdown(f"{_('get_free_api')}")
        #     else:
        #         if st.button(f"{_('analyze_polymer')}", type="primary"):
        #             analysis_result = get_ai_interpretation(
        #                 api_key, 
        #                 best_poly_data['smiles'], 
        #                 best_poly_data['preds'], 
        #                 saved_targets, 
        #                 saved_active_props
        #             )
        #             st.markdown(analysis_result)
                    
        #             st.session_state['ai_analysis'] = analysis_result
                
        #         elif 'ai_analysis' in st.session_state:
        #             st.markdown(st.session_state['ai_analysis'])
        with tab6:
            st.header(f"{_('retro_analysis_header')}")
            
            target_smiles = best_poly_data['smiles']
            
            st.subheader(f"1. {_('retro_structural')}")
            # Rule-based backbone disconnection -> REAL monomer structures.
            retro_routes = retro.retro_decompose(target_smiles)

            monomer_info_text = f"{_('retro_auto_failed')}"

            if retro_routes:
                route = retro_routes[0]
                monomer_info_text = (f"{_('yontem')}: {route['type']}\n"
                                     f"{_('mechanism')}: {route['mechanism']}\n"
                                     f"Monomers: {' . '.join(route['monomers'])}\n")

                st.info(f"**{_('yontem')}:** {route['type']}")
                st.write(f"**{_('mechanism')}:** {route['mechanism']}")
                if route.get('verified', False):
                    st.caption(f"🔬 {_('retro_rule_method')}")
                else:
                    st.caption(f"⚠️ {_('retro_tentative')}")

                st.markdown(f"{_('baslangic_monomerleri')}")
                img_retro = draw_retrosynthesis_grid(route['monomers'])
                if img_retro:
                    st.image(img_retro)

                st.markdown(f"{_('commercial_check')}")
                found_monomers = []
                for i, m in enumerate(route['monomers']):
                    col_code, col_check = st.columns([3, 1])
                    with col_code:
                        st.code(f"Monomer {i+1}: {m}")
                    with col_check:
                        if st.button(f"{_('control')} #{i+1}", key=f"chk_{i}"):
                            is_avail, cid, name = check_commercial_availability(m)
                            if is_avail:
                                st.success(f"{_('kayitli')}: {name}")
                                found_monomers.append(name)
                            else:
                                st.error(f"{_('kayitli_degil')}")

                if found_monomers:
                    monomer_info_text += f"{_('kayitli')}: {', '.join(found_monomers)}"

            else:
                # No clean disconnection -> show a mechanism guess only, and be honest.
                named = decompose_polymer(target_smiles)
                if isinstance(named, list) and named:
                    br = named[0]
                    st.info(f"**{_('yontem')}:** {br['type']}")
                    st.write(f"**{_('mechanism')}:** {br['mechanism']}")
                    st.caption(f"⚠️ {_('retro_named_only')}")
                    monomer_info_text = f"{_('yontem')}: {br['type']}\n{_('mechanism')}: {br['mechanism']}"
                else:
                    st.warning(f"{_('retro_auto_failed')}")
                    monomer_info_text = f"{_('retro_auto_failed')}"

            st.session_state['retro_manual_text'] = monomer_info_text

            # st.divider()

            # st.subheader("2. AI Sentez Reçetesi")
            
            # if api_key and st.button("Sentez Rotasını Oluştur (AI)", type="primary"):
            #     ai_retro_text = get_ai_retrosynthesis_guide(api_key, target_smiles, str(retro_results))
            #     st.markdown(ai_retro_text)
            #     st.session_state['ai_retro_text'] = ai_retro_text
            
            # elif 'ai_retro_text' in st.session_state:
            #     st.markdown(st.session_state['ai_retro_text'])
            
            st.divider()

            st.subheader(f"{_('retro_t5_pred')}")
            st.caption(f"{_('retro_t5_desc')}")

            if st.button(f"{_('btn_predict_monomer')}", type="primary"):
                with st.spinner(f"{_('msg_predicting')}..."):
                    prediction = predict_monomers_local(best_poly_data['smiles'])
                    
                    st.success(f"{_('msg_success')}")
                    
                    st.markdown(f"""
                    <div style="background-color:#e8f5e9; padding:15px; border-radius:10px; border:1px solid #4CAF50;">
                        <h4 style="color:#2e7d32; margin:0;">{_('proposed_monomers')}</h4>
                        <code style="font-size:1.1em; color:#1b5e20; background-color:#e8f5e9;">{prediction}</code>
                    </div>
                    """, unsafe_allow_html=True)
                    print("Predicted monomers:", prediction)
                    monomers_list = prediction.split(' . ') 
                    img_retro = draw_retrosynthesis_grid(monomers_list)
                    if img_retro:
                        st.image(img_retro, caption=f"{_('predicted_monomers_image_caption')}")
                    st.session_state['retro_manual_text'] = f"{_('predicted_monomers')}: {prediction}"



