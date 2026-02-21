import google.generativeai as genai
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
import torch
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies as sf
from datasets import load_dataset
import rdkit.Chem.rdChemReactions as rdChemReactions
import sys
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Draw
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
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
st.set_page_config(
    page_title="POLSEN", 
    page_icon="🧬",
)

@st.cache_resource
def load_my_trained_model():
    model_path = "OsBaran/POLSEN_T5"
    try:
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
    Önce eğitilmiş T5 modelini kullanır. 
    Eğer sonuç başarısızsa kural tabanlı motoru devreye sokar.
    """
    print(f"--- [LOG] {time.strftime('%X')} - Girdi (Input) hazırlanıyor...", flush=True)

    tokenizer, model = load_my_trained_model()
    ai_prediction = ""
    
    if model:
        try:
            input_text = "retrosynthesis: " + polymer_smiles
            inputs = tokenizer(input_text, return_tensors="pt")
            print(f"--- [LOG] {time.strftime('%X')} - MODEL GENERATE BAŞLIYOR (Burada bekleyebilir)...", flush=True)
            start_time = time.time()

            outputs = model.generate(
                inputs["input_ids"], 
                max_length=64, 
                num_beams=5,
                early_stopping=True,
            )
            ai_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            end_time = time.time()
            print(f"--- [LOG] {time.strftime('%X')} - MODEL GENERATE BİTTİ! Geçen süre: {round(end_time - start_time, 2)} saniye.", flush=True)
            
            print(f"--- [LOG] {time.strftime('%X')} - Çıktı okunuyor (Decode)...", flush=True)
            
            print("--- [LOG] İşlem başarıyla tamamlandı.", flush=True)
        except:
            ai_prediction = ""

    if ai_prediction and " . " in ai_prediction:
        return f"{ai_prediction} (T5 Modeli)"
    
    else:
        rules = decompose_polymer(polymer_smiles) 
        if rules:
            monomers = rules[0]['monomers']
            return f"{' . '.join(monomers)} (Kural Tabanlı)"
        else:
            if ai_prediction:
                return f"{ai_prediction} (T5 Modeli)"
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
def get_soluble_solvents(pred_val):
    """Tahmin edilen Hildebrand değerine göre uygun çözücüleri bulur."""
    soluble_list = []
    swelling_list = [] 
    
    for solvent, s_val in COMMON_SOLVENTS.items():
        diff = abs(pred_val - s_val)
        
        if diff <= 1.8: 
            soluble_list.append(solvent)
        elif diff <= 2.5: 
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
@st.cache_resource
def load_critic_models():
    """Tüm Eleştirmen (Critic) modellerini yükler."""
    models = {}
    try:
        models['Tg'] = joblib.load('xgb_tg.joblib')
        models['Td'] = joblib.load('xgb_td.joblib')
        models['EPS'] = joblib.load('rf_eps.joblib')
        models['Tm'] = joblib.load('xgb_tm.joblib')
        models['BandgapBulk'] = joblib.load('xgb_band gap bulk.joblib')
        models['BandgapChain'] = joblib.load('xgb_band gap chain.joblib')
        models['BandgapCrystal'] = joblib.load('xgb_bandgap-crystal.joblib')
        models['GasPerma'] = joblib.load('lgbm_gas_pipeline.joblib')
        models['Refractive'] = joblib.load('rf_refractive_index.joblib')
        models['LOI'] = joblib.load('xgb_loi.joblib')
        models['Solubility'] = joblib.load('xgb_solubility.joblib') 
        models['ThermalCond'] = joblib.load('xgb_thermal_cond.joblib') 
        models['CTE'] = joblib.load('xgb_cte.joblib')
        models['Recyclability'] = joblib.load('chemical_recyclability.joblib')
        models['Degradability'] = joblib.load('xgb_degradability.joblib')
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
    toolbox.register("mate", cxSelfies)
    toolbox.register("select", tools.selTournament, tournsize=7) 

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
            offspring[i] = generate_offspring(offspring[i], initial_pop, mutpb=mutpb, extendpb=extendpb, newpb=newpb, chempb=chempb)
            del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
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

def get_morgan_fp(p_smiles):
    smi_clean = str(p_smiles).replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, N_BITS)
    return np.array([fp])


def get_gas_features_combined(smiles):
    """
    Gaz geçirgenliği LGBM modeli için Morgan FP
    """
    try:
        mol = Chem.MolFromSmiles(smiles.replace('*', '[H]'))
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
    """Bozunabilirlik modeli için MAX HIZDA özellik çıkarımı (Pandas Yok)."""
    try:
        mol = Chem.MolFromSmiles(str(smiles).replace('*', '[H]'))
        if mol is None: return None
        
        desc_vals = [
            Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumRotatableBonds(mol)
        ]
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
        final_arr = np.concatenate([desc_vals, np.array(fp)])
        
        # PANDAS İPTAL: Doğrudan XGBoost'un sevdiği 2D Numpy Matrisini döndürüyoruz!
        return final_arr.reshape(1, -1)
    except:
        return None

def get_recyclability_features(smiles):
    """Geri Dönüşüm modeli için MAX HIZDA özellik çıkarımı (Pandas Yok)."""
    try:
        mol = Chem.MolFromSmiles(str(smiles).replace('*', '[H]'))
        if mol is None: return None
        
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms() if atom.HasProp('_GasteigerCharge')]
        
        if charges:
            max_abs_c = max(abs(c) for c in charges)
            max_c = max(charges)
            min_c = min(charges)
        else:
            max_abs_c, max_c, min_c = 0.0, 0.0, 0.0
            
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        try: balaban = GraphDescriptors.BalabanJ(mol)
        except: balaban = 0.0
        
        feats = [
            0.0, max_abs_c, max_c, min_c,
            1 if any(len(r) == 3 for r in atom_rings) else 0,
            1 if any(len(r) == 4 for r in atom_rings) else 0,
            1 if any(len(r) == 5 for r in atom_rings) else 0,
            1 if any(len(r) == 6 for r in atom_rings) else 0,
            1 if any(len(r) >= 7 for r in atom_rings) else 0,
            ring_info.NumRings(),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Fragments.fr_ester(mol),
            Fragments.fr_amide(mol),
            Fragments.fr_ether(mol),
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            balaban
        ]
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
        final_arr = np.concatenate([feats, np.array(fp)])
        
        # PANDAS İPTAL: Sadece Numpy
        return final_arr.reshape(1, -1)
    except:
        return None
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
def evaluate_individual_optimized(individual, models, targets, active_props, ranges):
    s_selfies = individual[0]
    
    if s_selfies in FITNESS_CACHE:
        return FITNESS_CACHE[s_selfies]

    s_smiles = selfies_to_smiles_safe(s_selfies)
    if s_smiles is None: return (1000.0,)

    fp = get_morgan_fp(s_smiles)
    
    gas_features = None
    rec_features = None # Yeni eklenen satır
    deg_features = None # Yeni eklenen satır

    if 'GasPerma' in active_props:
        gas_features = get_gas_features_combined(s_smiles)
    if 'Recyclability' in active_props: # Yeni eklenen blok
        rec_features = get_recyclability_features(s_smiles)
    if 'Degradability' in active_props: # Yeni eklenen blok
        deg_features = get_degradability_features(s_smiles)

    if fp is None: return (1000.0)

    preds = {}
    
    for prop in active_props:
        if prop in models:
            if prop == 'GasPerma':
                if gas_features is not None:
                    log_pred = models[prop].predict(gas_features)[0]
                    preds[prop] = 10 ** log_pred 
                else:
                    preds[prop] = 0.0
            elif prop == 'Recyclability': # Yeni eklenen blok
                if rec_features is not None:
                    preds[prop] = models[prop].predict(rec_features)[0]
                else:
                    preds[prop] = 0.0
            elif prop == 'Degradability': # Yeni eklenen blok
                if deg_features is not None:
                    preds[prop] = models[prop].predict(deg_features)[0]
                else:
                    preds[prop] = 0.0
            else:
                preds[prop] = models[prop].predict(fp)[0]
    
    total_error = 0.0
    if not active_props: return (1000.0,)

    for prop in active_props:
        if prop in preds:
            norm_error = abs(preds[prop] - targets[prop]) / (ranges[prop]['max'] - ranges[prop]['min'])
            total_error += np.exp(norm_error * 10) - 1
    
    sa_score = get_sa_score_local(s_smiles)
    total_error += sa_score * 2.0 
    
    result = (total_error,)
    FITNESS_CACHE[s_selfies] = result
    return result
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

def generate_offspring(individual, initial_selfies, mutpb=0.05, extendpb=0.05, newpb=0.01, chempb=0.05):
    """Mutasyon, zincir uzatma, yeni birey ve reaction mutasyonunu uygular."""
    # 1. SELFIES mutasyonu
    if random.random() < mutpb:
        individual = mutSelfies(individual)
        mutation_stats['SELFIES'] += 1

    # 2. Reaction tabanlı mutasyon
    if random.random() < chempb:
        smi = selfies_to_smiles_safe(individual[0])
        if smi:
            mutated = chemically_valid_mutate(smi)
            ind_selfies = smiles_to_selfies_safe(mutated)
            if ind_selfies:
                individual[0] = ind_selfies
                mutation_stats['REACTION'] += 1

    # 3. Zincir uzatma
    if random.random() < extendpb:
        individual[0] = extendPolymer(individual)
        mutation_stats['EXTEND'] += 1

    # 4. Rastgele yeni birey
    if random.random() < newpb:
        individual[0] = random.choice(initial_selfies)
        mutation_stats['NEW'] += 1

    # 5. Geçerlilik kontrolü
    if not is_valid_polymer(individual[0]):
        individual[0] = random.choice(initial_selfies)
    return individual

def run_single_objective_flow(models, generations, targets, active_props, initial_pop, ranges_dict):
    toolbox = base.Toolbox()
    toolbox.register("attr_selfies", random.choice, initial_pop)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_selfies, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_individual_optimized, models=models, targets=targets, active_props=active_props, ranges=ranges_dict)
    
    toolbox.register("mate", cxSelfies)
    toolbox.register("select", tools.selTournament, tournsize=7)
    
    pop_size = 100 
    pop = toolbox.population(n=pop_size)
    
    history = {
        "gen": [],
        "best_fitness": [],
        "avg_fitness": [],
        "diversity": [] 
    }

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    st.markdown("### 🧬 Evrimsel Süreç İzleme Paneli")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.caption("Yakınsama (Convergence)")
        chart_fitness_placeholder = st.empty()
    with col_chart2:
        st.caption("Popülasyon Çeşitliliği (Diversity)")
        chart_diversity_placeholder = st.empty()

    log_expander = st.expander("GA Logları (Detaylı)", expanded=False)
    with log_expander:
        log_placeholder = st.empty()
        mutation_placeholder = st.empty()

    log_data = [] 

    for gen in range(generations):
        scale = gen / generations
        cxpb = 0.7 - (0.2 * scale)
        mutpb = 0.05 - (0.2 * scale)
        extendpb = 0.05 - (0.15 * scale)
        newpb = 0.01 - (0.05 * scale)
        chempb = 0.05 - (0.15 * scale)

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
            offspring[i] = generate_offspring(offspring[i], initial_pop, mutpb=mutpb, extendpb=extendpb, newpb=newpb, chempb=chempb)
            del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
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
            "Nesil": gen + 1,
            "En İyi Hata": round(best_val, 4),
            "Ortalama (Valid)": round(mean_val, 4),
            "Canlılık Oranı %": round(survival_rate, 1) 
        })

        if gen % 2 == 0 or gen == generations - 1:
            progress_bar.progress((gen + 1) / generations)
            status_text.markdown(f"**Nesil {gen+1}/{generations}** | En İyi Hata: `{best_val:.4f}` | Çeşitlilik: `{std_val:.4f}`")
            
            df_fit = pd.DataFrame({
                "En İyi (Best)": history["best_fitness"],
                "Ortalama (Avg)": history["avg_fitness"]
            })
            chart_fitness_placeholder.line_chart(df_fit, height=250)
            
            df_div = pd.DataFrame({
                "Çeşitlilik (Std Dev)": history["diversity"]
            })
            chart_diversity_placeholder.line_chart(df_div, height=250)
            
            df_log = pd.DataFrame(log_data)
            log_placeholder.dataframe(df_log.sort_values(by="Nesil", ascending=False).head(5), use_container_width=True)
            mutation_placeholder.json(mutation_stats)

    best_ind = tools.selBest(pop, 5)[0]
    best_smiles = selfies_to_smiles_safe(best_ind[0])
    
    best_ind = tools.selBest(pop, 5)[0]
    best_smiles = selfies_to_smiles_safe(best_ind[0])
    
    if best_smiles:
        fp = get_morgan_fp(best_smiles)
        
        gas_features = get_gas_features_combined(best_smiles)
        rec_features = get_recyclability_features(best_smiles)
        deg_features = get_degradability_features(best_smiles)

        preds = {}
        
        for prop in models.keys():
            if prop == 'GasPerma':
                if gas_features is not None:
                    log_pred = models[prop].predict(gas_features)[0]
                    preds[prop] = 10 ** log_pred
                else:
                    preds[prop] = 0.0
            elif prop == 'Recyclability': # Yeni eklenen blok
                if rec_features is not None:
                    preds[prop] = models[prop].predict(rec_features)[0]
                else:
                    preds[prop] = 0.0
            elif prop == 'Degradability': # Yeni eklenen blok
                if deg_features is not None:
                    preds[prop] = models[prop].predict(deg_features)[0]
                else:
                    preds[prop] = 0.0
            else:
                preds[prop] = models[prop].predict(fp)[0]
        
        return {'smiles': best_smiles, 'preds': preds, 'total_error': best_ind.fitness.values[0]}, history
    else:
        return None, history

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
        return sascorer.calculateScore(mol)
    except:
        length = len(str(p_smiles))
        score = 2.0 + (length * 0.05)
        if "c1" in str(p_smiles): 
            score += 0.5
        return min(score, 10.0)

def calculate_green_score(smiles):
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
        notes.append("Ester bağı (Hidroliz olabilir)")
        
    # Amid 
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C;!R](=[O])[N;!R]")):
        score += 2.0
        notes.append("Amid bağı (Biyo-bozunurluk potansiyeli)")
        
    # Eter
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C][O][C]")):
        score += 1.0
        notes.append("Eter grubu (Hidrofilik özellik)")

    # Halojen
    halogens = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br']]
    if halogens:
        count = len(halogens)
        penalty = min(4.0, count * 1.0) 
        score -= penalty
        notes.append(f"{count} adet Halojen atomu (Kalıcılık/Toksisite riski)")
        
    # Aromatik
    aromatic_atoms = [atom for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    if len(aromatic_atoms) > 4: 
        score -= 2.0
        notes.append("Yüksek Aromatiklik (Zor parçalanma)")

    score = max(1.0, min(10.0, score)) 
    
    if score >= 7.0: color = "#2ecc71" 
    elif score >= 4.0: color = "#f1c40f" 
    else: color = "#e74c3c" 
    
    return score, ", ".join(notes), color

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
        name='Hedeflenen',
        line=dict(color='#3498db', dash='dash')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=pred_values,
        theta=categories,
        fill='toself',
        name='Uretilen Polimer',
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
            "type": "Poliimid Sentezi",
            "reaction": "Siklo-dehidrasyon",
            "monomers": ["Dianhidrit", "Diamin"],
            "mechanism": "Dianhidrit + Diamin -> Poliimid"
        }]

    # ÜRE 
    urea_pattern = Chem.MolFromSmarts("[N;!R][C;!R](=[O])[N;!R]")
    if mol.HasSubstructMatch(urea_pattern):
        breakdown_results.append({
            "type": "Poliüre (Polyurea) Sentezi",
            "reaction": "Basamaklı Polimerizasyon (Hızlı)",
            "monomers": ["Diizosiyanat (Diisocyanate)", "Diamin (Diamine)"],
            "mechanism": "İzosiyanat + Amin -> Üre Bağı (Yan ürün yok)"
        })

    # ÜRETAN 
    urethane_pattern = Chem.MolFromSmarts("[N;!R][C;!R](=[O])[O;!R]")
    if mol.HasSubstructMatch(urethane_pattern):
        breakdown_results.append({
            "type": "Poliüretan (PU) Sentezi",
            "reaction": "Poliladisyon",
            "monomers": ["Diizosiyanat (Örn: TDI, MDI)", "Diol / Polyol"],
            "mechanism": "İzosiyanat + Alkol -> Üretan Bağı"
        })

    # ESTER 
    ester_pattern = Chem.MolFromSmarts("[C;!R](=[O])[O;!R]") 
    if mol.HasSubstructMatch(ester_pattern) and not breakdown_results:
        breakdown_results.append({
            "type": "Polyester Sentezi",
            "reaction": "Kademeli Polimerizasyon",
            "monomers": ["Dikarboksilik Asit", "Diol"],
            "mechanism": "Asit + Alkol -> Ester + Su"
        })

    # AMİD 
    amide_pattern = Chem.MolFromSmarts("[C;!R](=[O])[N;!R]")
    if mol.HasSubstructMatch(amide_pattern) and not breakdown_results: 
         breakdown_results.append({
            "type": "Poliamid (Nylon) Sentezi",
            "reaction": "Polikondenzasyon",
            "monomers": ["Dikarboksilik Asit", "Diamin"],
            "mechanism": "Asit + Amin -> Amid + Su"
        })

    #VARSAYILAN 
    if not breakdown_results:
        has_hetero = any(atom.GetSymbol() in ['N', 'O', 'S'] for atom in mol.GetAtoms())
        if has_hetero and "C=C" not in smiles:
             breakdown_results.append({
                "type": "Kompleks Kondenzasyon Polimeri",
                "reaction": "Özel Sentez (AI Analizi Önerilir)",
                "monomers": ["Fonksiyonel Grup A", "Fonksiyonel Grup B"],
                "mechanism": "Uç grupların reaksiyonu"
            })
        else:
            breakdown_results.append({
                "type": "Vinil Polimerizasyonu (Katılma)",
                "reaction": "Radikalik",
                "monomers": [smiles.replace('*', '')],
                "mechanism": "Çift bağ açılması"
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
        self.cell(0, 10, 'PolimerX - Ar-Ge Proje Raporu', 0, 1, 'C')
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

st.markdown('<h1 class="main-title"> POLSEN <br><span style="font-size:1.5rem; color:#666; font-weight:400;">Yapay Zeka Destekli Polimer Tasarlayıcısı</span></h1>', unsafe_allow_html=True)

models = load_critic_models()
ALL_PROPS = list(models.keys()) 

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
    st.sidebar.header("Hedef Seçimi")
    
    active_props = []
    
    st.sidebar.markdown("### Dahil Edilecek Özellikler")
    if st.sidebar.checkbox("Tg (Camsı Geçiş Sıcaklığı)", value=True):
        active_props.append('Tg')
    if st.sidebar.checkbox("Td (Bozunma Sıcaklığı)"):
        active_props.append('Td')
    if st.sidebar.checkbox("EPS (Dielektrik Sabiti)"):
        active_props.append('EPS')
    if st.sidebar.checkbox("Tm (Erime Sıcaklığı)"):
        active_props.append('Tm')
    if st.sidebar.checkbox("Bandgap Bulk (Elektriksel Band Aralığı - Bulk)"):
        active_props.append('BandgapBulk')
    if st.sidebar.checkbox("Bandgap Chain (Elektriksel Band Aralığı - Zincir)"):
        active_props.append('BandgapChain')
    if st.sidebar.checkbox("Bandgap Crystal (Elektriksel Band Aralığı - Kristal)"):
        active_props.append('BandgapCrystal')
    if st.sidebar.checkbox("Gas Permeability (Gaz Geçirgenliği)"):
        active_props.append('GasPerma')
    if st.sidebar.checkbox("Refractive Index (Kırılma İndeksi)"):
        active_props.append('Refractive')
    if st.sidebar.checkbox("LOI (Yanıcılık İndeksi)"): 
        active_props.append('LOI')
    if st.sidebar.checkbox("Çözünürlük (Hildebrand)"): 
        active_props.append('Solubility') 
    if st.sidebar.checkbox("Isıl İletkenlik (Thermal Cond.)"): 
        active_props.append('ThermalCond')  
    if st.sidebar.checkbox("Isıl Genleşme (CTE)"): 
        active_props.append('CTE')
    if st.sidebar.checkbox("Kimyasal Geri Dönüşüm (Entalpi - ΔH)"):
        active_props.append('Recyclability')
    if st.sidebar.checkbox("Bozunabilirlik Skoru"):
        active_props.append('Degradability')
    if not active_props:
        st.sidebar.warning("Lütfen optimize edilecek en az bir hedef seçin.")
        st.stop()

    st.sidebar.markdown("### Hedef Değerler")
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
        'Refractive': {'min': 1.2, 'max': 2.0, 'default': 1.5, 'step': 0.01, 'is_int': False},
        'LOI': {'min': 15.0, 'max': 100.0, 'default': 28.0, 'step': 0.5, 'is_int': False},
        'Solubility': {'min': 5.0, 'max': 20.0, 'default': 9.5, 'step': 0.1, 'is_int': False},
        'ThermalCond': {'min': 0.0, 'max': 1.0, 'default': 0.2, 'step': 0.01, 'is_int': False},
        'CTE': {'min': 0.0, 'max': 300.0, 'default': 60.0, 'step': 5.0, 'is_int': False},
        'Recyclability': {'min': -200.0, 'max': 50.0, 'default': -50.0, 'step': 1.0, 'is_int': False},
        'Degradability': {'min': 0.0, 'max': 5.0, 'default': 2.0, 'step': 0.1, 'is_int': False},
    }

    for prop in active_props:
        if prop in ranges:
            r = ranges[prop]
            label = prop
            if prop == 'Tg': label = 'Hedef Tg (°C)'
            elif prop == 'Td': label = 'Hedef Td (°C)'
            elif prop == 'Tm': label = 'Hedef Tm (°C)'
            elif prop == 'EPS': label = 'Hedef EPS'
            elif prop == 'BandgapBulk': label = 'Hedef BandgapBulk (eV)'
            elif prop == 'BandgapChain': label = 'Hedef BandgapChain (eV)'
            elif prop == 'BandgapCrystal': label = 'Hedef BandgapCrystal (eV)'
            elif prop == 'GasPerma': label = 'Hedef GasPerma'
            elif prop == 'Refractive': label = 'Hedef Refractive Index'

            val = add_synced_input(prop, label, r['min'], r['max'], r['default'], r['step'], is_int=r['is_int'])
            targets[prop] = val
        else:
            targets[prop] = st.sidebar.number_input(f"Hedef {prop}:", value=0.0)

    # 3. GA Parametreleri
    generations = st.sidebar.slider("Evrim Nesli Sayısı", 10, 300, 10)

    initial_selfies, reference_smiles = get_initial_population()
    st.sidebar.divider()
    st.sidebar.markdown("LLM ChatBot Ayarları")
    api_key = st.sidebar.text_input("API Key", type="password", help="AI yorumu almak için https://aistudio.google.com/app/apikey adresinden ücretsiz anahtar alabilirsiniz.")
    if st.sidebar.button("Hedefi Ara", type="primary"):
        
        if not initial_selfies:
            st.error("Başlangıç popülasyonu boş veya geçersiz.")
            st.stop()
            
        with st.spinner(f'Genetik Algoritma Çalışıyor... Hedefler: {", ".join(active_props)}'):
            best_poly_data, history = run_single_objective_flow(models, generations, targets, active_props, initial_selfies, ranges)

        if best_poly_data:
            st.session_state['ga_results'] = best_poly_data
            st.session_state['ga_history'] = history
            st.session_state['ga_targets'] = targets 
            st.session_state['ga_active_props'] = active_props 
            
    
    if 'ga_results' in st.session_state:
        
        best_poly_data = st.session_state['ga_results']
        history = st.session_state['ga_history']
        saved_targets = st.session_state['ga_targets']
        saved_active_props = st.session_state['ga_active_props']
        
        preds = best_poly_data['preds']
        
        st.success("✅ Optimizasyon Başarıyla Tamamlandı!")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Genel", "Yapısal Analiz", "Evrim Geçmişi", "Raporlama", "ChatBot", "Retrosentez"])
        with tab1:
            col_main, col_score, col_green = st.columns([2, 1, 1])
            
            with col_main:
                st.markdown(f"### Toplam Hata: **{best_poly_data['total_error']:.4f}**")
                
            with col_score:
                sa = get_sa_score_local(best_poly_data['smiles'])
                st.metric("Sentez Zorluğu (SA)", f"{sa:.2f}", help="1 (Kolay) - 10 (Zor)")
                
            with col_green:
                g_score, g_note, g_color = calculate_green_score(best_poly_data['smiles'])
                
                st.markdown(f"""
                <div style="background-color:{g_color}20; border: 1px solid {g_color}; border-radius: 5px; padding: 5px; text-align: center;">
                    <strong style="color:{g_color}; font-size: 0.8rem;">🌱 Yeşil Skor</strong><br>
                    <span style="font-size: 1.5rem; font-weight: bold; color:{g_color};">{g_score:.1f}/10</span>
                </div>
                """, unsafe_allow_html=True)
            
            if g_note:
                st.caption(f"**Çevresel Analiz:** {g_note}")

            st.divider()
            if 'Solubility' in preds:
                sol_val = preds['Solubility']
                solvents, partials = get_soluble_solvents(sol_val)
                
                st.markdown("Tahmini Çözünürlük Analizi")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.info(f"**Çözünmesi Beklenenler:**")
                    if solvents:
                        for s in solvents:
                            st.markdown(f"- ✅ {s}")
                    else:
                        st.warning("Bu polimer çok dirençli görünüyor (veya çok özel bir çözücü gerektiriyor).")
                
                with c2:
                    st.warning(f"**Şişme / Zor Çözünme Beklenenler:**")
                    if partials:
                        for s in partials:
                            st.markdown(f"- ⚠️ {s}")
                    else:
                        st.write("-")
                
                st.caption(f"*Analiz, 'Benzer Benzeri Çözer' ilkesine göre Polimer (δ={sol_val:.1f}) ve Çözücü arasındaki Hildebrand farkına dayanır.*")
            cols = st.columns(3)
            for idx, prop in enumerate(ALL_PROPS):
                with cols[idx % 3]:
                    is_active = prop in saved_active_props
                    target_val = saved_targets.get(prop, '-')
                    target_text = f"Hedef: {target_val}" if is_active else "Takip Dışı"
                    border_color = "#2ecc71" if is_active else "#95a5a6"
                    pred_value = preds[prop]
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid {border_color};">
                        <small>{prop}</small><br>
                        <h3 style="margin:0; padding:0;">{pred_value:.2f}</h3>
                        <small style="opacity:0.7">{target_text}</small>
                    </div>
                    """, unsafe_allow_html=True)
            st.divider()
            st.subheader("Hedef Uyumluluk Analizi")
            if len(saved_active_props) >= 3:
                    fig = create_radar_chart(preds, saved_targets, saved_active_props, ranges)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                    st.info("Radar grafiği için en az 3 özellik (Örn: Tg, LOI, CTE) seçmelisiniz.")
                    st.progress(100) 
        with tab2:
            col_2d, col_3d = st.columns(2)
            with col_2d:
                st.subheader("2D Yapı (Teknik Çizim)")
                img = draw_2d_molecule(best_poly_data['smiles'])
                if img:
                    st.image(img, width=400)
                st.caption("SMILES Kodu:")
                st.code(best_poly_data['smiles'], language="text")
                st.caption("SELFIES Kodu:")
                selfies_str = smiles_to_selfies_safe(best_poly_data['smiles'])
                if selfies_str:
                    st.code(selfies_str, language="text")
                else:
                    st.warning("SELFIES formatına dönüştürülemedi.")
            with col_3d:
                st.subheader("3D Konformasyon")
                view, reason = make_3d_view_with_reason(best_poly_data["smiles"])
                if view:
                    showmol(view, height=400, width=400)
                else:
                    st.warning(f"3D Model oluşturulamadı: {reason}")
            
            is_avail, cid, name = check_pubchem_availability(best_poly_data['smiles'])
            if is_avail:
                 st.info(f"Bu molekül PubChem'de kayıtlı: **{name}** (CID: {cid})")
            st.divider()
            st.subheader("Özgünlük Analizi (Novelty Search)")
            
            similarity_score, similar_smi = calculate_novelty_optimized(best_poly_data['smiles'], reference_smiles)
            
            c1, c2 = st.columns([1, 3])
            
            with c1:
                st.metric("Eğitim Setine Benzerlik", f"%{similarity_score*100:.1f}")
                
            with c2:
                if similarity_score > 0.99:
                    st.error(f" **Kopya:** Yapay zeka eğitim setindeki bir veriyi ezberlemiş.")
                    st.code(f"Benzer Kayıt: {similar_smi}")
                elif similarity_score > 0.85:
                    st.warning(f"**Türev:** Eğitim setindeki bir yapıya çok benziyor.")
                    with st.expander("Benzer Yapıyı Gör"):
                        st.code(similar_smi)
                else:
                    st.success(f" **KEŞİF:** Bu yapı eğitim setinde YOK! Tamamen özgün bir tasarım.")
                    st.caption(f"En yakın benzerlik sadece %{similarity_score*100:.1f} oranında.")
            
            st.progress(similarity_score)
            st.caption("*Benzerlik, Tanimoto İndeksi (Morgan Fingerprints) kullanılarak hesaplanmıştır.*")

        with tab3:
            st.subheader("Genetik Algoritma Performans Raporu")
            
            if 'best_fitness' in history and len(history['best_fitness']) > 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                gens = range(len(history['best_fitness']))
                
                ax1.plot(gens, history['best_fitness'], label='En İyi Birey (Best)', color='green', linewidth=2)
                ax1.plot(gens, history['avg_fitness'], label='Popülasyon Ortalaması (Avg)', color='blue', linestyle='--', alpha=0.7)
                ax1.set_ylabel('Hata Skoru')
                ax1.set_title('Yakınsama Analizi (Convergence)', fontweight='bold')
                ax1.legend()
                ax1.grid(True, which='both', linestyle='--', alpha=0.5)
                
                ax2.plot(gens, history['diversity'], label='Standart Sapma (Diversity)', color='red', linewidth=2)
                ax2.fill_between(gens, history['diversity'], color='red', alpha=0.1)
                ax2.set_ylabel('Çeşitlilik (Std Dev)')
                ax2.set_xlabel('Jenerasyon')
                ax2.set_title('Popülasyon Çeşitliliği (Diversity)', fontweight='bold')
                ax2.legend()
                ax2.grid(True, which='both', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("""
                **Bu Grafikler Nasıl Okunur?**
                * **Yakınsama (Üst):** Yeşil çizgi sürekli düşmeli ve bir noktada yataylaşmalıdır (Plateau). Mavi çizgi yeşile çok yaklaşırsa popülasyon "öğrenmiş" demektir.
                * **Çeşitlilik (Alt):** Kırmızı çizginin sıfıra çok hızlı düşmemesi gerekir. Sıfıra düşerse model "Erken Yakınsama (Premature Convergence)" tuzağına düşmüştür; yani arama uzayını yeterince taramadan bir sonuca saplanmıştır.
                """)
            else:
                st.warning("Henüz grafik çizilecek veri yok.")
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
            st.header("💾 Raporlama")
            st.markdown("Proje verilerini CSV veya detaylı PDF raporu olarak dışa aktarabilirsiniz.")
            
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
                    label="Veri Setini İndir (.csv)",
                    data=csv_best,
                    file_name="polimer_data.csv",
                    mime="text/csv"
                )
            
            st.divider()
            
            st.subheader("PDF Raporu")
            st.info("Bu rapor; tüm tahminleri, molekül yapısını, AI yorumlarını ve varsa sentez planını içerir.")

            gen_ai_analysis = st.session_state.get('ai_analysis', "Genel AI analizi yapilmadi.")
            
            manual_retro = st.session_state.get('retro_manual_text', "Otomatik ayristirma verisi yok (Retrosentez sekmesini ziyaret edin).")
            ai_retro = st.session_state.get('ai_retro_text', "AI sentez recetesi olusturulmadi.")
            
            full_retro_info = manual_retro + "\n\n--- AI Sentez Notlari ---\n" + ai_retro

            if st.button("PDF Raporu Oluştur", type="primary", use_container_width=True):
                with st.spinner("Rapor derleniyor..."):
                    pdf_data = create_pdf_report(
                        best_poly_data, 
                        saved_targets, 
                        saved_active_props, 
                        gen_ai_analysis, 
                        full_retro_info
                    )
                    
                    st.success("Rapor hazır!")
                    st.download_button(
                        label="📥 PDF Dosyasını İndir",
                        data=pdf_data,
                        file_name="PolimerX_Final_Raporu.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        with tab5:
            st.subheader("ChatBot")
            st.markdown("Polimer yapısı ve tahmin edilen özellikler hakkında detaylı kimyasal yorum almak için AI destekli ChatBot'u kullanın.")
            if not api_key:
                st.info("Bu polimer hakkında detaylı kimyasal yorum almak için sol menüden **Google Gemini API Key** girmelisiniz.")
                st.markdown("Ücretsiz API Key Almak İçin Tıkla](https://aistudio.google.com/app/apikey)")
            else:
                if st.button("Polimeri Analiz Et", type="primary"):
                    analysis_result = get_ai_interpretation(
                        api_key, 
                        best_poly_data['smiles'], 
                        best_poly_data['preds'], 
                        saved_targets, 
                        saved_active_props
                    )
                    st.markdown(analysis_result)
                    
                    st.session_state['ai_analysis'] = analysis_result
                
                elif 'ai_analysis' in st.session_state:
                    st.markdown(st.session_state['ai_analysis'])
        with tab6:
            st.header("Retrosentez Analizi")
            
            target_smiles = best_poly_data['smiles']
            
            st.subheader("1. Yapısal Ayrıştırma")
            retro_results = decompose_polymer(target_smiles)
            
            monomer_info_text = "Otomatik analiz yapilmadi."
            
            if retro_results:
                best_route = retro_results[0]
                
                monomer_info_text = f"Yontem: {best_route['type']}\nMekanizma: {best_route['mechanism']}\n"
                
                st.info(f"**Algılanan Sentez Türü:** {best_route['type']}")
                st.write(f"**Mekanizma:** {best_route['mechanism']}")
                
                st.markdown("**Olası Başlangıç Monomerleri:**")
                img_retro = draw_retrosynthesis_grid(best_route['monomers'])
                if img_retro: st.image(img_retro)

                st.markdown("#### Ticari Bulunabilirlik")
                found_monomers = []
                for i, m in enumerate(best_route['monomers']):
                    col_code, col_check = st.columns([3, 1])
                    with col_code:
                        st.code(f"Monomer {i+1}: {m}")
                    with col_check:
                        if st.button(f"Kontrol #{i+1}", key=f"chk_{i}"):
                            is_avail, cid, name = check_commercial_availability(m)
                            if is_avail:
                                st.success(f"Var: {name}")
                                found_monomers.append(name)
                            else:
                                st.error("Ticari kayit yok")
                
                if found_monomers:
                    monomer_info_text += f"\nTicari Kaydi Olanlar: {', '.join(found_monomers)}"
                else:
                    monomer_info_text += "\nTicari kayit kontrolu yapilmadi veya bulunamadi."
            
            else:
                st.warning("Yapısal ayrıştırma başarısız.")
                monomer_info_text = "Yapısal ayrıştırma başarısız."

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

            st.subheader("2. T5-Model Tahmini ")
            st.caption("Eğittiğimiz model, moleküler yapıyı analiz ederek monomerleri tahmin ediyor.")

            if st.button("Monomerleri Tahmin Et", type="primary"):
                with st.spinner("Yapay zeka düşünüyor..."):
                    prediction = predict_monomers_local(best_poly_data['smiles'])
                    
                    st.success("Monomer Tahmini Başarılı!")
                    
                    st.markdown(f"""
                    <div style="background-color:#e8f5e9; padding:15px; border-radius:10px; border:1px solid #4CAF50;">
                        <h4 style="color:#2e7d32; margin:0;">Önerilen Monomerler:</h4>
                        <code style="font-size:1.1em; color:#1b5e20; background-color:#e8f5e9;">{prediction}</code>
                    </div>
                    """, unsafe_allow_html=True)
                    print("Predicted monomers:", prediction)
                    monomers_list = prediction.split(' . ') 
                    img_retro = draw_retrosynthesis_grid(monomers_list)
                    if img_retro:
                        st.image(img_retro, caption="Modelin Önerdiği Yapı Taşları")
                    st.session_state['retro_manual_text'] = f"AI Tahmini: {prediction}"



