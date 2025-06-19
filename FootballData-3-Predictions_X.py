"""
Football Predictions Model - Versione 2.1 (Modificata per filtrare DNB)
Questa versione verifica se esiste il file predictions.csv ed aggiorna solo RIS, Stato ed ETop#.
Nel caso non esista il file. lo crea nuovamente.
"""
import pandas as pd
import numpy as np
import math
import os
import argparse
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import logging
from functools import lru_cache

# Configurazione generale
class Config:
    MAX_GOALS = 6
    TOP_BETS = 5
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Dati')
    #REQUIRED_ODDS_COLUMNS = ['DNB1', 'DNB2']  # Nuova costante aggiunta

# Configurazione logging avanzata
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Handler per console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Precalcoli per Poisson
x, y = np.indices((Config.MAX_GOALS, Config.MAX_GOALS))
factorials = np.array([math.factorial(k) for k in range(Config.MAX_GOALS)])
x_fact = factorials[x]
y_fact = factorials[y]

@lru_cache(maxsize=None)
def load_condition_map() -> dict:
    base_conditions = {
        "1": lambda h, a: h > a,
        "X": lambda h, a: h == a,
        "2": lambda h, a: h < a,
        "1X": lambda h, a: h >= a,
        "X2": lambda h, a: h <= a,
        "12": lambda h, a: h != a,
        "O15": lambda h, a: h + a >= 2,
        "U15": lambda h, a: h + a < 2,
        "O25": lambda h, a: h + a >= 3,
        "U25": lambda h, a: h + a < 3,
        "O35": lambda h, a: h + a >= 4,
        "U35": lambda h, a: h + a < 4,
        "GOL": lambda h, a: (h >= 1) & (a >= 1),
        "NOG": lambda h, a: (h == 0) | (a == 0)
    }
    
    special_combinations = {
        "GOL+O25": lambda h, a: (h >= 1) & (a >= 1) & (h + a >= 3),
        "GOL+U35": lambda h, a: (h >= 1) & (a >= 1) & (h + a < 4),
        "NOG+O25": lambda h, a: ((h == 0) | (a == 0)) & (h + a >= 3),
        "NOG+U35": lambda h, a: ((h == 0) | (a == 0)) & (h + a < 4),
        "GOL+1X": lambda h, a: (h >= 1) & (a >= 1) & (h >= a),
        "GOL+12": lambda h, a: (h >= 1) & (a >= 1) & (h != a),
        "GOL+X2": lambda h, a: (h >= 1) & (a >= 1) & (h <= a),
        "MG1-3": lambda h, a: (1 <= h + a) & (h + a < 4),
        "MG2-5": lambda h, a: (2 <= h + a) & (h + a < 6),
        "MG3-5": lambda h, a: (3 <= h + a) & (h + a < 6),
        "NOG+1X": lambda h, a: ((h == 0) | (a == 0)) & (h >= a),
        "NOG+12": lambda h, a: ((h == 0) | (a == 0)) & (h != a),
        "NOG+X2": lambda h, a: ((h == 0) | (a == 0)) & (h <= a)
    }
    
    dynamic_combinations = {
        f"{outcome}+{total}": lambda h, a, o=outcome, t=total: 
            base_conditions[o](h, a) & base_conditions[t](h, a)
        for outcome in ["1", "X", "2", "1X", "X2", "12"]
        for total in ["O15", "U15", "O25", "U25", "O35", "U35"]
    }
    
    return {**base_conditions, **special_combinations, **dynamic_combinations}

condition_map = load_condition_map()

def calculate_poisson_probs(lambda_home, lambda_away):
    """Calcola le probabilità Poissoniane per tutti i risultati"""
    k = np.exp(-lambda_home) * (lambda_home ** x) / x_fact
    m = np.exp(-lambda_away) * (lambda_away ** y) / y_fact
    return k * m

def get_top_probabilities(home_xgv2, away_xgv2):
    """Calcola le migliori scommesse e punteggi esatti con gestione NaN"""
    if np.isnan(home_xgv2) or np.isnan(away_xgv2):
        return {'Top_Bets': [], 'Top_Scores': []}
    
    probs = calculate_poisson_probs(home_xgv2, away_xgv2)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    
    combo_probs = {
        name: np.sum(probs * cond_map(x, y)) 
        for name, cond_map in condition_map.items()
    }
    
    score_probs = {
        f"{i}-{j}": probs[i, j]
        for i in range(Config.MAX_GOALS)
        for j in range(Config.MAX_GOALS)
    }
    
    return {
        'Top_Bets': sorted(combo_probs.items(), key=lambda x: -x[1])[:Config.TOP_BETS],
        'Top_Scores': sorted(score_probs.items(), key=lambda x: -x[1])[:Config.TOP_BETS]
    }

# Funzioni originali per il modello xG
def get_last_n_matches(df, team, division, n=6):
    return df[(df['Div'] == division) & ((df['Home'] == team) | (df['Away'] == team))]\
           .sort_values('Date', ascending=False).head(n)

def prepare_training_data(team, division, played, metric_home, metric_away, n=6):
    df_team = get_last_n_matches(played, team, division, n)
    values = []
    for _, row in df_team.iterrows():
        if row['Home'] == team:
            values.append(row[metric_home])
        elif row['Away'] == team:
            values.append(row[metric_away])
    return np.array(values).reshape(-1, 1)

def predict_metric(team, division, played, metric_home, metric_away, n=6):
    X = prepare_training_data(team, division, played, metric_home, metric_away, n)
    valid_idx = ~np.isnan(X.flatten())
    X_valid = X[valid_idx].reshape(-1, 1)
    
    if len(X_valid) < 2:
        logging.warning(f"Dati insufficienti per {team} ({metric_home}/{metric_away})")
        return 0.0  # Restituisci 0 invece di NaN
    
    y = X_valid.flatten()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    dummy = np.arange(len(y)).reshape(-1, 1)
    model.fit(dummy, y)
    return model.predict([[len(y)]])[0]

def build_training_dataset(played, team, division, n=6):
    df_team = get_last_n_matches(played, team, division, n)
    data, targets = [], []
    for _, row in df_team.iterrows():
        if row['Home'] == team:
            features = {
                'S': row['HS'], 'T': row['HST'], 'C': row['HC'],
                'F': row['HF'], 'FK': row['FKH'], 'Y': row['HY'],
                'GC': row['GFTA'], 'SC': row['AS'], 'TC': row['AST'], 'CC': row['AC'],
                'GP': row['GFTH']  # Aggiunta della feature GP
            }
            target = row['GFTH']
        else:
            features = {
                'S': row['AS'], 'T': row['AST'], 'C': row['AC'],
                'F': row['AF'], 'FK': row['FKA'], 'Y': row['AY'],
                'GC': row['GFTH'], 'SC': row['HS'], 'TC': row['HST'], 'CC': row['HC'],
                'GP': row['GFTA']  # Aggiunta della feature GP
            }
            target = row['GFTA']
        data.append(features)
        targets.append(target)
    return pd.DataFrame(data), np.array(targets)

def get_dynamic_weights(played, team, division, n=6):
    """Calcola pesi dinamici o recupera da cache se esistenti"""
    # Controlla se i pesi sono già stati calcolati
    if Config.existing_weights:
        cached_weights = Config.existing_weights.get((team, division))
        if cached_weights is not None:
            return cached_weights
            
    df_features, y_targets = build_training_dataset(played, team, division, n)
    
    if df_features.empty:
        return {key: 0.0 for key in ['S','T','C','F','FK','Y','GP','GC','SC','TC','CC']}
    
    # Pulizia dati
    df_combined = df_features.copy()
    df_combined['target'] = y_targets
    df_combined = df_combined.dropna()
    
    if df_combined.empty:
        return {key: 0.0 for key in ['S','T','C','F','FK','Y','GP','GC','SC','TC','CC']}
    
    # Addestramento modello
    X_clean = df_combined.drop('target', axis=1)
    y_clean = df_combined['target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_clean, y_clean)
    
    # Calcolo importanze
    importances = model.feature_importances_
    total = importances.sum()
    
    epsilon = 1e-10
    if total < epsilon:
        return {col: 0.0 for col in X_clean.columns}
    
    # Normalizzazione e logging
    weights = {col: imp/(total + epsilon) for col, imp in zip(X_clean.columns, importances)}
    
    if sum(weights.values()) >= 0.01:
        weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in weights.items()])
        logging.info(
            f"🔧 [PESI] {team.ljust(20)} | {str(division).ljust(15)} | {weights_str}"
        )
    
    return weights

def calculate_expected_goals_dynamic(metrics, weights, exclude_features=None):
    """Versione ottimizzata con controllo errori"""
    exclude_features = exclude_features or []
    try:
        return sum(
            metrics.get(f, 0) * weights.get(f, 0)
            for f in weights.keys()
            if f not in exclude_features
        )
    except Exception as e:
        logging.error(f"Errore calcolo xG: {str(e)}")
        return 0.0

# Aggiungi questa nuova funzione dopo la funzione calculate_expected_goals_dynamic
def calculate_xg_v2(historical_data, current_metrics, team_weights, is_home=True):
    """Calcola xGv2 utilizzando i pesi dinamici e i dati storici"""
    try:
        required_metrics = ['S','T','C','F','FK','Y','GP','GC','SC','TC','CC']
        for metric in required_metrics:
            value = current_metrics.get(metric, np.nan)
            if pd.isna(value) or value < 0:
                logging.warning(f"Metrica {metric} non valida: {value}")
                return np.nan

        target_col = 'GFTH' if is_home else 'GFTA'
        total_shots = historical_data['HS' if is_home else 'AS'].sum()
        total_target = historical_data['HST' if is_home else 'AST'].sum()
        total_corner = historical_data['HC' if is_home else 'AC'].sum()
        total_goals = historical_data[target_col].sum()

        # 1. Gestione divisione per zero
        conversion_terms = []
        if total_target > 0:
            conversion_terms.append(total_goals / total_target)
        if total_shots > 0:
            conversion_terms.append(total_goals / total_shots)
        if total_corner > 0:
            conversion_terms.append(total_goals / total_corner)

        # 2. Calcolo realistico (media invece che somma)
        conversion_rate = np.mean(conversion_terms) if conversion_terms else 0

        weighted_sum = sum(
            current_metrics.get(feature, 0) * team_weights.get(feature, 0)
            for feature in team_weights.keys()
        )

        return (weighted_sum * conversion_rate)

    except Exception as e:
        logging.error(f"Errore xGv2: {str(e)}")
        return np.nan
        
def compute_RIS(row):
    """Calcola la colonna RIS dal DataFrame originale"""
    gft_h = row.get("GFTH", "")
    gft_a = row.get("GFTA", "")
    ght_h = row.get("GHTH", "")
    ght_a = row.get("GHTA", "")

    # Gestione valori mancanti/NaN
    gft_h = "" if pd.isna(gft_h) else int(gft_h)
    gft_a = "" if pd.isna(gft_a) else int(gft_a) 
    ght_h = "" if pd.isna(ght_h) else int(ght_h)
    ght_a = "" if pd.isna(ght_a) else int(ght_a)

    if all(val == "" for val in [gft_h, gft_a, ght_h, ght_a]):
        return "-"
    
    return f"{gft_h}-{gft_a} ({ght_h}-{ght_a})"

def calculate_stato(row):
    """Determina lo stato della partita con gestione avanzata degli errori"""
    try:
        # Converti la data e l'ora in formato datetime
        date_str = str(row['Date']).split()[0]  # Estrai solo la parte della data se contiene timestamp
        time_str = str(row['Time']).split()[0]   # Estrai solo ore:minuti
        
        match_time = pd.to_datetime(
            f"{date_str} {time_str}", 
            format="%Y-%m-%d %H:%M", 
            errors='coerce'
        )
        
        if pd.isnull(match_time):
            return "Formato Data Non Valido"
        
        now = datetime.now()
        
        if now > match_time + timedelta(hours=2):
            return "Finita"
        elif now >= match_time:
            return "In Corso"
        else:
            return "LIVE"
            
    except Exception as e:
        logging.warning(f"Errore stato partita {row['Home']}-{row['Away']}: {str(e)}")
        return "Errore Data"
        
def calculate_total_outcome(pred_h, pred_a, actual_h, actual_a):
    """Gestione valori mancanti e non numerici"""
    try:
        if pd.isna(actual_h) or pd.isna(actual_a) or pd.isna(pred_h) or pd.isna(pred_a):
            return "-"
        
        # Conversione sicura a float
        pred_h = float(pred_h)
        pred_a = float(pred_a)
        actual_h = float(actual_h)
        actual_a = float(actual_a)
        
        total_pred = pred_h + pred_a
        total_actual = actual_h + actual_a
        
        if total_pred > total_actual:
            return "SUP"
        elif total_pred < total_actual:
            return "INF"
        return "="
    except:
        return "-"
        
def calculate_top_outcome(condition, gtfh, gtfa):
    """Verifica se una condizione TOP# è soddisfatta"""
    if pd.isna(gtfh) or pd.isna(gtfa):
        return "N/A"
    
    condition_func = condition_map.get(condition, lambda h, a: False)
    return "W" if condition_func(gtfh, gtfa) else "L"

ODDS_CACHE = {}

def load_odds_report(match_date):
    """Carica l'OddsReport senza controllo colonne obbligatorie"""
    date_str = match_date.strftime('%Y%m%d')
    if date_str in ODDS_CACHE:
        return ODDS_CACHE[date_str]
    
    file_path = os.path.join(Config.DATA_DIR, f"{date_str}_OddsReport.csv")
    
    try:
        df = pd.read_csv(file_path)
        ODDS_CACHE[date_str] = df
        return df
    except FileNotFoundError:
        logging.warning(f"OddsReport non trovato: {file_path}")
        ODDS_CACHE[date_str] = None
        return None
    except Exception as e:
        logging.error(f"Errore nel caricare OddsReport: {e}")
        return None

# SOSTITUIRE la funzione is_valid_dnb con questa versione corretta
#def is_valid_dnb(match_odds):
#    """Verifica che DNB1 e DNB2 siano validi e > 0 con gestione tipi avanzata"""
#    try:
#        # Se match_odds è vuoto, salta
#        if match_odds.empty:
#            return False
#            
#        dnb1_val = match_odds['DNB1'].iloc[0]
#        dnb2_val = match_odds['DNB2'].iloc[0]
#        
#        # Se entrambi i valori sono vuoti o stringhe vuote, salta
#        if dnb1_val == "" and dnb2_val == "":
#            return False
#            
#        # Conversione a float con gestione di stringhe e virgole
#        try:
#            dnb1 = float(str(dnb1_val).replace(',', '.')) if dnb1_val != "" else 0
#            dnb2 = float(str(dnb2_val).replace(',', '.')) if dnb2_val != "" else 0
#        except (ValueError, TypeError):
#            return False
#            
#        # Controllo valori positivi
#        return dnb1 > 0 and dnb2 > 0
#    except (IndexError, KeyError):
#        return False

def process_schedule(played, schedule, n=6, exclude_features=None):
    exclude_features = exclude_features or ['FK', 'Y', 'GP', 'GC', 'SC', 'TC', 'CC']
    report = []
    
    for _, match in tqdm(schedule.iterrows(), total=len(schedule)):
        # Carica l'OddsReport senza filtrare per DNB
        odds_df = load_odds_report(match['Date'])
        match_odds = pd.DataFrame()
        
        if odds_df is not None:
            mask = (
                (odds_df['Div'] == match['Div']) &
                (odds_df['Home'] == match['Home']) &
                (odds_df['Away'] == match['Away'])
            )
            match_odds = odds_df[mask]
        
        #if match_odds.empty or not is_valid_dnb(match_odds):
        #    logging.info(f"Salto {match['Home']} vs {match['Away']} - DNB non valido")
        #    continue
        # MODIFICARE la sezione nel process_schedule come segue
        # (dopo la creazione di match_odds)
        
        #if match_odds.empty or not is_valid_dnb(match_odds):
        #    # Log dettagliato con valori reali
        #    try:
        #        dnb1 = match_odds['DNB1'].iloc[0] if not match_odds.empty else "N/A"
        #        dnb2 = match_odds['DNB2'].iloc[0] if not match_odds.empty else "N/A"
        #        logging.info(f"Salto {match['Home']} vs {match['Away']} - DNB non valido: DNB1={dnb1}, DNB2={dnb2}")
        #    except:
        #        logging.info(f"Salto {match['Home']} vs {match['Away']} - DNB non valido (errore estrazione)")
        #    continue
            
        # Resto del codice originale invariato
        division = match['Div']
        home_team = match['Home']
        away_team = match['Away']
        match_time = match['Time']
        country = match['Country']
        league = match['League']
        
        # Predizione metriche
        h_metrics = {
            'S': predict_metric(home_team, division, played, 'HS', 'AS', n),
            'T': predict_metric(home_team, division, played, 'HST', 'AST', n),
            'C': predict_metric(home_team, division, played, 'HC', 'AC', n),
            'F': predict_metric(home_team, division, played, 'HF', 'AF', n),
            'FK': predict_metric(home_team, division, played, 'FKH', 'FKA', n),
            'Y': predict_metric(home_team, division, played, 'HY', 'AY', n),
            'GP': predict_metric(home_team, division, played, 'GFTH', 'GFTA', n),
            'GC': predict_metric(home_team, division, played, 'GFTA', 'GFTH', n),
            'SC': predict_metric(home_team, division, played, 'AS', 'HS', n),
            'TC': predict_metric(home_team, division, played, 'AST', 'HST', n),
            'CC': predict_metric(home_team, division, played, 'AC', 'HC', n)
        }
        
        a_metrics = {
            'S': predict_metric(away_team, division, played, 'AS', 'HS', n),
            'T': predict_metric(away_team, division, played, 'AST', 'HST', n),
            'C': predict_metric(away_team, division, played, 'AC', 'HC', n),
            'F': predict_metric(away_team, division, played, 'AF', 'HF', n),
            'FK': predict_metric(away_team, division, played, 'FKA', 'FKH', n),
            'Y': predict_metric(away_team, division, played, 'AY', 'HY', n),
            'GP': predict_metric(away_team, division, played, 'GFTA', 'GFTH', n),
            'GC': predict_metric(away_team, division, played, 'GFTH', 'GFTA', n),
            'SC': predict_metric(away_team, division, played, 'HS', 'AS', n),
            'TC': predict_metric(away_team, division, played, 'HST', 'AST', n),
            'CC': predict_metric(away_team, division, played, 'HC', 'AC', n)
        }
        
        # Calcolo pesi
        home_weights = get_dynamic_weights(played, home_team, division, n)
        away_weights = get_dynamic_weights(played, away_team, division, n)
        # Recupera dati storici ultime 6 partite
        home_historical = get_last_n_matches(played, home_team, division, 6)
        away_historical = get_last_n_matches(played, away_team, division, 6)

        # Modifica 3: Aggiorna il processo principale per saltare partite con xGv2 non validi
        home_xgv2 = calculate_xg_v2(
            home_historical, 
            h_metrics, 
            home_weights, 
            is_home=True
        )
        away_xgv2 = calculate_xg_v2(
            away_historical, 
            a_metrics, 
            away_weights, 
            is_home=False
        )
        # Se entrambi gli xGv2 sono 0, la partita non viene riportata nel report
        if home_xgv2 == 0 or away_xgv2 == 0:
            logging.info(f"Partita saltata: {home_team} vs {away_team} - xGv2 pari a zero")
            continue
            
        # Aggiungi questo controllo dopo il calcolo degli xGv2
        if pd.isna(home_xgv2) or pd.isna(away_xgv2):
            logging.warning(f"Partita saltata: {home_team} vs {away_team} - Dati xGv2 non validi")
            continue
        
        # Calcolo xG
        home_weights = get_dynamic_weights(played, home_team, division, n)
        away_weights = get_dynamic_weights(played, away_team, division, n)
        home_xg = calculate_expected_goals_dynamic(h_metrics, home_weights, exclude_features)
        away_xg = calculate_expected_goals_dynamic(a_metrics, away_weights, exclude_features)
        
        # Calcolo EG_H e EG_A
        eg_h = np.sqrt(h_metrics['GP'] * a_metrics['GC']) if not pd.isna(h_metrics['GP']) and not pd.isna(a_metrics['GC']) else 0.0
        eg_a = np.sqrt(a_metrics['GP'] * h_metrics['GC']) if not pd.isna(a_metrics['GP']) and not pd.isna(h_metrics['GC']) else 0.0
        et_s = np.sqrt((h_metrics['S'] * a_metrics['SC']) + (a_metrics['S'] * h_metrics['SC']))
        et_t = np.sqrt((h_metrics['T'] * a_metrics['TC']) + (a_metrics['T'] * h_metrics['TC']))
        et_c = np.sqrt((h_metrics['C'] * a_metrics['CC']) + (a_metrics['C'] * h_metrics['CC']))
        et_y = np.sqrt((h_metrics['Y'] + a_metrics['Y'] ))
        
        # Calcolo probabilità
        prob_results = get_top_probabilities(home_xgv2, away_xgv2)
        
        # Recupera risultato reale
        actual_h = match.get('GFTH', np.nan)
        actual_a = match.get('GFTA', np.nan)
        actual_score = f"{int(actual_h)}-{int(actual_a)}" if not pd.isna(actual_h) and not pd.isna(actual_a) else None

        # Inizializza tutte le ER# a "-"
        er_entries = {f'ER#{i}': "-" for i in range(1, 6)}
        
        # Popola ER# solo se il risultato reale è disponibile
        if actual_score:
            for idx, (score, _) in enumerate(prob_results['Top_Scores'][:5], 1):
                er_entries[f'ER#{idx}'] = "W" if score == actual_score else "-"

        # Arrotondamento alle unità per le colonne specificate
        h_metrics_int = {
            k: int(round(v)) if not pd.isna(v) else "-"
            for k, v in h_metrics.items()
            }
        a_metrics_int = {
            k: int(round(v)) if not pd.isna(v) else "-"
            for k, v in a_metrics.items()
            }
        
        # Controllo presenza valori "-"
        #if any(v == "-" for v in h_metrics_int.values()) or any(v == "-" for v in a_metrics_int.values()):
        #    logging.warning(f"Partita saltata: {home_team} vs {away_team} - Dati incompleti")
        #    continue
                
        # Recupera valori reali dallo schedule
        actual = {
            'H_S': match.get('HS', np.nan),
            'H_T': match.get('HST', np.nan),
            'H_C': match.get('HC', np.nan),
            'H_F': match.get('HF', np.nan),
            'H_FK': match.get('FKH', np.nan),
            'H_Y': match.get('HY', np.nan),
            'H_GP': match.get('GFTH', np.nan),
            'A_S': match.get('AS', np.nan),
            'A_T': match.get('AST', np.nan),
            'A_C': match.get('AC', np.nan),
            'A_F': match.get('AF', np.nan),
            'A_FK': match.get('FKA', np.nan),
            'A_Y': match.get('AY', np.nan),
            'A_GP': match.get('GFTA', np.nan),
            'GTFH': match.get('GFTH', np.nan),
            'GTFA': match.get('GFTA', np.nan)
        }
        
        # Dopo h_metrics_int e a_metrics_int
        if any(v == "-" for v in h_metrics_int.values()) or any(v == "-" for v in a_metrics_int.values()):
            logging.warning(f"Partita saltata: {home_team} vs {away_team} - Dati incompleti")
            continue  # Salta questa iterazione del loop
        
        # Formattazione output
        record = {
            'Div': division,
            'Date': match['Date'].strftime('%Y-%m-%d'),
            'Time': match_time,
            'Country': country,
            'League': league,
            'Home': home_team,
            'Away': away_team,
            'RIS': compute_RIS(match),
            'Stato': calculate_stato(match),
            'xG_Home': round(home_xg, 2),
            'xG_Away': round(away_xg, 2),
            'HxGv2': round(home_xgv2, 2),  # NUOVO
            'AxGv2': round(away_xgv2, 2),   # NUOVO
            **{f'H_{k}': (v if v != "-" else "-") for k, v in h_metrics_int.items()},
            **{f'A_{k}': (v if v != "-" else "-") for k, v in a_metrics_int.items()},
            'G_H': round(eg_h, 2) if not pd.isna(eg_h) else "-",
            'G_A': round(eg_a, 2) if not pd.isna(eg_a) else "-",
            'T_S': round(et_s, 0) if not pd.isna(et_s) else "-",
            'T_T': round(et_t, 0) if not pd.isna(et_t) else "-",
            'T_C': round(et_c, 0) if not pd.isna(et_c) else "-",
            'T_Y': round(et_y, 0) if not pd.isna(et_y) else "-",
            'ET_S': calculate_total_outcome(h_metrics['S'], a_metrics['S'], actual['H_S'], actual['A_S']),
            'ET_T': calculate_total_outcome(h_metrics['T'], a_metrics['T'], actual['H_T'], actual['A_T']),
            'ET_C': calculate_total_outcome(h_metrics['C'], a_metrics['C'], actual['H_C'], actual['A_C']),
            'ET_Y': calculate_total_outcome(h_metrics['Y'], a_metrics['Y'], actual['H_Y'], actual['A_Y'])
            }
        
        # Aggiunta top bets
        for idx, (bet, prob) in enumerate(prob_results['Top_Bets'][:Config.TOP_BETS], 1):
            record[f'TOP#{idx}'] = bet
            #record[f'Q#{idx}'] = round(1/(prob + 1e-10), 2)  # Evita divisione per zero
            min_prob = 1e-7  # Imposta un valore minimo ragionevole
            safe_prob = max(prob, min_prob)
            record[f'Q#{idx}'] = round(1/safe_prob, 2) if safe_prob > 0 else 1000.0  # Valore massimo 1000
            
        # Aggiunta top scores
        for idx, (score, prob) in enumerate(prob_results['Top_Scores'][:Config.TOP_BETS], 1):
            record[f'TopR#{idx}'] = score
            record[f'P_TopR#{idx}'] = round(1/(prob + 1e-10), 2)
        
        # Riempimento colonne mancanti
        for i in range(1, Config.TOP_BETS+1):
            for prefix in ['TOP#', 'Q#', 'TopR#', 'P_TopR#']:
                key = f'{prefix}{i}'
                if key not in record:
                    record[key] = np.nan
        # Carica OddsReport per la data della partita
        odds_df = load_odds_report(match['Date'])
        
        # Inizializza le colonne Odd# a NaN
        for i in range(1, 6):
            record[f'Odd#{i}'] = np.nan
        
        if not match_odds.empty:
            # Estrai le quote solo se il DataFrame non è vuoto
            for i in range(1, 6):
                bet_type = record.get(f'TOP#{i}', '')
                if bet_type and bet_type in match_odds.columns:
                    # Verifica che ci siano valori disponibili
                    if len(match_odds[bet_type].values) > 0:
                        record[f'Odd#{i}'] = match_odds[bet_type].values[0]
        else:
            logging.debug(f"Nessun OddsReport per {match['Date'].strftime('%Y-%m-%d')}")      
                
        # Calcola esiti per le metriche
        #outcome_entries = {
        #    # Home outcomes
        #    'EH_S': calculate_metric_outcome(h_metrics['S'], actual['H_S']),
        #    'EH_T': calculate_metric_outcome(h_metrics['T'], actual['H_T']),
        #    'EH_C': calculate_metric_outcome(h_metrics['C'], actual['H_C']),
        #    'EH_F': calculate_metric_outcome(h_metrics['F'], actual['H_F']),
        #    'EH_FK': calculate_metric_outcome(h_metrics['FK'], actual['H_FK']),
        #    'EH_Y': calculate_metric_outcome(h_metrics['Y'], actual['H_Y']),
        #    'EH_GP': calculate_metric_outcome(h_metrics['GC'], actual['H_GP']),
        #    'EH_GC': calculate_metric_outcome(a_metrics['GC'], actual['A_GP']),
        #    'EH_SC': calculate_metric_outcome(a_metrics['S'], actual['A_S']),
        #    'EH_TC': calculate_metric_outcome(a_metrics['T'], actual['A_T']),
        #    'EH_CC': calculate_metric_outcome(a_metrics['C'], actual['A_C']),
        #    
        #    # Away outcomes
        #    'EA_S': calculate_metric_outcome(a_metrics['S'], actual['A_S']),
        #    'EA_T': calculate_metric_outcome(a_metrics['T'], actual['A_T']),
        #    'EA_C': calculate_metric_outcome(a_metrics['C'], actual['A_C']),
        #    'EA_F': calculate_metric_outcome(a_metrics['F'], actual['A_F']),
        #    'EA_FK': calculate_metric_outcome(a_metrics['FK'], actual['A_FK']),
        #    'EA_Y': calculate_metric_outcome(a_metrics['Y'], actual['A_Y']),
        #    'EA_GP': calculate_metric_outcome(a_metrics['GC'], actual['A_GP']),
        #    'EA_GC': calculate_metric_outcome(h_metrics['GC'], actual['H_GP']),
        #    'EA_SC': calculate_metric_outcome(h_metrics['S'], actual['H_S']),
        #    'EA_TC': calculate_metric_outcome(h_metrics['T'], actual['H_T']),
        #    'EA_CC': calculate_metric_outcome(h_metrics['C'], actual['H_C'])
        #}
        
        # Modifica nella sezione del calcolo degli esiti TOP
        top_outcomes = {}
        for i in range(1, 6):
            top_key = f'TOP#{i}'
            etop_key = f'ETOP#{i}'
            
            # Controllo stato partita
            if record.get('Stato') != "Finita":
                top_outcomes[etop_key] = "-"
                continue
            
            # Calcolo esito solo per partite finite
            if top_key in record and not pd.isna(record[top_key]) and str(record[top_key]).strip() not in ["", "nan"]:
                try:
                    top_outcomes[etop_key] = calculate_top_outcome(
                        record[top_key], 
                        actual['GTFH'], 
                        actual['GTFA']
                    )
                except:
                    top_outcomes[etop_key] = "ERR"
            else:
                top_outcomes[etop_key] = "-"
        
        # Aggiungi al record
        record.update(top_outcomes)
        
        # Riempimento colonne mancanti per tutte le TOP#/ETOP#
        for i in range(1, 6):
            for prefix in ['TOP#', 'ETOP#']:
                key = f'{prefix}{i}'
                if key not in record:
                    record[key] = np.nan if prefix == 'TOP#' else "-"
        
        # Costruzione record finale
        #record.update(outcome_entries) #Rimossa nella versione "1e" disattivata anche definizione "calculate_metric_outcome"
        record.update(top_outcomes)
        record.update({f'H_{k}': v for k, v in h_metrics_int.items()})
        record.update({f'A_{k}': v for k, v in a_metrics_int.items()})
        record['HxGv2'] = round(home_xgv2, 2)
        record['AxGv2'] = round(away_xgv2, 2)
        # Aggiungi le colonne ER# al record
        record.update(er_entries)
        
      
        # Sezione aggiornata per il salvataggio delle quote
        # Inizializza le colonne Odd# a NaN
        for i in range(1, 6):
            record[f'Odd#{i}'] = np.nan
        
        if not match_odds.empty:
            # Estrai le quote solo se il DataFrame non è vuoto
            for i in range(1, 6):
                bet_type = record.get(f'TOP#{i}', '')
                if bet_type and bet_type in match_odds.columns:
                    # Verifica che ci siano valori disponibili
                    if len(match_odds[bet_type].values) > 0:
                        record[f'Odd#{i}'] = match_odds[bet_type].values[0]
        
        report.append(record)
    
    return pd.DataFrame(report)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calcolo predizioni xG')
    parser.add_argument('--day', type=int, default=0,
                       help='Giorni da aggiungere/sottrarre alla data corrente (es. -7 per 7 giorni fa)')
    return parser.parse_args()

def get_schedule_path(target_date):
    filename = f"{target_date.strftime('%Y%m%d')}_Schedule.csv"
    return os.path.join(Config.DATA_DIR, filename)

# Aggiungi questa funzione prima di main()
def load_existing_weights(file_path):
    weights_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "[PESI]" in line:
                    # Esempio riga: "🔧 [PESI] TeamName         | Division        | S: 0.12, T: 0.15..."
                    parts = line.split('|')
                    if len(parts) < 3:
                        continue
                    team_part = parts[0].split('🔧 [PESI] ')[-1].strip()
                    team = team_part.split()[0].strip()  # Assume che il nome della squadra non abbia spazi
                    division = parts[1].strip()
                    weights_str = parts[2].strip()
                    weights = {}
                    for pair in weights_str.split(', '):
                        if ':' not in pair:
                            continue
                        key, val = pair.split(':', 1)
                        weights[key.strip()] = float(val.strip())
                    weights_dict[(team, division)] = weights
        return weights_dict
    except Exception as e:
        logging.warning(f"Errore nel caricamento dei pesi: {e}")
        return {}
        
# Aggiungi questa nuova funzione prima di main()
def update_predictions(schedule_path, predictions_path):
    """Aggiorna RIS, Stato e ETOP# nel file delle predizioni usando i dati dello Schedule"""
    try:
        # Carica i dati
        schedule_df = pd.read_csv(schedule_path)
        predictions_df = pd.read_csv(predictions_path)

        # Converti le date
        schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])

        # Unisci i dataframe
        merged = pd.merge(
            predictions_df,
            schedule_df[['Date', 'Time', 'Country', 'League', 'Div', 'Home', 'Away', 'GFTH', 'GFTA', 'GHTH', 'GHTA']],
            on=['Date', 'Time', 'Country', 'League', 'Div', 'Home', 'Away'],
            how='left',
            suffixes=('', '_new')
        )

        # Funzione per ottenere il valore più recente (preferisce '_new' se disponibile)
        def get_latest_value(row, col):
            new_col = f'{col}_new'
            return row[new_col] if pd.notna(row.get(new_col, np.nan)) else row[col]

        # Aggiorna RIS
        merged['RIS'] = merged.apply(lambda row: compute_RIS({
            'GFTH': get_latest_value(row, 'GFTH'),
            'GFTA': get_latest_value(row, 'GFTA'),
            'GHTH': get_latest_value(row, 'GHTH'),
            'GHTA': get_latest_value(row, 'GHTA')
        }), axis=1)

        # Aggiorna Stato
        merged['Stato'] = merged.apply(lambda row: calculate_stato({
            'Date': row['Date'],
            'Time': row['Time']
        }), axis=1)

        # Aggiorna ETOP#1-5
        for i in range(1, 6):
            top_col = f'TOP#{i}'
            etop_col = f'ETOP#{i}'
            if top_col in merged.columns:
                merged[etop_col] = merged.apply(lambda row: 
                    calculate_top_outcome(
                        row[top_col],
                        get_latest_value(row, 'GFTH'),
                        get_latest_value(row, 'GFTA')
                    ) if row['Stato'] == 'Finita' else '-', 
                    axis=1
                )

        # Mantieni solo le colonne originali
        original_columns = predictions_df.columns
        merged = merged[original_columns]

        # Sovrascrivi il file
        merged.to_csv(predictions_path, index=False)
        logging.info(f"Predizioni aggiornate: {predictions_path}")

    except Exception as e:
        logging.error(f"Errore durante l'aggiornamento: {str(e)}")

def main():
    args = parse_arguments()
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    target_date = datetime.now() + timedelta(days=args.day)
    target_date_played = datetime.now() + timedelta(days=args.day - 1)
    output_filename = f"{target_date.strftime('%Y%m%d')}_xG_predictions.csv"
    output_path = os.path.join(Config.DATA_DIR, output_filename)
    schedule_path = get_schedule_path(target_date)
    played_path = os.path.join(Config.DATA_DIR, 'PLAYED.csv')

    # 1. GESTIONE PESI ESISTENTI (MODIFICATO)
    file_weights_path = os.path.join(Config.DATA_DIR, f"{target_date.strftime('%Y%m%d')}_feature_weights.log")
    
    # Carica sempre i pesi se il file esiste
    if os.path.exists(file_weights_path):
        logging.info(f" Trovato file pesi esistente: {file_weights_path}")
        Config.existing_weights = load_existing_weights(file_weights_path)
        
        if Config.existing_weights:
            logging.info(f" Caricati {len(Config.existing_weights)} set di pesi dalla cache")
        else:
            logging.warning(" File pesi vuoto o non valido, verranno ricalcolati")
            Config.existing_weights = {}
    else:
        logging.info(" Nessun file pesi esistente, verranno calcolati da zero")
        Config.existing_weights = {}

    # 2. SETUP LOGGER PESI (MODIFICATO: rimossa condizione day==0)
    try:
        file_handler = logging.FileHandler(
            filename=file_weights_path,
            mode='a' if os.path.exists(file_weights_path) else 'w',
            encoding='utf-8'
        )
        file_handler.addFilter(lambda record: "[PESI]" in record.getMessage())
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        logging.info(" Handler pesi configurato per tutte le date")
    except Exception as e:
        logging.error(f" Impossibile configurare logger pesi: {str(e)}")

    # 3. MODALITÀ OPERATIVA
    # Resto del codice invariato...
    if args.day == 0 and os.path.exists(output_path):
        logging.info(f" Modalità aggiornamento per {target_date.strftime('%Y-%m-%d')}")
        
        if not os.path.exists(schedule_path):
            logging.error(f" File schedule non trovato: {schedule_path}")
            return
            
        try:
            update_predictions(schedule_path, output_path)
            logging.info(" Aggiornamento completato")
        except Exception as e:
            logging.error(f" Errore aggiornamento: {str(e)}")

    else:
        logging.info(f" Generazione nuovo report per {target_date.strftime('%Y-%m-%d')}")
        
        if not os.path.exists(schedule_path):
            logging.error(f" File schedule non trovato: {schedule_path}")
            return

        try:
            played = pd.read_csv(played_path, na_values=["", " ", "NA"])
            schedule = pd.read_csv(schedule_path, na_values=["", " ", "NA"])
            
            # Conversioni dati
            numeric_cols = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 
                          'FKH', 'FKA', 'HY', 'AY', 'GFTH', 'GFTA']
            schedule[numeric_cols] = schedule[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            played['Date'] = pd.to_datetime(played['Date'], format='%Y-%m-%d')
            schedule['Date'] = pd.to_datetime(schedule['Date'], format='%Y-%m-%d')
            played = played[played['Date'] <= target_date_played]
            
            report_df = process_schedule(
                played,
                schedule,
                n=6,
                exclude_features=['FK', 'Y', 'GP', 'GC', 'SC', 'TC', 'CC']
            )
            
            if not report_df.empty:
                report_df.to_csv(output_path, index=False)
                logging.info(f" Report generato: {output_path}")
                
                if args.day == 0:
                    update_predictions(schedule_path, output_path)
            else:
                logging.warning(" Nessuna partita elaborata - File non creato")

        except Exception as e:
            logging.error(f" Errore durante l'elaborazione: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == "__main__":
    main()