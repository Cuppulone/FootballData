import argparse
import os
import pandas as pd
from datetime import datetime, timedelta
import warnings
from sklearn.linear_model import LogisticRegression
import logging
import numpy as np

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurazione iniziale
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Dati')
CONDITION_MAP = None

def parse_ris(ris_str):
    """Analizza il risultato finale ignorando gli eventuali risultati intermedi"""
    try:
        if not isinstance(ris_str, str) or 'ANNUL' in ris_str.upper():
            return None, None
            
        # Estrae solo la parte del risultato finale (es. "3-1" da "3-1 (1-1)")
        final_score = ris_str.split()[0].strip()
        home, away = map(int, final_score.split('-'))
        return home, away
    except Exception as e:
        logger.debug(f"Errore nel parsing di '{ris_str}': {str(e)}")
        return None, None

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

def load_data(start_date, end_date):
    """Carica i dati utilizzando ETOP#x per determinare il target"""
    train_data = {'diff': [], 'target': []}
    current_date = start_date
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'total_rows': 0,
        'skipped_nan': 0,
        'skipped_etop': 0,
        'valid_occurrences': 0
    }
    
    while current_date <= end_date:
        file_date = current_date.strftime("%Y%m%d")
        file_path = os.path.join(DATA_DIR, f"{file_date}_xG_predictions.csv")
        stats['total_files'] += 1
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                stats['processed_files'] += 1
                stats['total_rows'] += len(df)
                
                for _, row in df.iterrows():
                    for i in range(1, 6):
                        top = row.get(f'TOP#{i}')
                        q = row.get(f'Q#{i}')
                        odd = row.get(f'Odd#{i}')
                        etop = row.get(f'ETOP#{i}')
                        
                        # Normalizza i valori
                        if isinstance(top, str):
                            top = top.strip().upper()
                        if isinstance(etop, str):
                            etop = etop.strip().upper()
                        
                        # Controllo completezza dati
                        if pd.isna(top) or pd.isna(q) or pd.isna(odd) or pd.isna(etop):
                            stats['skipped_nan'] += 1
                            continue
                            
                        # Utilizza direttamente ETOP per il target
                        if etop == 'W':
                            target_val = 1
                        elif etop == 'L':
                            target_val = 0
                        else:
                            stats['skipped_etop'] += 1
                            continue
                            
                        try:
                            diff = q - odd
                            train_data['diff'].append(diff)
                            train_data['target'].append(target_val)
                            stats['valid_occurrences'] += 1
                        except Exception as e:
                            logger.debug(f"Errore nel calcolo diff: {str(e)}")
                            continue
            except Exception as e:
                logger.error(f"Errore nel processare {file_path}: {str(e)}")
        else:
            logger.warning(f"File non trovato: {file_path}")
        
        current_date += timedelta(days=1)
    
    # Log delle statistiche
    logger.info(f"File totali: {stats['total_files']}, Processati: {stats['processed_files']}")
    logger.info(f"Righe totali: {stats['total_rows']}")
    logger.info(f"Scartate per NaN: {stats['skipped_nan']}")
    logger.info(f"Scartate per ETOP non valido: {stats['skipped_etop']}")
    logger.info(f"Occorrenze valide: {stats['valid_occurrences']}")
    
    return pd.DataFrame(train_data)

def generate_report(target_date, model):
    date_str = target_date.strftime("%Y%m%d")
    file_path = os.path.join(DATA_DIR, f"{date_str}_xG_predictions.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato")
    
    df = pd.read_csv(file_path)
    report = []
    
    for _, row in df.iterrows():
        home, away = parse_ris(row['RIS'])
        actual_outcome = "?"
        
        for i in range(1, 6):
            top = row.get(f'TOP#{i}')
            q = row.get(f'Q#{i}')
            odd = row.get(f'Odd#{i}')
            
            # Normalizza il TOP
            if isinstance(top, str):
                top = top.strip().upper()
            
            if pd.isna(top) or pd.isna(q) or pd.isna(odd):
                continue
                
            # Calcola l'esito reale se disponibile
            if home is not None and away is not None and top in CONDITION_MAP:
                try:
                    actual_outcome = "Y" if CONDITION_MAP[top](home, away) else "N"
                except:
                    actual_outcome = "?"
            else:
                actual_outcome = "?"
                
            diff = round(q - odd, 2)
            try:
                proba = model.predict_proba([[diff]])[0][1]
            except:
                proba = 0.0
                
            report.append({
                'TOP': top,
                'Q': round(q, 2),
                'Odd': round(odd, 2),
                'Diff': diff,
                'Probability': round(proba, 2),
                'Predicted': 'Y' if proba >= 0.8 else 'N',
                'Actual': actual_outcome
            })
    
    report_df = pd.DataFrame(report)
    report_filename = os.path.join(DATA_DIR, f"{date_str}_report_v2.csv")
    report_df.to_csv(report_filename, index=False)
    logger.info(f"Report generato: {report_filename} con {len(report)} occorrenze")
    return report_filename

def main():
    global CONDITION_MAP
    CONDITION_MAP = load_condition_map()
    logger.info(f"Caricate {len(CONDITION_MAP)} condizioni")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, default=0, help="Day offset from today (0=today, -1=yesterday)")
    parser.add_argument('--window', type=int, default=30, help="Training window size in days (default: 8)")
    args = parser.parse_args()
    
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    target_date = base_date + timedelta(days=args.day)
    
    # MODIFICA: Calcolo dinamico delle date di training
    if target_date >= base_date:
        # Previsione per oggi o futuro: training fino a ieri
        end_date = base_date - timedelta(days=1)
        start_date = end_date - timedelta(days=args.window - 1)
    else:
        # Previsione per il passato
        start_date = target_date - timedelta(days=args.window)
        end_date = target_date - timedelta(days=1)
    
    logger.info(f"Caricamento dati da {start_date.date()} a {end_date.date()}")
    train_df = load_data(start_date, end_date)
    
    if train_df.empty:
        raise ValueError("Nessun dato disponibile per il training")
    
    # Addestramento del modello
    X = train_df[['diff']].values
    y = train_df['target'].values
    
    # Aggiungi intercetta per stabilità numerica
    model = LogisticRegression(fit_intercept=True, max_iter=1000)
    model.fit(X, y)
    
    logger.info(f"Modello addestrato su {len(X)} campioni")
    logger.info(f"Coefficienti: intercetta={model.intercept_[0]:.4f}, coeff={model.coef_[0][0]:.4f}")
    
    # Genera il report
    report_file = generate_report(target_date, model)
    print(f"Report generato con successo: {report_file}")

if __name__ == "__main__":
    main()