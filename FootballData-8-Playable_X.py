import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Configurazione percorsi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Dati')

def setup_argparse():
    parser = argparse.ArgumentParser(description='Analisi combinazioni scommesse')
    parser.add_argument('--day', 
                        type=int,
                        default=0,
                        help="Offset giorni dalla data corrente (0=oggi, -1=ieri)")
    return parser.parse_args()

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

def generate_historical_dates(target_date, start_offset=9, end_offset=1):
    start_date = target_date - timedelta(days=start_offset)
    end_date = target_date - timedelta(days=end_offset)
    delta_days = (end_date - start_date).days + 1  # +1 per includere entrambe le date
    return [(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(delta_days)]

def load_historical_data(date_list):
    dfs = []
    for date in date_list:
        file_name = f"{date}_xG_predictions.csv"
        file_path = os.path.join(DATA_DIR, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = process_ris_column(df)
            dfs.append(df)
        else:
            print(f"Attenzione: File {file_name} non trovato, skipping...")
    return pd.concat(dfs, ignore_index=True) if dfs else None

def load_target_data(target_date):
    target_file = f"{target_date}_xG_predictions.csv"
    file_path = os.path.join(DATA_DIR, target_file)
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Errore: File {target_file} non trovato")
        sys.exit(1)

def process_ris_column(df):
    ris_split = df['RIS'].str.split(r'[()-]', expand=True)
    df['h'] = ris_split[0].astype(int)
    df['a'] = ris_split[1].astype(int)
    return df

def create_features(df, is_training=True):
    samples = []
    for _, row in df.iterrows():
        for i in range(1, 6):
            q = round(row[f'Q#{i}'], 2)
            odd = round(row[f'Odd#{i}'], 2)
            q_odd_ratio = round(q / odd, 2)
            delta_q_odd = round(q - odd, 2)
            
            sample = {
                'TOP': row[f'TOP#{i}'],
                'Q': q,
                'Odd': odd,
                'Q/Odd_ratio': q_odd_ratio,
                'Delta_Q_Odd': delta_q_odd
            }
            
            if is_training:
                # Utilizzo diretto delle colonne ETOP#x se disponibili
                etop_value = row.get(f'ETOP#{i}')
                if etop_value in ['W', 'L']:
                    sample['label'] = etop_value
                else:
                    # Fallback al calcolo con condition_map
                    condition = condition_map.get(sample['TOP'], lambda h, a: False)
                    sample['label'] = 'W' if condition(row['h'], row['a']) else 'L'
            
            samples.append(sample)
    return pd.DataFrame(samples)

def train_model(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            class_weight='balanced',
            random_state=42
        )
    )
    model.fit(X_train, y_train)
    return model

def analyze_combinations(model, target_df, threshold=0.65):
    features = ['Q', 'Odd', 'Q/Odd_ratio', 'Delta_Q_Odd']
    target_data = create_features(target_df, is_training=False)
    
    # Calcola probabilità e valore atteso
    target_data['Prob_W'] = model.predict_proba(target_data[features])[:, 1]
    target_data['EV'] = target_data['Q'] * target_data['Prob_W'] - (1 - target_data['Prob_W'])
    
    # Arrotonda i risultati finali
    target_data['Prob_W'] = target_data['Prob_W'].round(2)
    target_data['EV'] = target_data['EV'].round(2)
    
    return target_data[target_data['EV'] > 0].sort_values('EV', ascending=False)

def main():
    args = setup_argparse()
    
    # Calcola data target
    base_date = datetime.now()
    target_date_obj = base_date + timedelta(days=args.day)
    target_date = target_date_obj.strftime('%Y%m%d')
    
    # Carica dati storici (da oggi-18 a oggi-3)
    historical_dates = generate_historical_dates(target_date_obj)  # Usa i default 18 e 3
    historical_df = load_historical_data(historical_dates)
    
    if historical_df is None or len(historical_df) == 0:
        print("Errore: Dati storici insufficienti per l'analisi")
        sys.exit(1)
    
    # Preparazione modello
    labeled_data = create_features(historical_df)
    model = train_model(labeled_data[['Q', 'Odd', 'Q/Odd_ratio', 'Delta_Q_Odd']], labeled_data['label'])
    
    # Analizza dati target
    target_df = load_target_data(target_date)
    results = analyze_combinations(model, target_df)
    
    # Salva risultati
    output_path = os.path.join(os.path.join(BASE_DIR, 'Dati'), f'{target_date}_playable.csv')
    results.to_csv(output_path, index=False)
    
    print("\n Top 10 combinazioni giocabili:")
    print(results[['TOP', 'Q', 'Odd', 'EV', 'Prob_W']]
          .head(10)
          .round({'Q': 2, 'Odd': 2, 'EV': 2, 'Prob_W': 2})
          .to_string(index=False, formatters={
              'Prob_W': '{:.0%}'.format,
              'EV': '{:.2f}'.format
          }))

if __name__ == '__main__':
    main()