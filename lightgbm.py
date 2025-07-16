# 1차 최고 점수: 0.4307 최고 점수: 0.4302

import pandas as pd
import numpy as np
import os
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from rdkit.Chem.Descriptors import MolLogP, MolWt, TPSA, NumHDonors, NumHAcceptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold # StratifiedKFold로 변경
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# --- 개선된 설정값 ---
CFG = {
    'NBITS': 2048,  # 비트 수를 2048로 조정하여 희소성 완화
    'SEED': 42,
    'N_SPLITS': 10,  # CV 분할 수를 10으로 늘려 안정성 확보   10 ,20(x) 최고 점수: 0.4300
    'N_TRIALS': 150, # 탐색 횟수를 150으로 증가              150 200(x)
    'RADIUS': 2, #2
    'FEATURE_SELECTION': True,
    'USE_BALANCING': False, # 데이터 균형화 전략 비활성화 (StratifiedKFold로 대체)
    'ENSEMBLE': True,       # 앙상블 전략 활성화
    'OUTLIER_REMOVAL': True,
    'ADVANCED_FEATURES': True,
    'MIN_SAMPLES_PER_CLASS': 50, #100
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

# --- 데이터 로드 및 전처리 (기존과 동일) ---
def load_data():
    # ... (데이터 로드 함수는 원본 스크립트와 동일하게 유지) ...
    encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    
    def read_csv_with_encodings(filepath, sep=None):
        for enc in encodings_to_try:
            try:
                if sep:
                    df = pd.read_csv(filepath, sep=sep, encoding=enc)
                else:
                    df = pd.read_csv(filepath, encoding=enc)
                print(f"Successfully loaded {filepath} with encoding: {enc}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error loading {filepath} with encoding {enc}: {e}")
                raise
        raise UnicodeDecodeError(f"Failed to decode {filepath} with any of {encodings_to_try} encodings.")

    print("Loading ChEMBL_ASK1(IC50).csv...")
    chembl = read_csv_with_encodings("./ChEMBL_ASK1(IC50).csv", sep=';')
    
    print("Loading Pubchem_ASK1.csv...")
    pubchem = read_csv_with_encodings("./Pubchem_ASK1.csv")

    print("Loading test.csv...")
    test = read_csv_with_encodings("./test.csv")
    
    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    chembl = chembl[chembl['Standard Type'] == 'IC50']
    chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'})
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
    
    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'})
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
    
    train = pd.concat([chembl, pubchem], ignore_index=True)
    train = train.dropna(subset=['smiles', 'ic50_nM'])
    train = train.drop_duplicates(subset='smiles').reset_index(drop=True)
    train = train[train['ic50_nM'] > 0]
    
    if CFG['OUTLIER_REMOVAL']:
        log_ic50 = np.log10(train['ic50_nM'])
        Q1 = log_ic50.quantile(0.05)
        Q3 = log_ic50.quantile(0.95)
        train = train[(log_ic50 >= Q1) & (log_ic50 <= Q3)]
        print(f"이상치 제거 후 데이터 수: {len(train)}")
    
    return train, test

# --- 클래스 불균형 처리 함수 (선택적으로 사용) ---
def balance_dataset_by_activity(df, min_samples_per_class=50):
    df['activity_class'] = pd.cut(df['pIC50'], 
                                 bins=[0, 5, 6, 7, 8, float('inf')],
                                 labels=['inactive', 'low', 'medium', 'high', 'very_high'])
    
    balanced_dfs = []
    for class_label in df['activity_class'].unique():
        if pd.isna(class_label): continue
        class_df = df[df['activity_class'] == class_label].copy()
        if len(class_df) < min_samples_per_class:
            n_needed = min_samples_per_class - len(class_df)
            additional_samples = class_df.sample(n=n_needed, replace=True, random_state=CFG['SEED'])
            class_df = pd.concat([class_df, additional_samples], ignore_index=True)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.drop('activity_class', axis=1)
    print(f"Balanced dataset size: {len(balanced_df)}")
    return balanced_df

# --- 특성 생성 함수들 (기존과 동일) ---
def smiles_to_safe_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.full(20, 0.0)
    features = [MolLogP(mol), MolWt(mol), TPSA(mol), NumHDonors(mol), NumHAcceptors(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumHeterocycles(mol), mol.GetNumAtoms(), mol.GetNumBonds(),
                mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol), rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                rdMolDescriptors.CalcLabuteASA(mol), rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumHBA(mol),
                rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumRings(mol)]
    return np.nan_to_num(np.array(features, dtype=float))

def smiles_to_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'morgan': np.zeros(CFG['NBITS']), 'maccs': np.zeros(167)}
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, CFG['RADIUS'], nBits=CFG['NBITS'])
    morgan_arr = np.zeros((CFG['NBITS'],), dtype=int)
    DataStructs.ConvertToNumpyArray(morgan, morgan_arr)
    maccs = AllChem.GetMACCSKeysFingerprint(mol)
    maccs_arr = np.zeros((167,), dtype=int)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return {'morgan': morgan_arr, 'maccs': maccs_arr}

def smiles_to_descriptors(smiles, desc_mean=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return desc_mean.copy() if desc_mean is not None else np.full(len(Descriptors._descList), 0.0)
    desc = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.nan_to_num(np.array(desc, dtype=float))

# --- 변환 및 평가 함수 (기존과 동일) ---
def IC50_to_pIC50(ic50): return 9 - np.log10(ic50)
def pIC50_to_IC50(pIC50): return 10**(9 - pIC50)

def get_score(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    rmse = mean_squared_error(y_true_ic50, y_pred_ic50, squared=False)
    nrmse = rmse / (np.max(y_true_ic50) - np.min(y_true_ic50))
    A = 1 - min(nrmse, 1)
    B = r2_score(y_true_pic50, y_pred_pic50)
    score = 0.4 * A + 0.6 * B
    return score

# --- 개선된 Optuna 목적 함수 (StratifiedKFold 적용) ---
def objective(trial, X, y, y_binned): # y_binned를 추가로 받음
    model_type = trial.suggest_categorical('model_type', ['lgb', 'xgb'])
    
    if model_type == 'lgb':
        params = { 'objective': 'regression_l1', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1, 'seed': CFG['SEED'],
                   'boosting_type': 'gbdt', 'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
                   'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                   'num_leaves': trial.suggest_int('num_leaves', 20, 300), 'max_depth': trial.suggest_int('max_depth', 5, 16),
                   'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                   'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                   'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                   'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)}
    else: # XGBoost
        params = { 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'verbosity': 0, 'n_jobs': -1, 'random_state': CFG['SEED'],
                   'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
                   'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                   'max_depth': trial.suggest_int('max_depth', 5, 16),
                   'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                   'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                   'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                   'early_stopping_rounds': 100}

    # --- StratifiedKFold 사용 ---
    skf = StratifiedKFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
    oof_preds = np.zeros(len(X))

    # split 메서드에 X와 함께 계층화의 기준이 될 y_binned를 전달
    for train_idx, val_idx in skf.split(X, y_binned):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
        else: # XGBoost
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        oof_preds[val_idx] = model.predict(X_val)

    y_ic50_true = pIC50_to_IC50(y)
    oof_ic50_preds = pIC50_to_IC50(oof_preds)
    score = get_score(y_ic50_true, oof_ic50_preds, y, oof_preds)
    return score

# --- 테스트 데이터 컬럼 찾기 함수 (기존과 동일) ---
def find_smiles_column(df):
    for col in ['smiles', 'SMILES', 'Smiles']:
        if col in df.columns: return col
    raise ValueError("SMILES column not found in test data")

# --- 메인 실행부 ---
if __name__ == "__main__":
    train_df, test_df = load_data()
    train_df['pIC50'] = IC50_to_pIC50(train_df['ic50_nM'])

    if CFG['USE_BALANCING']:
        print("데이터 균형화 전략 수행 중...")
        train_df = balance_dataset_by_activity(train_df, CFG['MIN_SAMPLES_PER_CLASS'])

    # --- 특성 생성 ---
    print("특성 생성 중...")
    train_fps = train_df['smiles'].apply(smiles_to_fingerprints)
    train_descriptors = np.stack([smiles_to_descriptors(s) for s in train_df['smiles']])
    desc_mean = np.nanmean(train_descriptors, axis=0)
    train_descriptors = np.nan_to_num(train_descriptors, nan=desc_mean)

    X_morgan = np.stack([fp['morgan'] for fp in train_fps])
    X_maccs = np.stack([fp['maccs'] for fp in train_fps])
    
    desc_scaler = StandardScaler()
    X_desc_scaled = desc_scaler.fit_transform(train_descriptors)
    
    X = np.hstack([X_morgan, X_maccs, X_desc_scaled])

    if CFG['ADVANCED_FEATURES']:
        train_safe_features = np.stack([smiles_to_safe_features(s) for s in train_df['smiles']])
        safe_scaler = StandardScaler()
        X_safe_scaled = safe_scaler.fit_transform(train_safe_features)
        X = np.hstack([X, X_safe_scaled])

    y = train_df['pIC50'].values
    print(f"초기 특성 차원: {X.shape}")

    # --- 특성 선택 ---
    if CFG['FEATURE_SELECTION']:
        print("특성 선택 수행 중...")
        var_selector = VarianceThreshold(threshold=0.01)
        X = var_selector.fit_transform(X)
        
        k_best = min(3000, X.shape[1])
        kbest_selector = SelectKBest(score_func=f_regression, k=k_best)
        X = kbest_selector.fit_transform(X, y)
        print(f"선택된 특성 차원: {X.shape}")

    # --- Optuna 최적화 (StratifiedKFold 기준 생성) ---
    # StratifiedKFold를 위한 y값 binning
    y_binned = pd.cut(y, bins=CFG['N_SPLITS'], labels=False, include_lowest=True)
    
    print("Optuna 하이퍼파라미터 최적화 시작...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=CFG['SEED']))
    study.optimize(lambda trial: objective(trial, X, y, y_binned), n_trials=CFG['N_TRIALS'])

    print(f"최고 점수: {study.best_value:.4f}")

    # --- 최종 모델 학습 및 예측 ---
    smiles_col = find_smiles_column(test_df)
    
    # 테스트 데이터 특성 생성
    print("테스트 데이터 특성 생성 중...")
    test_fps = test_df[smiles_col].apply(smiles_to_fingerprints)
    test_descriptors = np.stack([smiles_to_descriptors(s, desc_mean) for s in test_df[smiles_col]])
    
    test_X_morgan = np.stack([fp['morgan'] for fp in test_fps])
    test_X_maccs = np.stack([fp['maccs'] for fp in test_fps])
    test_X_desc_scaled = desc_scaler.transform(test_descriptors)
    
    X_test = np.hstack([test_X_morgan, test_X_maccs, test_X_desc_scaled])
    
    if CFG['ADVANCED_FEATURES']:
        test_safe_features = np.stack([smiles_to_safe_features(s) for s in test_df[smiles_col]])
        test_X_safe_scaled = safe_scaler.transform(test_safe_features)
        X_test = np.hstack([X_test, test_X_safe_scaled])

    if CFG['FEATURE_SELECTION']:
        X_test = var_selector.transform(X_test)
        X_test = kbest_selector.transform(X_test)

    # --- 앙상블 로직 ---
    if CFG['ENSEMBLE']:
        print("앙상블 모델 학습 및 예측 수행 중...")
        
        # 각 모델 타입별 최고 성능 파라미터 찾기
        lgb_trials = [t for t in study.trials if t.params.get('model_type') == 'lgb']
        xgb_trials = [t for t in study.trials if t.params.get('model_type') == 'xgb']

        if not lgb_trials or not xgb_trials:
            raise ValueError("Optuna study did not complete enough trials for both models.")

        best_lgb_params = max(lgb_trials, key=lambda t: t.value).params
        best_xgb_params = max(xgb_trials, key=lambda t: t.value).params
        
        best_lgb_params.pop('model_type')
        best_xgb_params.pop('model_type')
        
        # LGBM 모델 학습 및 예측
        lgb_model = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', verbose=-1, n_jobs=-1, seed=CFG['SEED'], **best_lgb_params)
        lgb_model.fit(X, y)
        preds_lgb = lgb_model.predict(X_test)

        # XGBoost 모델 학습 및 예측
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', verbosity=0, n_jobs=-1, random_state=CFG['SEED'], **best_xgb_params)
        xgb_model.fit(X, y)
        preds_xgb = xgb_model.predict(X_test)

        # 예측 결과 앙상블 (가중치 0.5/0.5)
        test_preds_pIC50 = (preds_lgb + preds_xgb) / 2.0
    
    else: # 앙상블을 사용하지 않을 경우, 전체 최고 모델로 예측
        print("단일 최적 모델 학습 및 예측 수행 중...")
        best_params = study.best_params.copy()
        model_type = best_params.pop('model_type')
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', verbose=-1, n_jobs=-1, seed=CFG['SEED'], **best_params)
        else:
            model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', verbosity=0, n_jobs=-1, random_state=CFG['SEED'], **best_params)
        
        model.fit(X, y)
        test_preds_pIC50 = model.predict(X_test)

    # pIC50을 IC50으로 변환
    test_preds_ic50 = pIC50_to_IC50(test_preds_pIC50)

    # --- 결과 저장 ---
    print("결과 저장 중...")
    submission = pd.DataFrame({'id': test_df.get('id', test_df.get('ID', range(len(test_df)))), 'IC50_nM': test_preds_ic50})
    submission.to_csv('submission_improved.csv', index=False)
    print("submission_improved.csv 파일이 저장되었습니다.")
    print(submission.head())
    print("\n예측 완료!")# 1차 최고 점수: 0.4307 최고 점수: 0.4302

import pandas as pd
import numpy as np
import os
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from rdkit.Chem.Descriptors import MolLogP, MolWt, TPSA, NumHDonors, NumHAcceptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold # StratifiedKFold로 변경
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# --- 개선된 설정값 ---
CFG = {
    'NBITS': 2048,  # 비트 수를 2048로 조정하여 희소성 완화
    'SEED': 42,
    'N_SPLITS': 10,  # CV 분할 수를 10으로 늘려 안정성 확보   10 ,20(x) 최고 점수: 0.4300
    'N_TRIALS': 150, # 탐색 횟수를 150으로 증가              150 200(x)
    'RADIUS': 2, #2
    'FEATURE_SELECTION': True,
    'USE_BALANCING': False, # 데이터 균형화 전략 비활성화 (StratifiedKFold로 대체)
    'ENSEMBLE': True,       # 앙상블 전략 활성화
    'OUTLIER_REMOVAL': True,
    'ADVANCED_FEATURES': True,
    'MIN_SAMPLES_PER_CLASS': 50, #100
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

# --- 데이터 로드 및 전처리 (기존과 동일) ---
def load_data():
    # ... (데이터 로드 함수는 원본 스크립트와 동일하게 유지) ...
    encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    
    def read_csv_with_encodings(filepath, sep=None):
        for enc in encodings_to_try:
            try:
                if sep:
                    df = pd.read_csv(filepath, sep=sep, encoding=enc)
                else:
                    df = pd.read_csv(filepath, encoding=enc)
                print(f"Successfully loaded {filepath} with encoding: {enc}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error loading {filepath} with encoding {enc}: {e}")
                raise
        raise UnicodeDecodeError(f"Failed to decode {filepath} with any of {encodings_to_try} encodings.")

    print("Loading ChEMBL_ASK1(IC50).csv...")
    chembl = read_csv_with_encodings("./ChEMBL_ASK1(IC50).csv", sep=';')
    
    print("Loading Pubchem_ASK1.csv...")
    pubchem = read_csv_with_encodings("./Pubchem_ASK1.csv")

    print("Loading test.csv...")
    test = read_csv_with_encodings("./test.csv")
    
    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    chembl = chembl[chembl['Standard Type'] == 'IC50']
    chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'})
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
    
    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'})
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
    
    train = pd.concat([chembl, pubchem], ignore_index=True)
    train = train.dropna(subset=['smiles', 'ic50_nM'])
    train = train.drop_duplicates(subset='smiles').reset_index(drop=True)
    train = train[train['ic50_nM'] > 0]
    
    if CFG['OUTLIER_REMOVAL']:
        log_ic50 = np.log10(train['ic50_nM'])
        Q1 = log_ic50.quantile(0.05)
        Q3 = log_ic50.quantile(0.95)
        train = train[(log_ic50 >= Q1) & (log_ic50 <= Q3)]
        print(f"이상치 제거 후 데이터 수: {len(train)}")
    
    return train, test

# --- 클래스 불균형 처리 함수 (선택적으로 사용) ---
def balance_dataset_by_activity(df, min_samples_per_class=50):
    df['activity_class'] = pd.cut(df['pIC50'], 
                                 bins=[0, 5, 6, 7, 8, float('inf')],
                                 labels=['inactive', 'low', 'medium', 'high', 'very_high'])
    
    balanced_dfs = []
    for class_label in df['activity_class'].unique():
        if pd.isna(class_label): continue
        class_df = df[df['activity_class'] == class_label].copy()
        if len(class_df) < min_samples_per_class:
            n_needed = min_samples_per_class - len(class_df)
            additional_samples = class_df.sample(n=n_needed, replace=True, random_state=CFG['SEED'])
            class_df = pd.concat([class_df, additional_samples], ignore_index=True)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.drop('activity_class', axis=1)
    print(f"Balanced dataset size: {len(balanced_df)}")
    return balanced_df

# --- 특성 생성 함수들 (기존과 동일) ---
def smiles_to_safe_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.full(20, 0.0)
    features = [MolLogP(mol), MolWt(mol), TPSA(mol), NumHDonors(mol), NumHAcceptors(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumHeterocycles(mol), mol.GetNumAtoms(), mol.GetNumBonds(),
                mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(mol), rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                rdMolDescriptors.CalcLabuteASA(mol), rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumHBA(mol),
                rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumRings(mol)]
    return np.nan_to_num(np.array(features, dtype=float))

def smiles_to_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'morgan': np.zeros(CFG['NBITS']), 'maccs': np.zeros(167)}
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, CFG['RADIUS'], nBits=CFG['NBITS'])
    morgan_arr = np.zeros((CFG['NBITS'],), dtype=int)
    DataStructs.ConvertToNumpyArray(morgan, morgan_arr)
    maccs = AllChem.GetMACCSKeysFingerprint(mol)
    maccs_arr = np.zeros((167,), dtype=int)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return {'morgan': morgan_arr, 'maccs': maccs_arr}

def smiles_to_descriptors(smiles, desc_mean=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return desc_mean.copy() if desc_mean is not None else np.full(len(Descriptors._descList), 0.0)
    desc = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.nan_to_num(np.array(desc, dtype=float))

# --- 변환 및 평가 함수 (기존과 동일) ---
def IC50_to_pIC50(ic50): return 9 - np.log10(ic50)
def pIC50_to_IC50(pIC50): return 10**(9 - pIC50)

def get_score(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    rmse = mean_squared_error(y_true_ic50, y_pred_ic50, squared=False)
    nrmse = rmse / (np.max(y_true_ic50) - np.min(y_true_ic50))
    A = 1 - min(nrmse, 1)
    B = r2_score(y_true_pic50, y_pred_pic50)
    score = 0.4 * A + 0.6 * B
    return score

# --- 개선된 Optuna 목적 함수 (StratifiedKFold 적용) ---
def objective(trial, X, y, y_binned): # y_binned를 추가로 받음
    model_type = trial.suggest_categorical('model_type', ['lgb', 'xgb'])
    
    if model_type == 'lgb':
        params = { 'objective': 'regression_l1', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1, 'seed': CFG['SEED'],
                   'boosting_type': 'gbdt', 'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
                   'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                   'num_leaves': trial.suggest_int('num_leaves', 20, 300), 'max_depth': trial.suggest_int('max_depth', 5, 16),
                   'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                   'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                   'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                   'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)}
    else: # XGBoost
        params = { 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'verbosity': 0, 'n_jobs': -1, 'random_state': CFG['SEED'],
                   'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
                   'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                   'max_depth': trial.suggest_int('max_depth', 5, 16),
                   'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                   'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                   'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                   'early_stopping_rounds': 100}

    # --- StratifiedKFold 사용 ---
    skf = StratifiedKFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
    oof_preds = np.zeros(len(X))

    # split 메서드에 X와 함께 계층화의 기준이 될 y_binned를 전달
    for train_idx, val_idx in skf.split(X, y_binned):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
        else: # XGBoost
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        oof_preds[val_idx] = model.predict(X_val)

    y_ic50_true = pIC50_to_IC50(y)
    oof_ic50_preds = pIC50_to_IC50(oof_preds)
    score = get_score(y_ic50_true, oof_ic50_preds, y, oof_preds)
    return score

# --- 테스트 데이터 컬럼 찾기 함수 (기존과 동일) ---
def find_smiles_column(df):
    for col in ['smiles', 'SMILES', 'Smiles']:
        if col in df.columns: return col
    raise ValueError("SMILES column not found in test data")

# --- 메인 실행부 ---
if __name__ == "__main__":
    train_df, test_df = load_data()
    train_df['pIC50'] = IC50_to_pIC50(train_df['ic50_nM'])

    if CFG['USE_BALANCING']:
        print("데이터 균형화 전략 수행 중...")
        train_df = balance_dataset_by_activity(train_df, CFG['MIN_SAMPLES_PER_CLASS'])

    # --- 특성 생성 ---
    print("특성 생성 중...")
    train_fps = train_df['smiles'].apply(smiles_to_fingerprints)
    train_descriptors = np.stack([smiles_to_descriptors(s) for s in train_df['smiles']])
    desc_mean = np.nanmean(train_descriptors, axis=0)
    train_descriptors = np.nan_to_num(train_descriptors, nan=desc_mean)

    X_morgan = np.stack([fp['morgan'] for fp in train_fps])
    X_maccs = np.stack([fp['maccs'] for fp in train_fps])
    
    desc_scaler = StandardScaler()
    X_desc_scaled = desc_scaler.fit_transform(train_descriptors)
    
    X = np.hstack([X_morgan, X_maccs, X_desc_scaled])

    if CFG['ADVANCED_FEATURES']:
        train_safe_features = np.stack([smiles_to_safe_features(s) for s in train_df['smiles']])
        safe_scaler = StandardScaler()
        X_safe_scaled = safe_scaler.fit_transform(train_safe_features)
        X = np.hstack([X, X_safe_scaled])

    y = train_df['pIC50'].values
    print(f"초기 특성 차원: {X.shape}")

    # --- 특성 선택 ---
    if CFG['FEATURE_SELECTION']:
        print("특성 선택 수행 중...")
        var_selector = VarianceThreshold(threshold=0.01)
        X = var_selector.fit_transform(X)
        
        k_best = min(3000, X.shape[1])
        kbest_selector = SelectKBest(score_func=f_regression, k=k_best)
        X = kbest_selector.fit_transform(X, y)
        print(f"선택된 특성 차원: {X.shape}")

    # --- Optuna 최적화 (StratifiedKFold 기준 생성) ---
    # StratifiedKFold를 위한 y값 binning
    y_binned = pd.cut(y, bins=CFG['N_SPLITS'], labels=False, include_lowest=True)
    
    print("Optuna 하이퍼파라미터 최적화 시작...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=CFG['SEED']))
    study.optimize(lambda trial: objective(trial, X, y, y_binned), n_trials=CFG['N_TRIALS'])

    print(f"최고 점수: {study.best_value:.4f}")

    # --- 최종 모델 학습 및 예측 ---
    smiles_col = find_smiles_column(test_df)
    
    # 테스트 데이터 특성 생성
    print("테스트 데이터 특성 생성 중...")
    test_fps = test_df[smiles_col].apply(smiles_to_fingerprints)
    test_descriptors = np.stack([smiles_to_descriptors(s, desc_mean) for s in test_df[smiles_col]])
    
    test_X_morgan = np.stack([fp['morgan'] for fp in test_fps])
    test_X_maccs = np.stack([fp['maccs'] for fp in test_fps])
    test_X_desc_scaled = desc_scaler.transform(test_descriptors)
    
    X_test = np.hstack([test_X_morgan, test_X_maccs, test_X_desc_scaled])
    
    if CFG['ADVANCED_FEATURES']:
        test_safe_features = np.stack([smiles_to_safe_features(s) for s in test_df[smiles_col]])
        test_X_safe_scaled = safe_scaler.transform(test_safe_features)
        X_test = np.hstack([X_test, test_X_safe_scaled])

    if CFG['FEATURE_SELECTION']:
        X_test = var_selector.transform(X_test)
        X_test = kbest_selector.transform(X_test)

    # --- 앙상블 로직 ---
    if CFG['ENSEMBLE']:
        print("앙상블 모델 학습 및 예측 수행 중...")
        
        # 각 모델 타입별 최고 성능 파라미터 찾기
        lgb_trials = [t for t in study.trials if t.params.get('model_type') == 'lgb']
        xgb_trials = [t for t in study.trials if t.params.get('model_type') == 'xgb']

        if not lgb_trials or not xgb_trials:
            raise ValueError("Optuna study did not complete enough trials for both models.")

        best_lgb_params = max(lgb_trials, key=lambda t: t.value).params
        best_xgb_params = max(xgb_trials, key=lambda t: t.value).params
        
        best_lgb_params.pop('model_type')
        best_xgb_params.pop('model_type')
        
        # LGBM 모델 학습 및 예측
        lgb_model = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', verbose=-1, n_jobs=-1, seed=CFG['SEED'], **best_lgb_params)
        lgb_model.fit(X, y)
        preds_lgb = lgb_model.predict(X_test)

        # XGBoost 모델 학습 및 예측
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', verbosity=0, n_jobs=-1, random_state=CFG['SEED'], **best_xgb_params)
        xgb_model.fit(X, y)
        preds_xgb = xgb_model.predict(X_test)

        # 예측 결과 앙상블 (가중치 0.5/0.5)
        test_preds_pIC50 = (preds_lgb + preds_xgb) / 2.0
    
    else: # 앙상블을 사용하지 않을 경우, 전체 최고 모델로 예측
        print("단일 최적 모델 학습 및 예측 수행 중...")
        best_params = study.best_params.copy()
        model_type = best_params.pop('model_type')
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', verbose=-1, n_jobs=-1, seed=CFG['SEED'], **best_params)
        else:
            model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', verbosity=0, n_jobs=-1, random_state=CFG['SEED'], **best_params)
        
        model.fit(X, y)
        test_preds_pIC50 = model.predict(X_test)

    # pIC50을 IC50으로 변환
    test_preds_ic50 = pIC50_to_IC50(test_preds_pIC50)

    # --- 결과 저장 ---
    print("결과 저장 중...")
    submission = pd.DataFrame({'id': test_df.get('id', test_df.get('ID', range(len(test_df)))), 'IC50_nM': test_preds_ic50})
    submission.to_csv('submission_improved.csv', index=False)
    print("submission_improved.csv 파일이 저장되었습니다.")
    print(submission.head())
    print("\n예측 완료!")
