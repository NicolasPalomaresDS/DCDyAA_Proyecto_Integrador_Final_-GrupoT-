import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap

def get_xgb_model(X_train_resampled, y_train_resampled, use_grid_search=True):
    """
    Entrena un modelo XGBoost con o sin GridSearchCV.
    
    Parámetros:
        X_train_resampled: Features de entrenamiento (balanceadas con SMOTE).
        y_train_resampled: Etiquetas de entrenamiento correspondientes.
        use_grid_search (bool): Si True, aplica búsqueda de hiperparámetros con GridSearchCV.
    
    Retorna:
        model: Modelo XGBoost entrenado.
    """
    base_model = XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        enable_categorical=True
    )

    if use_grid_search:
        param_grid = {
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'n_estimators': [100, 200, 500]
        }

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train_resampled, y_train_resampled)
        model = grid_search.best_estimator_
        print("Mejores parámetros encontrados:")
        print(grid_search.best_params_)
    else:
        model = model or base_model
        model.fit(X_train_resampled, y_train_resampled)
    return model

# ====================================================================================

def get_rf_model(X_train_resampled, y_train_resampled, use_grid_search=True):
    """
    Entrena un modelo Random Forest con o sin GridSearchCV.
    
    Parámetros:
        X_train_resampled: Features de entrenamiento (balanceadas con SMOTE).
        y_train_resampled: Etiquetas de entrenamiento correspondientes.
        use_grid_search (bool): Si True, aplica búsqueda de hiperparámetros con GridSearchCV.
    
    Retorna:
        model: Modelo Random Forest entrenado.
    """
    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    if use_grid_search:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train_resampled, y_train_resampled)
        model = grid_search.best_estimator_
        print("Mejores parámetros encontrados:")
        print(grid_search.best_params_)
    else:
        model = model or base_model
        model.fit(X_train_resampled, y_train_resampled)
    return model

# ====================================================================================

def get_lgbm_model(X_train_resampled, y_train_resampled, use_grid_search=True):
    """
    Entrena un modelo LightGBM con o sin GridSearchCV.
    
    Parámetros:
        X_train_resampled: Features de entrenamiento (balanceadas con SMOTE).
        y_train_resampled: Etiquetas de entrenamiento correspondientes.
        use_grid_search (bool): Si True, aplica búsqueda de hiperparámetros con GridSearchCV.
    
    Retorna:
        model: Modelo LightGBM entrenado.
    """
    try:
        base_model = LGBMClassifier(
            random_state=42,
            n_jobs=1,           
            verbose=-1,
            force_row_wise=True
        )
    except TypeError:
        base_model = LGBMClassifier(
            random_state=42,
            n_jobs=1,
            verbose=-1
        )

    if use_grid_search:
        param_dist = {
            'num_leaves': [31, 63, 127, 255],
            'max_depth': [-1, 5, 10, 20],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.0, 0.1, 0.5, 1.0]
        }

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=40,
            cv=5,
            scoring='f1_weighted',
            verbose=0,
            n_jobs=-1,                 
            random_state=42,
            return_train_score=False,
            pre_dispatch='2*n_jobs' 
        )

        random_search.fit(X_train_resampled, y_train_resampled)
        model = random_search.best_estimator_
        print("Mejores parámetros encontrados:")
        print(random_search.best_params_)
        return model
    else:
        base_model.fit(X_train_resampled, y_train_resampled)
        return base_model
    
# ====================================================================================
    
def ML_report(X_train, X_test, y_train, y_test, le, model_name):
    """
    Genera reporte de métricas, matriz de confusión y gráficos SHAP para un modelo ML.
    
    Parámetros:
        X_train, X_test: Features de train y test.
        y_train, y_test: Etiquetas de train y test.
        le: LabelEncoder ya ajustado, para mapear etiquetas.
        model_name (str): Nombre del modelo a usar ('xgb', 'rf', 'lgbm').
    
    Retorna:
        model: El modelo final ya entrenado.
        report: Reporte sklearn de métricas de clasificación.
        cm_labeled: Matriz de confusión con etiquetas.
    """
    # Aplicar SMOTE al set de entrenamiento
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    if model_name == 'xgb':
        model = get_xgb_model(X_train_resampled, y_train_resampled)
    elif model_name == 'rf':
        model = get_rf_model(X_train_resampled, y_train_resampled)
    elif model_name == 'lgbm':
        model = get_lgbm_model(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test, y_pred)
    
    cm_labeled = pd.DataFrame(
        cm,
        index=[f"Actual: {label}" for label in le.classes_],
        columns=[f"Predicción: {label}" for label in le.classes_]
    )
    
    # SHAP
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(
        shap_values.values.mean(2),
        X_test,
        show=True,
        feature_names=X_test.columns if hasattr(X_test, "columns") else None
    )
    
    if model_name == 'lgbm':
        return model, report, cm_labeled, X_train_resampled, y_train_resampled
    else:
        return model, report, cm_labeled
        
