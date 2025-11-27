"""Prediction module for liver cancer risk.

This module loads the preprocessing pipeline and trained model artifacts, and exposes
predict_user_input() to estimate risk from user-provided answers. It is robust to
missing lab values (AFP and liver function).
"""
from typing import Any, Dict
import os
import pickle
import json
import numpy as np
import pandas as pd

# Expected schema (align with preprocessing script)
TARGET_COL = "liver_cancer"
NUMERIC_COLS = ["age", "bmi", "liver_function_score", "alpha_fetoprotein_level"]
CATEGORICAL_COLS = ["gender", "alcohol_consumption", "smoking_status", "physical_activity_level"]
BINARY_COLS = ["hepatitis_b", "hepatitis_c", "cirrhosis_history", "family_history_cancer", "diabetes"]
ALL_INPUT_COLS = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS

# Resolve project root (package is at <root>/liver_cancer_risk)
PKG_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(PKG_DIR, os.pardir))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _load_pipeline() -> Any:
    pipe_path = os.path.join(MODELS_DIR, "pipeline.pkl")
    if not os.path.exists(pipe_path):
        raise FileNotFoundError(f"Preprocessing pipeline not found at {pipe_path}.")
    with open(pipe_path, "rb") as f:
        return pickle.load(f)


def _load_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "nn":
        try:
            from tensorflow.keras.models import load_model  # type: ignore
        except Exception as e:
            raise RuntimeError("TensorFlow/Keras not available to load NN model") from e
        path = os.path.join(MODELS_DIR, "nn_classifier.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"NN model not found at {path}")
        model = load_model(path)
        return model
    elif model_type == "rf":
        path = os.path.join(MODELS_DIR, "random_forest.pkl")
    elif model_type == "dt":
        path = os.path.join(MODELS_DIR, "decision_tree.pkl")
    else:
        raise ValueError("model_type must be one of: 'nn', 'rf', 'dt'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _best_model_by_metrics() -> str:
    metrics_paths = {
        "rf": os.path.join(MODELS_DIR, "metrics_random_forest.json"),
        "dt": os.path.join(MODELS_DIR, "metrics_decision_tree.json"),
        "nn": os.path.join(MODELS_DIR, "metrics_nn.json"),
    }
    best_type = None
    best_auc = -1.0
    for mtype, path in metrics_paths.items():
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                auc = float(data.get("roc_auc", -1))
                if auc > best_auc:
                    best_auc = auc
                    best_type = mtype
            except Exception:
                continue
    return best_type or "rf"


def _coerce_and_validate_inputs(user_answers: Dict[str, Any]) -> pd.DataFrame:
    data: Dict[str, Any] = {col: None for col in ALL_INPUT_COLS}
    data.update(user_answers or {})

    for col in NUMERIC_COLS:
        val = data.get(col)
        if val in ("", None):
            data[col] = np.nan
        else:
            try:
                data[col] = float(val)
            except Exception:
                raise ValueError(f"Invalid numeric value for {col}: {val}")

    for col in CATEGORICAL_COLS:
        val = data.get(col)
        if val in ("", None):
            data[col] = None
        else:
            data[col] = str(val)

    for col in BINARY_COLS:
        val = data.get(col)
        if val in ("", None):
            data[col] = 0
        else:
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("1", "true", "yes", "y"): val = 1
                elif v in ("0", "false", "no", "n"): val = 0
            try:
                data[col] = int(val)
            except Exception:
                raise ValueError(f"Invalid binary value for {col}: {val}")
            if data[col] not in (0, 1):
                raise ValueError(f"Binary field {col} must be 0 or 1, got {data[col]}")

    df = pd.DataFrame([data], columns=ALL_INPUT_COLS)
    return df


def predict_user_input(user_answers: Dict[str, Any], model_preference: str = 'auto', allow_missing_labs: bool = True) -> Dict[str, Any]:
    """Predict liver cancer risk from user inputs.

    Args:
        user_answers: dict with keys matching expected columns. Missing labs allowed.
        model_preference: 'auto', 'nn', 'rf', or 'dt'.
        allow_missing_labs: if False, raise an error when lab fields are missing.

    Returns:
        dict with fields: model_type, probability, risk_label, confidence, note
    """
    chosen = _best_model_by_metrics() if model_preference == 'auto' else model_preference.lower()

    pipe = _load_pipeline()
    try:
        model = _load_model(chosen)
    except Exception:
        if chosen == 'nn':
            model = _load_model('rf')
            chosen = 'rf'
        else:
            raise

    input_df = _coerce_and_validate_inputs(user_answers)

    labs_present = True
    missing_lab_fields = []
    for lab in ["alpha_fetoprotein_level", "liver_function_score"]:
        val = input_df.iloc[0][lab]
        if pd.isna(val):
            labs_present = False
            missing_lab_fields.append(lab)

    if not labs_present and not allow_missing_labs:
        raise ValueError(f"Missing required lab fields: {', '.join(missing_lab_fields)}")

    X_t = pipe.transform(input_df)
    if hasattr(X_t, 'toarray'):
        X_t = X_t.toarray()

    if chosen == 'nn':
        from tensorflow.keras import backend as K  # type: ignore
        prob = float(model.predict(X_t, verbose=0).ravel()[0])
    else:
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(X_t)[0, 1])
        else:
            scores = model.decision_function(X_t)
            s_min, s_max = scores.min(), scores.max()
            prob = float((scores[0] - s_min) / (s_max - s_min + 1e-9))

    risk_label = 'High' if prob >= 0.5 else 'Low'
    proximity = abs(prob - 0.5) * 2.0
    base_conf = min(max(proximity, 0.0), 1.0)
    if not labs_present:
        conf = max(base_conf * 0.8, 0.0)
        note = f"AFP/Liver function labs missing ({', '.join(missing_lab_fields)}). Estimate may be less confident."
    else:
        conf = base_conf
        note = "All key labs provided."

    return {
        "model_type": chosen,
        "probability": prob,
        "risk_label": risk_label,
        "confidence": conf,
        "note": note,
    }

