"""
End-to-end example:
1. Load a time-series dataset from CSV.
2. Build supervised samples (lookback -> horizon).
3. Train an A-CNN-LSTM forecasting model.
4. Predict on the test set.
5. Classify each test sample into S-FEC / M-FEC / I-FEC
   based on fluctuation metrics (CV and RR) computed from the
   ground-truth test sequences.
6. Optionally compute metrics per FEC class.

This script assumes:
- The CSV file contains at least one numeric column named "power"
  as the prediction target.
- All numeric columns are used as input features; "power" is the
  target for forecasting.
"""

import os
import numpy as np
import pandas as pd
from math import sqrt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, Input


# ============================================================
# 1. Configuration
# ============================================================

CSV_PATH = "data.csv"      # path to your CSV file
TARGET_COL = "power"       # target column name
LOOKBACK = 36              # number of historical time steps
HORIZON = 12               # number of future steps to predict

TEST_SIZE = 0.2            # ratio for test set
VAL_SIZE = 0.1             # ratio for validation (from train)
RANDOM_SEED = 42


# ============================================================
# 2. Utility functions
# ============================================================

def load_dataset(csv_path, target_col):
    """
    Load dataset from CSV and split into features and target.
    Assumes target_col is numeric.
    All numeric columns are used as features.
    """
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if target_col not in numeric_df.columns:
        raise ValueError(f"Target column '{target_col}' not found or not numeric in the CSV file.")

    y = numeric_df[target_col].values  # shape (T,)
    X = numeric_df.values              # shape (T, F)
    return X, y, numeric_df.columns


def create_supervised_samples(X, y, lookback, horizon):
    """
    Create supervised samples for sequence-to-sequence forecasting.

    Input:
        X: ndarray, shape (T, F)
        y: ndarray, shape (T,)
        lookback: int, number of past time steps
        horizon: int, number of future steps

    Output:
        X_seq: ndarray, shape (N, lookback, F)
        y_seq: ndarray, shape (N, horizon)
    """
    T, F = X.shape
    X_seq, y_seq = [], []
    max_start = T - lookback - horizon + 1

    for start in range(max_start):
        end_lookback = start + lookback
        end_horizon = end_lookback + horizon

        X_seq.append(X[start:end_lookback, :])
        y_seq.append(y[end_lookback:end_horizon])

    X_seq = np.array(X_seq)  # (N, lookback, F)
    y_seq = np.array(y_seq)  # (N, horizon)
    return X_seq, y_seq


def split_train_val_test(X_seq, y_seq, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split sequence samples into train, validation, and test sets.
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, shuffle=True, random_state=random_state
    )

    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, shuffle=True, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_a_cnn_lstm_model(input_shape, horizon):
    """
    Build an Attention-CNN-LSTM model.

    input_shape: (lookback, num_features)
    horizon: number of forecast steps (output dimension)
    """
    inputs = Input(shape=input_shape)

    # CNN block
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Simple attention mechanism (Dense-based)
    attention_scores = layers.Dense(x.shape[-1], activation="softmax", name="attention_scores")(x)
    x = layers.Multiply(name="attention_mul")([x, attention_scores])

    # BiLSTM block
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(horizon)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="A_CNN_LSTM")
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["mae"]
    )
    return model


def compute_regression_metrics(y_true, y_pred):
    """
    Compute RMSE, MAE, R2 for 2D arrays (N, H).
    Flatten along time for aggregate metrics.
    """
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    rmse = sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    return rmse, mae, r2


def classify_fec(testY, eps=1e-6):
    """
    Classify each sample in testY into S-FEC / M-FEC / I-FEC based on:
    - CV: coefficient of variation
    - RR: normalized average absolute slope

    testY: ndarray, shape (N, H)
    returns:
        cv, rr, mask_S, mask_M, mask_I
    """
    mean_y = np.mean(testY, axis=1)         # (N,)
    std_y = np.std(testY, axis=1)           # (N,)
    cv = std_y / (mean_y + eps)

    diff_y = np.diff(testY, axis=1)         # (N, H-1)
    mean_abs_diff = np.mean(np.abs(diff_y), axis=1)
    rr = mean_abs_diff / (mean_y + eps)

    cv_q1, cv_q2 = np.quantile(cv, [1.0 / 3.0, 2.0 / 3.0])
    rr_q1, rr_q2 = np.quantile(rr, [1.0 / 3.0, 2.0 / 3.0])

    mask_S = (cv <= cv_q1) & (rr <= rr_q1)
    mask_I = (cv >= cv_q2) & (rr >= rr_q2)
    mask_M = ~(mask_S | mask_I)

    return cv, rr, mask_S, mask_M, mask_I


# ============================================================
# 3. Main pipeline
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # 3.1 Load dataset
    X_raw, y_raw, feature_names = load_dataset(CSV_PATH, TARGET_COL)
    print("Loaded data:", X_raw.shape, "features")
    print("Target:", TARGET_COL)

    # 3.2 Optional normalization (standardization using mean/std of entire series)
    X_mean = X_raw.mean(axis=0, keepdims=True)
    X_std = X_raw.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_raw - X_mean) / X_std

    # 3.3 Create supervised samples
    X_seq, y_seq = create_supervised_samples(X_norm, y_raw, LOOKBACK, HORIZON)
    print("Supervised samples:", X_seq.shape, y_seq.shape)

    # 3.4 Split into train / val / test
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X_seq, y_seq, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # 3.5 Build model
    input_shape = (LOOKBACK, X_seq.shape[-1])
    model = build_a_cnn_lstm_model(input_shape, HORIZON)
    model.summary()

    # 3.6 Train model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # 3.7 Predict on test set
    y_pred_test = model.predict(X_test)
    print("Predictions on test:", y_pred_test.shape)

    # 3.8 Compute overall metrics
    rmse_all, mae_all, r2_all = compute_regression_metrics(y_test, y_pred_test)
    print("\nOverall test metrics:")
    print(f"RMSE: {rmse_all:.4f}, MAE: {mae_all:.4f}, R2: {r2_all:.4f}")

    # 3.9 FEC classification based on test ground truth sequences
    cv, rr, mask_S, mask_M, mask_I = classify_fec(y_test)

    print("\nFEC class counts:")
    print(f"S-FEC: {mask_S.sum()} samples")
    print(f"M-FEC: {mask_M.sum()} samples")
    print(f"I-FEC: {mask_I.sum()} samples")

    # 3.10 Compute metrics per FEC class (using the same A-CNN-LSTM predictions)
    def print_metrics_for_mask(name, mask):
        if mask.sum() == 0:
            print(f"\n{name}: no samples.")
            return
        rmse, mae, r2 = compute_regression_metrics(y_test[mask], y_pred_test[mask])
        print(f"\n{name} metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.44f}")

    print_metrics_for_mask("S-FEC", mask_S)
    print_metrics_for_mask("M-FEC", mask_M)
    print_metrics_for_mask("I-FEC", mask_I)


if __name__ == "__main__":
    # Ensure the CSV path exists before running
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV file not found at '{CSV_PATH}'. Please update CSV_PATH."
        )
    main()