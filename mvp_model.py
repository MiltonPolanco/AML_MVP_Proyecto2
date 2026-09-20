"""Inferencia ligera del sistema AML de dos etapas usado por el MVP."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


EMBED_DIM = 4
HIDDEN_DIM = 64
LATENT_DIM = 24
POSITION_DIM = 8


class SequenceEncoder(nn.Module):
    def __init__(self, n_types: int, num_features: int):
        super().__init__()
        self.type_embedding = nn.Embedding(n_types, EMBED_DIM)
        self.gru = nn.GRU(num_features + EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.to_latent = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x_num, x_type, mask, return_sequence=False):
        embedded_type = self.type_embedding(x_type)
        inputs = torch.cat([x_num, embedded_type], dim=-1)
        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed)
        latent = self.to_latent(hidden[-1])
        if not return_sequence:
            return latent
        sequence_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=mask.shape[1]
        )
        return latent, sequence_output


class SequenceDecoder(nn.Module):
    def __init__(self, max_len: int, n_types: int, num_features: int):
        super().__init__()
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, POSITION_DIM)
        self.initial_hidden = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.gru = nn.GRU(LATENT_DIM + POSITION_DIM, HIDDEN_DIM, batch_first=True)
        self.numeric_output = nn.Linear(HIDDEN_DIM, num_features)
        self.type_output = nn.Linear(HIDDEN_DIM, n_types)

    def forward(self, latent):
        batch_size = latent.shape[0]
        positions = torch.arange(self.max_len, device=latent.device)
        position_features = self.position_embedding(positions)
        position_features = position_features.unsqueeze(0).expand(batch_size, -1, -1)
        repeated_latent = latent.unsqueeze(1).expand(-1, self.max_len, -1)
        decoder_input = torch.cat([repeated_latent, position_features], dim=-1)
        hidden0 = self.initial_hidden(latent).unsqueeze(0)
        output, _ = self.gru(decoder_input, hidden0)
        return self.numeric_output(output), self.type_output(output)


class SequenceAutoencoder(nn.Module):
    def __init__(self, max_len: int, n_types: int, num_features: int):
        super().__init__()
        self.encoder = SequenceEncoder(n_types, num_features)
        self.decoder = SequenceDecoder(max_len, n_types, num_features)

    def forward(self, x_num, x_type, mask):
        latent = self.encoder(x_num, x_type, mask)
        numeric_hat, type_logits = self.decoder(latent)
        return latent, numeric_hat, type_logits


class AttentionClassifier(nn.Module):
    def __init__(self, encoder: SequenceEncoder, use_anomaly_score: bool):
        super().__init__()
        self.encoder = encoder
        self.use_anomaly_score = use_anomaly_score
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        classifier_input = LATENT_DIM + HIDDEN_DIM + int(use_anomaly_score)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x_num, x_type, mask, anomaly_score, return_attention=False):
        latent, sequence_output = self.encoder(
            x_num, x_type, mask, return_sequence=True
        )
        attention_logits = self.attention(sequence_output).squeeze(-1)
        attention_logits = attention_logits.masked_fill(mask == 0, -1e4)
        attention_weights = torch.softmax(attention_logits, dim=1)
        context = (sequence_output * attention_weights.unsqueeze(-1)).sum(dim=1)
        features = [latent, context]
        if self.use_anomaly_score:
            features.append(anomaly_score.unsqueeze(-1))
        logits = self.classifier(torch.cat(features, dim=1)).squeeze(-1)
        if return_attention:
            return logits, attention_weights
        return logits


@dataclass
class Prediction:
    probability: float
    threshold: float
    anomaly_score: float
    anomaly_threshold: float
    is_alert: bool
    transactions: pd.DataFrame


def load_models(artifact_dir: str | Path):
    """Carga los mismos checkpoints seleccionados en el notebook final."""
    artifact_dir = Path(artifact_dir)
    stage_a_checkpoint = torch.load(
        artifact_dir / "stage_a_autoencoder.pt", map_location="cpu", weights_only=False
    )
    classifier_paths = list(artifact_dir.glob("stage_b_transferencia_mas_anom*.pt"))
    if len(classifier_paths) != 1:
        raise FileNotFoundError("No se encontró un único checkpoint del modelo final.")
    stage_b_checkpoint = torch.load(
        classifier_paths[0], map_location="cpu", weights_only=False
    )

    max_len = int(stage_a_checkpoint["max_len"])
    n_types = len(stage_a_checkpoint["type_map"])
    num_features = len(stage_a_checkpoint["feature_cols"]) - 1

    autoencoder = SequenceAutoencoder(max_len, n_types, num_features)
    autoencoder.load_state_dict(stage_a_checkpoint["model_state"])
    autoencoder.eval()

    encoder = SequenceEncoder(n_types, num_features)
    classifier = AttentionClassifier(
        encoder, use_anomaly_score=bool(stage_b_checkpoint["use_anomaly_score"])
    )
    classifier.load_state_dict(stage_b_checkpoint["model_state"])
    classifier.eval()
    return autoencoder, classifier, stage_a_checkpoint, stage_b_checkpoint


def engineer_account_features(
    account_transactions: pd.DataFrame,
    stage_a_checkpoint: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Replica la preparación temporal del notebook para una sola cuenta."""
    frame = account_transactions.sort_values("step", kind="mergesort").copy()
    type_map = stage_a_checkpoint["type_map"]
    unknown_types = set(frame["type"]) - set(type_map)
    if unknown_types:
        raise ValueError(f"Tipos de transacción desconocidos: {sorted(unknown_types)}")

    frame["log_amount"] = np.log1p(frame["amount"]).astype("float32")
    frame["type_idx"] = frame["type"].map(type_map).astype("int64")
    frame["delta_t_hours"] = frame["step"].diff().fillna(0).astype("float32")
    frame["delta_t"] = frame["delta_t_hours"]
    frame["log_amount_change"] = frame["log_amount"].diff().fillna(0).astype("float32")

    transaction_index = np.arange(len(frame), dtype=np.float32)
    previous_sum = frame["log_amount"].cumsum() - frame["log_amount"]
    previous_mean = previous_sum / pd.Series(
        np.where(transaction_index == 0, np.nan, transaction_index), index=frame.index
    )
    frame["amount_vs_history"] = (
        frame["log_amount"] - previous_mean
    ).fillna(0).astype("float32")

    hour = (frame["step"] % 24).astype("float32")
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")

    numeric_cols = [
        "log_amount",
        "delta_t",
        "log_amount_change",
        "amount_vs_history",
        "hour_sin",
        "hour_cos",
    ]
    frame[numeric_cols] = (
        (frame[numeric_cols] - stage_a_checkpoint["scaler_mean"])
        / stage_a_checkpoint["scaler_scale"]
    ).astype("float32")

    max_len = int(stage_a_checkpoint["max_len"])
    window = frame.tail(max_len).copy().reset_index(drop=True)
    window["position"] = np.arange(1, len(window) + 1)
    length = len(window)

    x_num = np.zeros((1, max_len, len(numeric_cols)), dtype=np.float32)
    x_type = np.zeros((1, max_len), dtype=np.int64)
    mask = np.zeros((1, max_len), dtype=np.float32)
    x_num[0, :length] = window[numeric_cols].to_numpy(dtype=np.float32)
    x_type[0, :length] = window["type_idx"].to_numpy(dtype=np.int64)
    mask[0, :length] = 1

    return (
        torch.from_numpy(x_num),
        torch.from_numpy(x_type),
        torch.from_numpy(mask),
        window,
    )


def predict_account(
    account_transactions: pd.DataFrame,
    autoencoder: SequenceAutoencoder,
    classifier: AttentionClassifier,
    stage_a_checkpoint: dict,
    stage_b_checkpoint: dict,
) -> Prediction:
    x_num, x_type, mask, window = engineer_account_features(
        account_transactions, stage_a_checkpoint
    )
    valid_count = mask.sum(dim=1)
    num_features = x_num.shape[-1]

    with torch.inference_mode():
        _, numeric_hat, type_logits = autoencoder(x_num, x_type, mask)
        numeric_error = ((numeric_hat - x_num) ** 2 * mask.unsqueeze(-1)).sum((1, 2))
        numeric_error = numeric_error / (valid_count * num_features + 1e-8)
        type_error = nn.functional.cross_entropy(
            type_logits.reshape(-1, len(stage_a_checkpoint["type_map"])),
            x_type.reshape(-1),
            reduction="none",
        ).reshape(mask.shape)
        type_error = (type_error * mask).sum(1) / (valid_count + 1e-8)
        anomaly_score = numeric_error + 0.5 * type_error

        anomaly_z = (
            anomaly_score - float(stage_b_checkpoint["anomaly_mean"])
        ) / float(stage_b_checkpoint["anomaly_std"])
        logits, attention = classifier(
            x_num, x_type, mask, anomaly_z, return_attention=True
        )
        calibrated_logit = (
            float(stage_b_checkpoint["calibration_coef"]) * logits
            + float(stage_b_checkpoint["calibration_intercept"])
        )
        probability = torch.sigmoid(calibrated_logit).item()

    length = len(window)
    window["attention"] = attention[0, :length].numpy()
    threshold = float(stage_b_checkpoint["threshold"])
    anomaly_value = float(anomaly_score.item())
    return Prediction(
        probability=probability,
        threshold=threshold,
        anomaly_score=anomaly_value,
        anomaly_threshold=float(stage_a_checkpoint["anomaly_threshold"]),
        is_alert=probability >= threshold,
        transactions=window,
    )


def explain_prediction(prediction: Prediction) -> str:
    """Genera una explicación breve basada en las señales visibles del modelo."""
    top = prediction.transactions.sort_values("attention", ascending=False).iloc[0]
    signals = []
    if top["delta_t_hours"] <= 1 and int(top["position"]) > 1:
        signals.append("ocurrió como parte de una ráfaga de operaciones")
    if abs(float(top["amount_vs_history"])) > 1:
        signals.append("el monto se apartó del historial de la cuenta")
    if top["type"] in ("TRANSFER", "CASH_OUT"):
        signals.append(f"fue una operación {top['type']}")
    if not signals:
        signals.append("combinó monto, momento y tipo de forma poco habitual")

    attention = float(top["attention"])
    anomaly_relation = (
        "superó" if prediction.anomaly_score >= prediction.anomaly_threshold else "no superó"
    )
    decision = (
        "La cuenta se prioriza para revisión humana."
        if prediction.is_alert
        else "La cuenta no supera el umbral de priorización."
    )
    return (
        f"La transacción {int(top['position'])} concentró {attention:.1%} de la atención "
        f"del modelo: {'; '.join(signals)}. El score de anomalía {anomaly_relation} su "
        f"umbral de referencia. {decision} Este resultado orienta el análisis y no "
        "constituye por sí solo una conclusión de lavado de dinero."
    )
