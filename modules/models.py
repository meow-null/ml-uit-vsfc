"""Các pipeline mô hình: ML truyền thống, Lai (PhoBERT .npy), và Fine-tuning End-to-End."""

import contextlib
import copy
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyvi import ViTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from modules.evaluation import evaluate_predictions
from modules.feature_extraction import DEFAULT_PHOBERT_MODEL, TraditionalFeatureExtractor


def set_all_seeds(seed: int = 42) -> None:
    """Cố định seed cho Python, NumPy và PyTorch để đảm bảo tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pipeline 1: Machine Learning truyền thống (trên BoW / TF-IDF)
# ---------------------------------------------------------------------------
def get_traditional_models(
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Khởi tạo 3 bộ phân loại ML truyền thống: MultinomialNB, LogisticRegression, LinearSVC."""
    return {
        "MultinomialNB": MultinomialNB(alpha=1.0),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "LinearSVC": LinearSVC(
            C=1.0,
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
            dual="auto",
        ),
    }


def train_and_evaluate_traditional(
    X_train,
    y_train: Iterable[int],
    X_val,
    y_val: Iterable[int],
    X_test,
    y_test: Iterable[int],
    feature_label: str = "TF-IDF (1,2)",
    class_weight: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Huấn luyện và đánh giá các mô hình ML truyền thống trên ma trận BoW/TF-IDF."""
    y_train_arr = np.asarray(list(y_train), dtype=int)
    models = get_traditional_models(class_weight=class_weight, random_state=random_state)
    cw_tag = "balanced" if class_weight == "balanced" else "none"

    rows = []
    trained = {}
    for clf_name, clf in models.items():
        if isinstance(clf, MultinomialNB) and class_weight == "balanced":
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_arr)
            clf.fit(X_train, y_train_arr, sample_weight=sample_weights)
        else:
            clf.fit(X_train, y_train_arr)

        val_pred = clf.predict(X_val)
        test_pred = clf.predict(X_test)
        full_name = f"{feature_label} + {clf_name} (cw={cw_tag})"

        val_metrics = evaluate_predictions(y_val, val_pred, model_name=full_name, split_name="Val", verbose=False)
        test_metrics = evaluate_predictions(
            y_test, test_pred, model_name=full_name, split_name="Test", verbose=verbose
        )
        test_metrics["Val Macro F1"] = val_metrics["Macro F1"]
        test_metrics["Nhóm"] = "Truyền thống"
        rows.append(test_metrics)
        trained[full_name] = {
            "model": clf,
            "val_pred": val_pred,
            "test_pred": test_pred,
            "metrics": test_metrics,
        }

    return pd.DataFrame(rows), trained


def run_traditional_experiments(
    train_texts: Iterable[str],
    y_train: Iterable[int],
    val_texts: Iterable[str],
    y_val: Iterable[int],
    test_texts: Iterable[str],
    y_test: Iterable[int],
    vectorizer_configs: Optional[List[Dict[str, Any]]] = None,
    class_weights: Tuple[Optional[str], ...] = (None, "balanced"),
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Chạy ma trận thực nghiệm BoW/TF-IDF × Unigram/Bigram × NB/LR/SVM × class_weight."""
    if vectorizer_configs is None:
        vectorizer_configs = [
            {"label": "BoW (1,1)", "method": "bow", "ngram_range": (1, 1), "max_features": 15000, "min_df": 2},
            {"label": "TF-IDF (1,1)", "method": "tfidf", "ngram_range": (1, 1), "max_features": 15000, "min_df": 2},
            {"label": "TF-IDF (1,2)", "method": "tfidf", "ngram_range": (1, 2), "max_features": 20000, "min_df": 2},
        ]

    train_list = list(train_texts)
    val_list = list(val_texts)
    test_list = list(test_texts)

    all_dfs = []
    all_artifacts: Dict[str, Any] = {}

    for cfg in vectorizer_configs:
        extractor = TraditionalFeatureExtractor(
            method=cfg["method"],
            ngram_range=cfg["ngram_range"],
            max_features=cfg.get("max_features", 15000),
            min_df=cfg.get("min_df", 2),
        )
        X_tr = extractor.fit_transform(train_list)
        X_va = extractor.transform(val_list)
        X_te = extractor.transform(test_list)

        for cw in class_weights:
            df_res, trained = train_and_evaluate_traditional(
                X_tr,
                y_train,
                X_va,
                y_val,
                X_te,
                y_test,
                feature_label=cfg["label"],
                class_weight=cw,
                random_state=random_state,
                verbose=verbose,
            )
            all_dfs.append(df_res)
            for name, bundle in trained.items():
                bundle["vectorizer"] = extractor.vectorizer
                all_artifacts[name] = bundle

    return pd.concat(all_dfs, ignore_index=True), all_artifacts


# ---------------------------------------------------------------------------
# Pipeline 2: Mô hình Lai (trên PhoBERT embeddings .npy đã lưu)
# ---------------------------------------------------------------------------
def get_hybrid_models(
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Khởi tạo các bộ phân loại chạy trên PhoBERT .npy embeddings: LR, LinearSVC, MLP."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5,
                        max_iter=1500,
                        class_weight=class_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "LinearSVC": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        C=0.1,
                        max_iter=3000,
                        class_weight=class_weight,
                        random_state=random_state,
                        dual="auto",
                    ),
                ),
            ]
        ),
        "MLP (256-64)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        batch_size=128,
                        max_iter=60,
                        early_stopping=True,
                        n_iter_no_change=6,
                        validation_fraction=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def run_hybrid_experiments(
    X_train: np.ndarray,
    y_train: Iterable[int],
    X_val: np.ndarray,
    y_val: Iterable[int],
    X_test: np.ndarray,
    y_test: Iterable[int],
    class_weights: Tuple[Optional[str], ...] = (None, "balanced"),
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Huấn luyện và đánh giá LR, LinearSVC, MLP trên vector PhoBERT .npy."""
    y_train_arr = np.asarray(list(y_train), dtype=int)
    rows = []
    artifacts: Dict[str, Any] = {}

    for cw in class_weights:
        cw_tag = "balanced" if cw == "balanced" else "none"
        models = get_hybrid_models(class_weight=cw, random_state=random_state)
        for clf_name, pipe in models.items():
            # MLPClassifier trong scikit-learn không hỗ trợ tham số class_weight
            if clf_name.startswith("MLP"):
                if cw == "balanced":
                    continue
                full_name = f"PhoBERT (.npy) + {clf_name}"
            else:
                full_name = f"PhoBERT (.npy) + {clf_name} (cw={cw_tag})"

            pipe.fit(X_train, y_train_arr)
            val_pred = pipe.predict(X_val)
            test_pred = pipe.predict(X_test)
            test_prob = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

            val_metrics = evaluate_predictions(
                y_val, val_pred, model_name=full_name, split_name="Val", verbose=False
            )
            test_metrics = evaluate_predictions(
                y_test, test_pred, model_name=full_name, split_name="Test", verbose=verbose
            )
            test_metrics["Val Macro F1"] = val_metrics["Macro F1"]
            test_metrics["Nhóm"] = "Lai (PhoBERT .npy)"
            rows.append(test_metrics)
            artifacts[full_name] = {
                "model": pipe,
                "val_pred": val_pred,
                "test_pred": test_pred,
                "test_prob": test_prob,
                "metrics": test_metrics,
            }

    return pd.DataFrame(rows), artifacts


# ---------------------------------------------------------------------------
# Pipeline 3: Học sâu End-to-End Fine-tuning PhoBERT (+5% điểm cộng)
# ---------------------------------------------------------------------------
class _SentimentTextDataset(Dataset):
    """Dataset PyTorch cho phân loại câu với PhoBERT."""

    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _amp_autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(enabled=True)
    return contextlib.nullcontext()


def _make_grad_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=True)
        return torch.cuda.amp.GradScaler(enabled=True)
    return None


class PhoBERTFineTuner:
    """
    Fine-tuning trực tiếp vinai/phobert-base-v2 theo kiểu End-to-End.

    Hỗ trợ:
        - AdamW + Linear Warmup Scheduler.
        - Class-weighted CrossEntropyLoss để cải thiện lớp thiểu số Trung tính.
        - Tự động lưu checkpoint tốt nhất theo Validation Macro F1.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_PHOBERT_MODEL,
        num_labels: int = 3,
        max_length: int = 128,
        word_segment: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.word_segment = word_segment
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def _segment_texts(self, texts: Iterable[str]) -> List[str]:
        out = []
        for t in texts:
            s = "" if t is None else str(t).strip()
            if not s:
                s = " "
            elif self.word_segment:
                s = ViTokenizer.tokenize(s)
            out.append(s)
        return out

    def _predict_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with _amp_autocast(self.device, enabled=True):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits.float(), dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_preds), np.vstack(all_probs)

    def fit_and_evaluate(
        self,
        train_texts: Iterable[str],
        y_train: Iterable[int],
        val_texts: Iterable[str],
        y_val: Iterable[int],
        test_texts: Iterable[str],
        y_test: Iterable[int],
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        use_class_weights: bool = True,
        checkpoint_dir: str = "checkpoints/phobert_finetuned",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Huấn luyện 2–3 epochs, chọn checkpoint tốt nhất theo Val Macro F1 và đánh giá trên Test."""
        set_all_seeds(seed)
        y_train_arr = np.asarray(list(y_train), dtype=int)
        y_val_arr = np.asarray(list(y_val), dtype=int)
        y_test_arr = np.asarray(list(y_test), dtype=int)

        tr_seg = self._segment_texts(train_texts)
        va_seg = self._segment_texts(val_texts)
        te_seg = self._segment_texts(test_texts)

        tr_ds = _SentimentTextDataset(tr_seg, y_train_arr, self.tokenizer, self.max_length)
        va_ds = _SentimentTextDataset(va_seg, y_val_arr, self.tokenizer, self.max_length)
        te_ds = _SentimentTextDataset(te_seg, y_test_arr, self.tokenizer, self.max_length)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=batch_size * 2, shuffle=False)

        if use_class_weights:
            cls_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.array([0, 1, 2]),
                y=y_train_arr,
            )
            # Giảm nhẹ độ gắt căn bậc hai để cân bằng giữa Recall Trung tính và Precision tổng thể
            cls_weights = np.sqrt(cls_weights)
            weight_tensor = torch.tensor(cls_weights, dtype=torch.float32, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = max(1, len(tr_loader) * epochs)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scaler = _make_grad_scaler(self.device, enabled=True)

        best_val_f1 = -1.0
        history = []

        print(f"[PhoBERT Fine-tune] Thiết bị: {self.device} | Epochs: {epochs} | Class weights: {use_class_weights}")
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(tr_loader, desc=f"Epoch {epoch}/{epochs}"):
                optimizer.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with _amp_autocast(self.device, enabled=True):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                running_loss += float(loss.item())

            avg_loss = running_loss / max(len(tr_loader), 1)
            val_pred, _ = self._predict_loader(va_loader)
            val_metrics = evaluate_predictions(
                y_val_arr,
                val_pred,
                model_name=f"PhoBERT Fine-tune (Epoch {epoch})",
                split_name="Val",
                verbose=False,
            )
            val_macro_f1 = val_metrics["Macro F1"]
            history.append(
                {
                    "Epoch": epoch,
                    "Train Loss": round(avg_loss, 4),
                    "Val Accuracy": val_metrics["Accuracy"],
                    "Val Macro F1": val_macro_f1,
                    "Val F1 (Trung tính)": val_metrics["F1 (Trung tính)"],
                }
            )
            print(
                f"Epoch {epoch}/{epochs} — Train Loss: {avg_loss:.4f} | "
                f"Val Acc: {val_metrics['Accuracy']:.4f} | "
                f"Val Macro F1: {val_macro_f1:.4f} | "
                f"Val F1 (Trung tính): {val_metrics['F1 (Trung tính)']:.4f}"
            )

            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.model.save_pretrained(checkpoint_dir)
                    self.tokenizer.save_pretrained(checkpoint_dir)

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        val_pred_best, _ = self._predict_loader(va_loader)
        test_pred, test_prob = self._predict_loader(te_loader)
        model_label = f"PhoBERT End-to-End Fine-tune (cw={'sqrt-bal' if use_class_weights else 'none'})"
        test_metrics = evaluate_predictions(
            y_test_arr,
            test_pred,
            model_name=model_label,
            split_name="Test",
            verbose=True,
        )
        test_metrics["Val Macro F1"] = round(float(best_val_f1), 4)
        test_metrics["Nhóm"] = "Deep Learning (End-to-End)"

        return {
            "model_name": model_label,
            "history": pd.DataFrame(history),
            "val_pred": val_pred_best,
            "test_pred": test_pred,
            "test_prob": test_prob,
            "metrics": test_metrics,
        }


def finetune_phobert_end_to_end(
    train_texts: Iterable[str],
    y_train: Iterable[int],
    val_texts: Iterable[str],
    y_val: Iterable[int],
    test_texts: Iterable[str],
    y_test: Iterable[int],
    model_name: str = DEFAULT_PHOBERT_MODEL,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 128,
    use_class_weights: bool = True,
    word_segment: bool = True,
    checkpoint_dir: str = "checkpoints/phobert_finetuned",
    seed: int = 42,
) -> Dict[str, Any]:
    """Hàm wrapper tiện lợi để chạy Experiment 4 (Fine-tuning PhoBERT End-to-End)."""
    tuner = PhoBERTFineTuner(
        model_name=model_name,
        num_labels=3,
        max_length=max_length,
        word_segment=word_segment,
    )
    return tuner.fit_and_evaluate(
        train_texts=train_texts,
        y_train=y_train,
        val_texts=val_texts,
        y_val=y_val,
        test_texts=test_texts,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_class_weights=use_class_weights,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
