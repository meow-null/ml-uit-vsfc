"""Đánh giá mô hình, trực quan hóa Confusion Matrix, so sánh cấu hình và Error Analysis."""

import os
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_NAMES = ["Tiêu cực", "Trung tính", "Tích cực"]
LABEL_IDS = [0, 1, 2]


def _ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def evaluate_predictions(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    model_name: str = "Model",
    split_name: str = "Test",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Tính đầy đủ các chỉ số đánh giá (lấy Macro F1 làm thước đo chính).

    Trả về dict gồm Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
    và F1 riêng cho từng nhãn (Tiêu cực, Trung tính, Tích cực).
    """
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    macro_p = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    macro_r = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    per_class_f1 = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=LABEL_IDS,
        average=None,
        zero_division=0,
    )

    if verbose:
        print(f"\n=== [{split_name}] {model_name} ===")
        print(
            f"Accuracy: {acc:.4f} | Macro Precision: {macro_p:.4f} | "
            f"Macro Recall: {macro_r:.4f} | Macro F1: {macro_f1:.4f} | "
            f"Weighted F1: {weighted_f1:.4f}"
        )
        print(
            classification_report(
                y_true_arr,
                y_pred_arr,
                labels=LABEL_IDS,
                target_names=LABEL_NAMES,
                digits=4,
                zero_division=0,
            )
        )

    return {
        "Mô hình": model_name,
        "Split": split_name,
        "Accuracy": round(float(acc), 4),
        "Macro Precision": round(float(macro_p), 4),
        "Macro Recall": round(float(macro_r), 4),
        "Macro F1": round(float(macro_f1), 4),
        "Weighted F1": round(float(weighted_f1), 4),
        "F1 (Tiêu cực)": round(float(per_class_f1[0]), 4),
        "F1 (Trung tính)": round(float(per_class_f1[1]), 4),
        "F1 (Tích cực)": round(float(per_class_f1[2]), 4),
    }


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    model_name: str = "Best Model",
    save_path: str = "reports/figures/confusion_matrix_best.png",
    show: bool = True,
) -> str:
    """Vẽ và lưu Confusion Matrix (chuẩn hóa phần trăm kèm số lượng mẫu thực tế)."""
    _ensure_parent_dir(save_path)
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)

    cm_raw = confusion_matrix(y_true_arr, y_pred_arr, labels=LABEL_IDS)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_raw.astype(float), np.where(row_sums == 0, 1, row_sums)) * 100.0

    annot = np.empty_like(cm_raw, dtype=object)
    for i in range(cm_raw.shape[0]):
        for j in range(cm_raw.shape[1]):
            annot[i, j] = f"{cm_pct[i, j]:.1f}%\n({cm_raw[i, j]})"

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar_kws={"label": "Tỷ lệ theo nhãn thật (%)"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (Chuẩn hóa %) — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Nhãn dự đoán (Predicted)", fontsize=10.5)
    ax.set_ylabel("Nhãn thực tế (True)", fontsize=10.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric_col: str = "Macro F1",
    save_path: str = "reports/figures/model_comparison.png",
    show: bool = True,
) -> str:
    """Vẽ biểu đồ cột ngang so sánh tổng hợp các cấu hình đã thử nghiệm."""
    _ensure_parent_dir(save_path)
    if results_df.empty:
        raise ValueError("results_df rỗng, không thể vẽ biểu đồ so sánh.")

    sorted_df = results_df.sort_values(by=metric_col, ascending=True).reset_index(drop=True)
    n_models = len(sorted_df)
    fig_height = max(5.0, 0.45 * n_models + 1.5)

    fig, axes = plt.subplots(1, 2, figsize=(16, fig_height), sharey=True)

    # Biểu đồ 1: Macro F1 vs Accuracy
    melted = sorted_df.melt(
        id_vars=["Mô hình"],
        value_vars=["Macro F1", "Accuracy"],
        var_name="Chỉ số",
        value_name="Điểm",
    )
    sns.barplot(
        data=melted,
        x="Điểm",
        y="Mô hình",
        hue="Chỉ số",
        palette={"Macro F1": "#1f77b4", "Accuracy": "#ff7f0e"},
        ax=axes[0],
    )
    axes[0].set_title("So sánh Macro F1 và Accuracy giữa các mô hình", fontsize=12, fontweight="bold")
    axes[0].set_xlim(0.0, 1.05)
    axes[0].set_xlabel("Điểm số (0 - 1)")
    axes[0].set_ylabel("Cấu hình mô hình")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    # Biểu đồ 2: F1 lớp thiểu số Trung tính
    if "F1 (Trung tính)" in sorted_df.columns:
        sns.barplot(
            data=sorted_df,
            x="F1 (Trung tính)",
            y="Mô hình",
            color="#f0ad4e",
            ax=axes[1],
        )
        axes[1].set_title("Khả năng nhận diện lớp thiểu số: F1 (Trung tính)", fontsize=12, fontweight="bold")
        axes[1].set_xlim(0.0, 1.05)
        axes[1].set_xlabel("F1-score lớp Trung tính")
        axes[1].set_ylabel("")
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def perform_error_analysis(
    texts: Iterable[str],
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: Optional[np.ndarray] = None,
    n_samples: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Xuất bảng 10–15 mẫu dự đoán sai tiêu biểu (đa dạng cặp nhãn nhầm lẫn)
    kèm văn bản gốc, nhãn thật và nhãn dự đoán của mô hình tốt nhất.
    """
    text_list = list(texts)
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)

    records: List[Dict[str, Any]] = []
    for idx, (txt, t, p) in enumerate(zip(text_list, y_true_arr, y_pred_arr)):
        if t != p:
            true_name = LABEL_NAMES[t]
            pred_name = LABEL_NAMES[p]
            conf = float(np.max(y_prob[idx])) if y_prob is not None else np.nan
            records.append(
                {
                    "Index": idx,
                    "Văn bản gốc": txt,
                    "Nhãn thật": true_name,
                    "Nhãn dự đoán": pred_name,
                    "Cặp nhầm lẫn": f"{true_name} -> {pred_name}",
                    "Độ dài (từ)": len(str(txt).split()),
                    "Độ tự tin": round(conf, 4) if not np.isnan(conf) else None,
                }
            )

    err_df = pd.DataFrame(records)
    if err_df.empty:
        return err_df

    # Lấy mẫu phân tầng theo cặp nhầm lẫn để bao quát hết các dạng lỗi tiêu biểu
    rng = np.random.default_rng(random_state)
    selected_indices: List[int] = []
    grouped = list(err_df.groupby("Cặp nhầm lẫn", sort=True))
    per_group = max(1, n_samples // max(len(grouped), 1))

    for _, group in grouped:
        take = min(per_group, len(group))
        chosen = rng.choice(group.index.to_numpy(), size=take, replace=False)
        selected_indices.extend(chosen.tolist())

    remaining_needed = n_samples - len(selected_indices)
    if remaining_needed > 0:
        pool = err_df.index.difference(selected_indices).to_numpy()
        if len(pool) > 0:
            extra = rng.choice(pool, size=min(remaining_needed, len(pool)), replace=False)
            selected_indices.extend(extra.tolist())

    result_df = err_df.loc[selected_indices[:n_samples]].copy()
    result_df = result_df.sort_values(by=["Cặp nhầm lẫn", "Index"]).reset_index(drop=True)
    if y_prob is None:
        result_df = result_df.drop(columns=["Độ tự tin"])
    return result_df
