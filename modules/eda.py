"""Phân tích khám phá dữ liệu (EDA) toàn diện cho tập dữ liệu UIT-VSFC."""

import os
import re
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from modules.data_loader import LABEL_MAP

# Thiết lập style chuẩn học thuật
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.unicode_minus"] = False

LABEL_ORDER = ["Tiêu cực", "Trung tính", "Tích cực"]
LABEL_PALETTE = {
    "Tiêu cực": "#d9534f",
    "Trung tính": "#f0ad4e",
    "Tích cực": "#5cb85c",
}

SPECIAL_TOKENS_REGEX = {
    "colon": re.compile(r"\bcolon\b", flags=re.IGNORECASE),
    "dot": re.compile(r"\bdot\b", flags=re.IGNORECASE),
    "doublespace": re.compile(r"\bdoublespace\b", flags=re.IGNORECASE),
    "fraction": re.compile(r"\bfraction\b", flags=re.IGNORECASE),
    "wzjwz*": re.compile(r"\bwzjwz\d*\b", flags=re.IGNORECASE),
}


def _ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def compute_dataset_overview(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Thống kê kích thước, giá trị thiếu, mẫu rỗng và trùng lặp trên 3 tập dữ liệu."""
    splits = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]
    rows = []

    for split_name, df in splits:
        n_rows = len(df)
        missing_text = int(df["text"].isna().sum())
        empty_text = int((df["text"].fillna("").str.strip() == "").sum())
        missing_label = int(df["label"].isna().sum())
        dup_text = int(df.duplicated(subset=["text"]).sum())
        dup_exact = int(df.duplicated(subset=["text", "label"]).sum())

        counts = df["label_name"].value_counts()
        row = {
            "Tập dữ liệu": split_name,
            "Số mẫu": n_rows,
            "Thiếu text": missing_text,
            "Mẫu rỗng": empty_text,
            "Thiếu label": missing_label,
            "Trùng text": dup_text,
            "Trùng (text, label)": dup_exact,
        }
        for label_name in LABEL_ORDER:
            cnt = int(counts.get(label_name, 0))
            pct = (cnt / n_rows * 100.0) if n_rows > 0 else 0.0
            row[f"{label_name} (SL)"] = cnt
            row[f"{label_name} (%)"] = round(pct, 2)
        rows.append(row)

    # Kiểm tra trùng lặp chéo giữa Train và Test (data leakage check)
    train_texts = set(train_df["text"].tolist())
    test_overlap = int(test_df["text"].isin(train_texts).sum())
    val_overlap = int(val_df["text"].isin(train_texts).sum())

    overview_df = pd.DataFrame(rows)
    overview_df.attrs["train_val_overlap"] = val_overlap
    overview_df.attrs["train_test_overlap"] = test_overlap
    return overview_df


def plot_label_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: str = "reports/figures",
    show: bool = True,
) -> str:
    """Vẽ và lưu biểu đồ phân phối nhãn cảm xúc trên 3 tập (nhấn mạnh mất cân bằng Trung tính ~5%)."""
    _ensure_dir(save_dir)

    frames = []
    for split_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        vc = df["label_name"].value_counts().reindex(LABEL_ORDER, fill_value=0)
        total = max(len(df), 1)
        for lbl, cnt in vc.items():
            frames.append(
                {
                    "Tập": split_name,
                    "Nhãn cảm xúc": lbl,
                    "Số lượng": int(cnt),
                    "Tỷ lệ (%)": cnt / total * 100.0,
                }
            )
    plot_df = pd.DataFrame(frames)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Subplot 1: Số lượng tuyệt đối
    sns.barplot(
        data=plot_df,
        x="Tập",
        y="Số lượng",
        hue="Nhãn cảm xúc",
        hue_order=LABEL_ORDER,
        palette=LABEL_PALETTE,
        ax=axes[0],
    )
    axes[0].set_title("Phân phối số lượng mẫu theo nhãn cảm xúc", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tập dữ liệu")
    axes[0].set_ylabel("Số lượng câu")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d", fontsize=8.5, padding=2)

    # Subplot 2: Tỷ lệ phần trăm (nhấn mạnh lớp Trung tính ~5%)
    sns.barplot(
        data=plot_df,
        x="Tập",
        y="Tỷ lệ (%)",
        hue="Nhãn cảm xúc",
        hue_order=LABEL_ORDER,
        palette=LABEL_PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("Tỷ lệ phần trăm nhãn (Mất cân bằng lớp Trung tính ~5%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Tập dữ liệu")
    axes[1].set_ylabel("Tỷ lệ (%)")
    axes[1].set_ylim(0, 62)
    axes[1].axhline(5.0, color="#d9534f", linestyle="--", linewidth=1.0, alpha=0.7, label="Ngưỡng 5%")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%", fontsize=8.5, padding=2)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "label_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def analyze_and_plot_sequence_lengths(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: str = "reports/figures",
    show: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Thống kê và vẽ biểu đồ phân phối độ dài câu (theo số từ và số ký tự)."""
    _ensure_dir(save_dir)

    all_frames = []
    stats_rows = []
    for split_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        tmp = df[["text", "label_name"]].copy()
        tmp["word_count"] = tmp["text"].fillna("").str.split().str.len()
        tmp["char_count"] = tmp["text"].fillna("").str.len()
        tmp["split"] = split_name
        all_frames.append(tmp)

        for metric, col in [("Số từ (Word)", "word_count"), ("Số ký tự (Char)", "char_count")]:
            s = tmp[col]
            stats_rows.append(
                {
                    "Tập": split_name,
                    "Đơn vị": metric,
                    "Mean": round(float(s.mean()), 2),
                    "Std": round(float(s.std(ddof=0)), 2),
                    "Min": int(s.min()),
                    "Median (50%)": round(float(s.median()), 1),
                    "P95": round(float(np.percentile(s, 95)), 1),
                    "P99": round(float(np.percentile(s, 99)), 1),
                    "Max": int(s.max()),
                }
            )

    combined = pd.concat(all_frames, ignore_index=True)
    stats_df = pd.DataFrame(stats_rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # Phân phối số từ theo tập
    sns.histplot(
        data=combined,
        x="word_count",
        hue="split",
        bins=45,
        kde=True,
        stat="density",
        common_norm=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Phân phối mật độ độ dài câu theo số từ", fontsize=11.5, fontweight="bold")
    axes[0, 0].set_xlabel("Số từ (khoảng trắng)")
    axes[0, 0].set_ylabel("Mật độ")
    axes[0, 0].set_xlim(0, max(60, int(np.percentile(combined["word_count"], 99.5)) + 5))

    # Phân phối số ký tự theo tập
    sns.histplot(
        data=combined,
        x="char_count",
        hue="split",
        bins=45,
        kde=True,
        stat="density",
        common_norm=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Phân phối mật độ độ dài câu theo số ký tự", fontsize=11.5, fontweight="bold")
    axes[0, 1].set_xlabel("Số ký tự")
    axes[0, 1].set_ylabel("Mật độ")
    axes[0, 1].set_xlim(0, max(250, int(np.percentile(combined["char_count"], 99.5)) + 15))

    # Boxplot số từ theo nhãn (trên tập Train)
    train_only = combined[combined["split"] == "Train"]
    sns.boxplot(
        data=train_only,
        x="label_name",
        y="word_count",
        hue="label_name",
        order=LABEL_ORDER,
        hue_order=LABEL_ORDER,
        palette=LABEL_PALETTE,
        dodge=False,
        showfliers=False,
        ax=axes[1, 0],
    )
    if axes[1, 0].get_legend() is not None:
        axes[1, 0].get_legend().remove()
    axes[1, 0].set_title("Phân phối số từ theo nhãn (Tập Train, bỏ outlier cực đoan)", fontsize=11.5, fontweight="bold")
    axes[1, 0].set_xlabel("Nhãn cảm xúc")
    axes[1, 0].set_ylabel("Số từ")

    # Boxplot số ký tự theo nhãn (trên tập Train)
    sns.boxplot(
        data=train_only,
        x="label_name",
        y="char_count",
        hue="label_name",
        order=LABEL_ORDER,
        hue_order=LABEL_ORDER,
        palette=LABEL_PALETTE,
        dodge=False,
        showfliers=False,
        ax=axes[1, 1],
    )
    if axes[1, 1].get_legend() is not None:
        axes[1, 1].get_legend().remove()
    axes[1, 1].set_title("Phân phối số ký tự theo nhãn (Tập Train, bỏ outlier cực đoan)", fontsize=11.5, fontweight="bold")
    axes[1, 1].set_xlabel("Nhãn cảm xúc")
    axes[1, 1].set_ylabel("Số ký tự")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "length_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return stats_df, save_path


def analyze_special_tokens(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: str = "reports/figures",
    show: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Kiểm tra tần suất các token đặc thù trong UIT-VSFC (colon, dot, doublespace, fraction, wzjwz...)."""
    _ensure_dir(save_dir)

    rows = []
    for split_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        texts = df["text"].fillna("").astype(str)
        n_docs = max(len(texts), 1)
        for token_name, pattern in SPECIAL_TOKENS_REGEX.items():
            match_counts = texts.apply(lambda s: len(pattern.findall(s)))
            total_occurrences = int(match_counts.sum())
            doc_freq = int((match_counts > 0).sum())
            rows.append(
                {
                    "Tập": split_name,
                    "Token đặc thù": token_name,
                    "Tổng số lần xuất hiện": total_occurrences,
                    "Số câu chứa token": doc_freq,
                    "Tỷ lệ câu chứa (%)": round(doc_freq / n_docs * 100.0, 2),
                }
            )

    special_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    sns.barplot(
        data=special_df,
        x="Token đặc thù",
        y="Tổng số lần xuất hiện",
        hue="Tập",
        ax=axes[0],
    )
    axes[0].set_title("Tổng số lần xuất hiện token đặc thù UIT-VSFC", fontsize=11.5, fontweight="bold")
    axes[0].set_ylabel("Số lần xuất hiện")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d", fontsize=8, padding=2)

    sns.barplot(
        data=special_df,
        x="Token đặc thù",
        y="Tỷ lệ câu chứa (%)",
        hue="Tập",
        ax=axes[1],
    )
    axes[1].set_title("Tỷ lệ câu (%) chứa token đặc thù", fontsize=11.5, fontweight="bold")
    axes[1].set_ylabel("Tỷ lệ câu (%)")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%", fontsize=8, padding=2)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "special_tokens_frequency.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return special_df, save_path


def _top_ngrams_for_texts(
    texts: pd.Series,
    ngram_range: Tuple[int, int],
    top_k: int = 15,
) -> pd.DataFrame:
    if len(texts) == 0:
        return pd.DataFrame(columns=["ngram", "count"])
    vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b")
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["ngram", "count"])
    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(freqs)[::-1][:top_k]
    return pd.DataFrame({"ngram": vocab[top_idx], "count": freqs[top_idx]})


def plot_top_ngrams_by_label(
    train_df: pd.DataFrame,
    top_k: int = 15,
    save_dir: str = "reports/figures",
    show: bool = True,
) -> Dict[str, Any]:
    """Vẽ biểu đồ Top 15 unigram và Top 15 bigram phổ biến nhất theo từng nhãn cảm xúc."""
    _ensure_dir(save_dir)
    outputs: Dict[str, Any] = {"unigrams": {}, "bigrams": {}, "paths": []}

    # Loại bỏ các token kỹ thuật UIT-VSFC khi thống kê từ vựng ngữ nghĩa cho dễ quan sát
    cleaned_series = (
        train_df["text"]
        .fillna("")
        .str.lower()
        .str.replace(r"\b(colon|dot|doublespace|fraction|wzjwz\d*)\b", " ", regex=True)
    )

    for title_tag, ngram_range, fname in [
        ("Unigram (1-gram)", (1, 1), "top_unigrams_by_label.png"),
        ("Bigram (2-gram)", (2, 2), "top_bigrams_by_label.png"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
        key_name = "unigrams" if ngram_range == (1, 1) else "bigrams"

        for idx, label_name in enumerate(LABEL_ORDER):
            mask = train_df["label_name"] == label_name
            top_df = _top_ngrams_for_texts(cleaned_series[mask], ngram_range=ngram_range, top_k=top_k)
            outputs[key_name][label_name] = top_df

            ax = axes[idx]
            if not top_df.empty:
                sns.barplot(
                    data=top_df,
                    x="count",
                    y="ngram",
                    color=LABEL_PALETTE[label_name],
                    ax=ax,
                )
                for container in ax.containers:
                    ax.bar_label(container, fmt="%d", fontsize=8, padding=2)
            ax.set_title(f"Top {top_k} {title_tag} — {label_name}", fontsize=11.5, fontweight="bold")
            ax.set_xlabel("Tần suất")
            ax.set_ylabel("")

        plt.tight_layout()
        save_path = os.path.join(save_dir, fname)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        outputs["paths"].append(save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return outputs


def run_full_eda(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: str = "reports/figures",
    show: bool = True,
) -> Dict[str, Any]:
    """Chạy toàn bộ pipeline EDA và lưu các biểu đồ chất lượng cao vào reports/figures/."""
    _ensure_dir(save_dir)

    overview_df = compute_dataset_overview(train_df, val_df, test_df)
    label_plot_path = plot_label_distribution(train_df, val_df, test_df, save_dir=save_dir, show=show)
    length_stats_df, length_plot_path = analyze_and_plot_sequence_lengths(
        train_df, val_df, test_df, save_dir=save_dir, show=show
    )
    special_tokens_df, special_plot_path = analyze_special_tokens(
        train_df, val_df, test_df, save_dir=save_dir, show=show
    )
    ngram_outputs = plot_top_ngrams_by_label(train_df, top_k=15, save_dir=save_dir, show=show)

    return {
        "overview": overview_df,
        "length_stats": length_stats_df,
        "special_tokens": special_tokens_df,
        "ngram_outputs": ngram_outputs,
        "saved_figures": [
            label_plot_path,
            length_plot_path,
            special_plot_path,
            *ngram_outputs["paths"],
        ],
    }
