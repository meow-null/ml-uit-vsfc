"""Tải và chuẩn hóa tập dữ liệu UIT-VSFC từ Hugging Face."""

from typing import Optional, Tuple

import pandas as pd
from datasets import DatasetDict, load_dataset

DATASET_ID = "uitnlp/vietnamese_students_feedback"
LABEL_MAP = {
    0: "Tiêu cực",
    1: "Trung tính",
    2: "Tích cực",
}


def _normalize_split(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột và kiểu dữ liệu về (text, label, label_name)."""
    rename_map = {}
    if "sentence" in df.columns and "text" not in df.columns:
        rename_map["sentence"] = "text"
    if "sentiment" in df.columns and "label" not in df.columns:
        rename_map["sentiment"] = "label"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Không tìm thấy cột 'text'/'sentence' hoặc 'label'/'sentiment' trong dữ liệu. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    out_df = df[["text", "label"]].copy()
    out_df["text"] = out_df["text"].fillna("").astype(str)
    out_df["label"] = out_df["label"].astype(int)
    out_df["label_name"] = out_df["label"].map(LABEL_MAP)

    if out_df["label_name"].isna().any():
        unknown_labels = out_df.loc[out_df["label_name"].isna(), "label"].unique().tolist()
        raise ValueError(f"Phát hiện nhãn cảm xúc không hợp lệ: {unknown_labels}")

    return out_df.reset_index(drop=True)


def _load_dataset_with_fallbacks(cache_dir: Optional[str] = None) -> DatasetDict:
    """Tải DatasetDict bằng refs/convert/parquet, fallback trust_remote_code và parquet URL."""
    errors = []

    try:
        return load_dataset(
            DATASET_ID,
            revision="refs/convert/parquet",
            cache_dir=cache_dir,
        )
    except Exception as exc:
        errors.append(f"refs/convert/parquet: {exc}")

    try:
        return load_dataset(
            DATASET_ID,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        errors.append(f"trust_remote_code=True: {exc}")

    try:
        base_url = f"hf://datasets/{DATASET_ID}@refs/convert/parquet/default"
        data_files = {
            "train": f"{base_url}/train/0000.parquet",
            "validation": f"{base_url}/validation/0000.parquet",
            "test": f"{base_url}/test/0000.parquet",
        }
        return load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)
    except Exception as exc:
        errors.append(f"parquet URL fallback: {exc}")

    raise RuntimeError(
        "Không thể tải tập dữ liệu UIT-VSFC từ Hugging Face:\n- " + "\n- ".join(errors)
    )


def load_uit_vsfc(
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tải tập dữ liệu UIT-VSFC và trả về 3 DataFrame: (train_df, val_df, test_df).

    Mỗi DataFrame gồm 3 cột:
        - text (str): Câu phản hồi của sinh viên.
        - label (int): Nhãn số nguyên (0: Tiêu cực, 1: Trung tính, 2: Tích cực).
        - label_name (str): Tên nhãn tiếng Việt.
    """
    dataset = _load_dataset_with_fallbacks(cache_dir=cache_dir)

    val_key = "validation" if "validation" in dataset else "val"
    if "train" not in dataset or val_key not in dataset or "test" not in dataset:
        raise KeyError(
            f"Dataset thiếu split cần thiết. Các split hiện có: {list(dataset.keys())}"
        )

    train_df = _normalize_split(dataset["train"].to_pandas())
    val_df = _normalize_split(dataset[val_key].to_pandas())
    test_df = _normalize_split(dataset["test"].to_pandas())

    return train_df, val_df, test_df
