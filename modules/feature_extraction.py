"""Trích xuất đặc trưng: Nhánh truyền thống (BoW/TF-IDF) và Nhánh PhoBERT (.npy)."""

import contextlib
import os
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

DEFAULT_PHOBERT_MODEL = "vinai/phobert-base-v2"


def build_traditional_vectorizer(
    method: str = "tfidf",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: Optional[int] = 15000,
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> Union[CountVectorizer, TfidfVectorizer]:
    """Khởi tạo CountVectorizer (BoW) hoặc TfidfVectorizer với cấu hình linh hoạt."""
    method_norm = method.strip().lower()
    token_pattern = r"(?u)\b\w+\b"

    if method_norm == "bow":
        return CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            token_pattern=token_pattern,
        )
    if method_norm == "tfidf":
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            token_pattern=token_pattern,
        )
    raise ValueError(f"Phương pháp '{method}' không hợp lệ. Chỉ hỗ trợ 'bow' hoặc 'tfidf'.")


class TraditionalFeatureExtractor:
    """Wrapper thống nhất cho Bag-of-Words và TF-IDF."""

    def __init__(
        self,
        method: str = "tfidf",
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = 15000,
        min_df: int = 2,
        sublinear_tf: bool = True,
    ):
        self.method = method
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.vectorizer = build_traditional_vectorizer(
            method=method,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        )

    def fit_transform(self, texts: Iterable[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Iterable[str]):
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()


def _autocast_context(device: torch.device, use_fp16: bool):
    """Trả về context manager mixed-precision an toàn giữa các phiên bản PyTorch."""
    if device.type == "cuda" and use_fp16:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(enabled=True)
    return contextlib.nullcontext()


class PhoBERTEmbeddingExtractor:
    """
    Trích xuất sentence embeddings bằng vinai/phobert-base-v2 với Mean Pooling.

    Hỗ trợ GPU (FP16 autocast + batching) trên Google Colab T4 và tự fallback về CPU.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_PHOBERT_MODEL,
        max_length: int = 128,
        batch_size: int = 64,
        use_fp16: bool = True,
        word_segment: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.word_segment = word_segment
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Tính trung bình các token embeddings có trọng số attention_mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _prepare_texts(self, texts: Iterable[str]) -> List[str]:
        prepared = []
        for text in texts:
            s = "" if text is None else str(text).strip()
            if not s:
                s = " "
            elif self.word_segment:
                s = ViTokenizer.tokenize(s)
            prepared.append(s)
        return prepared

    def encode(self, texts: Iterable[str], desc: str = "Extracting PhoBERT") -> np.ndarray:
        """Mã hóa danh sách câu thành ma trận numpy float32 có kích thước (N, 768)."""
        prepared_texts = self._prepare_texts(texts)
        if len(prepared_texts) == 0:
            hidden_size = getattr(self.model.config, "hidden_size", 768)
            return np.empty((0, hidden_size), dtype=np.float32)

        all_embeddings = []
        total_batches = (len(prepared_texts) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for start_idx in tqdm(
                range(0, len(prepared_texts), self.batch_size),
                total=total_batches,
                desc=desc,
            ):
                batch_texts = prepared_texts[start_idx : start_idx + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with _autocast_context(self.device, self.use_fp16):
                    outputs = self.model(**encoded)
                    pooled = self._mean_pooling(
                        outputs.last_hidden_state,
                        encoded["attention_mask"],
                    )

                all_embeddings.append(pooled.float().cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings).astype(np.float32)


def extract_and_save_phobert_embeddings(
    train_texts: Iterable[str],
    val_texts: Iterable[str],
    test_texts: Iterable[str],
    output_dir: str = "features",
    model_name: str = DEFAULT_PHOBERT_MODEL,
    batch_size: int = 64,
    max_length: int = 128,
    use_fp16: bool = True,
    word_segment: bool = True,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trích xuất Mean Pooling embeddings từ PhoBERT và lưu ra:
        - features/phobert_train.npy
        - features/phobert_val.npy
        - features/phobert_test.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "phobert_train.npy")
    val_path = os.path.join(output_dir, "phobert_val.npy")
    test_path = os.path.join(output_dir, "phobert_test.npy")

    if (
        not force_recompute
        and os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    ):
        print(f"[PhoBERT] Đã tìm thấy file .npy trong '{output_dir}/', đang đọc trực tiếp...")
        return load_embeddings(features_dir=output_dir)

    extractor = PhoBERTEmbeddingExtractor(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        use_fp16=use_fp16,
        word_segment=word_segment,
    )
    print(f"[PhoBERT] Thiết bị trích xuất: {extractor.device} | FP16: {use_fp16 and extractor.device.type == 'cuda'}")

    X_train = extractor.encode(train_texts, desc="PhoBERT Train")
    X_val = extractor.encode(val_texts, desc="PhoBERT Val")
    X_test = extractor.encode(test_texts, desc="PhoBERT Test")

    np.save(train_path, X_train.astype(np.float32))
    np.save(val_path, X_val.astype(np.float32))
    np.save(test_path, X_test.astype(np.float32))

    print(
        f"[PhoBERT] Đã lưu thành công:\n"
        f"  - {train_path}: {X_train.shape} ({X_train.dtype})\n"
        f"  - {val_path}: {X_val.shape} ({X_val.dtype})\n"
        f"  - {test_path}: {X_test.shape} ({X_test.dtype})"
    )
    return X_train, X_val, X_test


def load_embeddings(
    features_dir: str = "features",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đọc lại 3 ma trận PhoBERT embeddings đã lưu từ thư mục features/."""
    train_path = os.path.join(features_dir, "phobert_train.npy")
    val_path = os.path.join(features_dir, "phobert_val.npy")
    test_path = os.path.join(features_dir, "phobert_test.npy")

    for path in (train_path, val_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy file embedding '{path}'. "
                f"Hãy chạy extract_and_save_phobert_embeddings() trước."
            )

    X_train = np.load(train_path).astype(np.float32)
    X_val = np.load(val_path).astype(np.float32)
    X_test = np.load(test_path).astype(np.float32)
    return X_train, X_val, X_test
