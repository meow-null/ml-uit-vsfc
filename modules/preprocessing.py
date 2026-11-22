"""Tiền xử lý văn bản tiếng Việt cấu hình linh hoạt cho dữ liệu UIT-VSFC."""

import re
import string
from typing import Dict, Iterable, List, Optional

from pyvi import ViTokenizer
from sklearn.base import BaseEstimator, TransformerMixin

# Bản đồ các token đặc thù trong UIT-VSFC
SPECIAL_TOKEN_PATTERNS = {
    "colon": re.compile(r"\bcolon\b", flags=re.IGNORECASE),
    "dot": re.compile(r"\bdot\b", flags=re.IGNORECASE),
    "doublespace": re.compile(r"\bdoublespace\b", flags=re.IGNORECASE),
    "fraction": re.compile(r"\bfraction\b", flags=re.IGNORECASE),
    "wzjwz": re.compile(r"\bwzjwz\d*\b", flags=re.IGNORECASE),
}

# Các từ phủ định, chuyển ý và phó từ chỉ mức độ CẦN GIỮ LẠI khi lọc stopwords
NEGATION_AND_INTENSIFIER_WHITELIST = {
    "không",
    "chưa",
    "chẳng",
    "chả",
    "đừng",
    "chớ",
    "nhưng",
    "nhưng_mà",
    "tuy",
    "tuy_nhiên",
    "mặc_dù",
    "dù",
    "hơi",
    "quá",
    "rất",
    "lắm",
    "cực",
    "cực_kỳ",
    "cực_kì",
    "khá",
    "kém",
    "ít",
    "nhiều",
    "chưa_bao_giờ",
    "không_bao_giờ",
    "không_thể",
    "chẳng_hề",
    "không_hề",
    "thiếu",
    "thêm",
    "bớt",
    "hơn",
    "nhất",
    "vẫn",
    "còn",
    "bị",
    "được",
}

# Danh sách stopwords tiếng Việt cơ bản (đã loại trừ các từ trong whitelist)
BASE_VIETNAMESE_STOPWORDS = {
    "à",
    "ạ",
    "ai",
    "anh",
    "ấy",
    "bà",
    "bạn",
    "bao_nhiêu",
    "bằng",
    "bên",
    "bởi",
    "bởi_vì",
    "cả",
    "các",
    "cái",
    "cần",
    "càng",
    "chỉ",
    "chiếc",
    "cho",
    "chú",
    "chúng",
    "chúng_ta",
    "chúng_tôi",
    "có",
    "có_thể",
    "cô",
    "con",
    "của",
    "cùng",
    "cũng",
    "đã",
    "đang",
    "đây",
    "để",
    "đến",
    "đều",
    "đi",
    "điều",
    "đó",
    "đối_với",
    "em",
    "gì",
    "hay",
    "họ",
    "hoặc",
    "khi",
    "khoảng",
    "là",
    "lại",
    "làm",
    "lần",
    "lên",
    "lúc",
    "mà",
    "mình",
    "mỗi",
    "một",
    "một_số",
    "này",
    "nên",
    "nếu",
    "người",
    "nhau",
    "nhiều",
    "những",
    "như",
    "như_vậy",
    "nó",
    "nói",
    "nữa",
    "ở",
    "ông",
    "phải",
    "qua",
    "ra",
    "rằng",
    "rồi",
    "sau",
    "sẽ",
    "so",
    "sự",
    "tại",
    "tại_sao",
    "theo",
    "thế",
    "thì",
    "thôi",
    "tôi",
    "tới",
    "trong",
    "trên",
    "trước",
    "từ",
    "từng",
    "và",
    "vào",
    "vậy",
    "về",
    "vì",
    "việc",
    "với",
    "vừa",
    "vừa_mới",
    "xuống",
} - NEGATION_AND_INTENSIFIER_WHITELIST

PUNCTUATION_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))
MULTI_SPACE_RE = re.compile(r"\s+")


class VietnameseTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Bộ tiền xử lý văn bản tiếng Việt tương thích scikit-learn Pipeline.

    Cho phép bật/tắt linh hoạt các bước tiền xử lý phục vụ thực nghiệm ablation:
        - lowercase: Chuyển chữ thường.
        - clean_special_tokens: Làm sạch các token đặc thù của UIT-VSFC
          (colon, dot, doublespace, fraction, wzjwz\\d+).
        - remove_punctuation: Loại bỏ dấu câu ASCII và một số ký tự đặc biệt.
        - word_segment: Tách từ tiếng Việt bằng pyvi.ViTokenizer.
        - remove_stopwords: Loại bỏ stopwords cơ bản nhưng giữ lại từ phủ định
          và từ chuyển ý/mức độ quan trọng cho phân loại cảm xúc.
    """

    def __init__(
        self,
        lowercase: bool = True,
        clean_special_tokens: bool = True,
        remove_punctuation: bool = True,
        word_segment: bool = True,
        remove_stopwords: bool = False,
        extra_stopwords: Optional[Iterable[str]] = None,
    ):
        self.lowercase = lowercase
        self.clean_special_tokens = clean_special_tokens
        self.remove_punctuation = remove_punctuation
        self.word_segment = word_segment
        self.remove_stopwords = remove_stopwords
        self.extra_stopwords = tuple(extra_stopwords) if extra_stopwords else ()

    def _get_stopword_set(self) -> set:
        stopwords = set(BASE_VIETNAMESE_STOPWORDS)
        if self.extra_stopwords:
            stopwords.update(self.extra_stopwords)
        return stopwords - NEGATION_AND_INTENSIFIER_WHITELIST

    def _clean_uit_vsfc_tokens(self, text: str) -> str:
        text = SPECIAL_TOKEN_PATTERNS["colon"].sub(" ", text)
        text = SPECIAL_TOKEN_PATTERNS["dot"].sub(" ", text)
        text = SPECIAL_TOKEN_PATTERNS["doublespace"].sub(" ", text)
        text = SPECIAL_TOKEN_PATTERNS["fraction"].sub(" ", text)
        text = SPECIAL_TOKEN_PATTERNS["wzjwz"].sub(" ", text)
        return text

    def preprocess_text(self, text: str) -> str:
        """Tiền xử lý một câu đơn lẻ theo cấu hình hiện tại."""
        if text is None:
            return ""
        processed = str(text).strip()

        if self.lowercase:
            processed = processed.lower()

        if self.clean_special_tokens:
            processed = self._clean_uit_vsfc_tokens(processed)

        if self.remove_punctuation:
            processed = processed.translate(PUNCTUATION_TABLE)
            processed = re.sub(r"[“”‘’…–—•·«»]", " ", processed)

        processed = MULTI_SPACE_RE.sub(" ", processed).strip()
        if not processed:
            return ""

        if self.word_segment:
            processed = ViTokenizer.tokenize(processed)

        if self.remove_stopwords:
            stopwords = self._get_stopword_set()
            tokens = processed.split()
            filtered_tokens = [tok for tok in tokens if tok.lower() not in stopwords]
            processed = " ".join(filtered_tokens)

        return MULTI_SPACE_RE.sub(" ", processed).strip()

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        return [self.preprocess_text(text) for text in X]

    def get_config_name(self) -> str:
        """Trả về tên mô tả ngắn gọn của cấu hình hiện tại."""
        seg_str = "WordSeg=ON" if self.word_segment else "WordSeg=OFF"
        sw_str = "Stopwords=RM" if self.remove_stopwords else "Stopwords=KEEP"
        tok_str = "CleanTok=ON" if self.clean_special_tokens else "CleanTok=OFF"
        return f"{seg_str} | {sw_str} | {tok_str}"


def get_ablation_configs() -> Dict[str, VietnameseTextPreprocessor]:
    """Trả về 4 cấu hình chuẩn để khảo sát ảnh hưởng của tách từ và stopwords."""
    return {
        "C1: NoSeg + KeepSW": VietnameseTextPreprocessor(
            lowercase=True,
            clean_special_tokens=True,
            remove_punctuation=True,
            word_segment=False,
            remove_stopwords=False,
        ),
        "C2: NoSeg + RemoveSW": VietnameseTextPreprocessor(
            lowercase=True,
            clean_special_tokens=True,
            remove_punctuation=True,
            word_segment=False,
            remove_stopwords=True,
        ),
        "C3: WordSeg + KeepSW": VietnameseTextPreprocessor(
            lowercase=True,
            clean_special_tokens=True,
            remove_punctuation=True,
            word_segment=True,
            remove_stopwords=False,
        ),
        "C4: WordSeg + RemoveSW": VietnameseTextPreprocessor(
            lowercase=True,
            clean_special_tokens=True,
            remove_punctuation=True,
            word_segment=True,
            remove_stopwords=True,
        ),
    }
