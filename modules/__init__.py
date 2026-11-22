"""Các module hỗ trợ pipeline phân loại cảm xúc UIT-VSFC."""

from modules.data_loader import LABEL_MAP, load_uit_vsfc
from modules.eda import (
    analyze_and_plot_sequence_lengths,
    analyze_special_tokens,
    compute_dataset_overview,
    plot_label_distribution,
    plot_top_ngrams_by_label,
    run_full_eda,
)
from modules.evaluation import (
    LABEL_NAMES,
    evaluate_predictions,
    perform_error_analysis,
    plot_confusion_matrix,
    plot_model_comparison,
)
from modules.feature_extraction import (
    PhoBERTEmbeddingExtractor,
    TraditionalFeatureExtractor,
    build_traditional_vectorizer,
    extract_and_save_phobert_embeddings,
    load_embeddings,
)
from modules.models import (
    PhoBERTFineTuner,
    finetune_phobert_end_to_end,
    get_hybrid_models,
    get_traditional_models,
    run_hybrid_experiments,
    run_traditional_experiments,
    set_all_seeds,
    train_and_evaluate_traditional,
)
from modules.preprocessing import (
    NEGATION_AND_INTENSIFIER_WHITELIST,
    VietnameseTextPreprocessor,
    get_ablation_configs,
)

__all__ = [
    "LABEL_MAP",
    "LABEL_NAMES",
    "load_uit_vsfc",
    "compute_dataset_overview",
    "plot_label_distribution",
    "analyze_and_plot_sequence_lengths",
    "analyze_special_tokens",
    "plot_top_ngrams_by_label",
    "run_full_eda",
    "VietnameseTextPreprocessor",
    "NEGATION_AND_INTENSIFIER_WHITELIST",
    "get_ablation_configs",
    "TraditionalFeatureExtractor",
    "build_traditional_vectorizer",
    "PhoBERTEmbeddingExtractor",
    "extract_and_save_phobert_embeddings",
    "load_embeddings",
    "set_all_seeds",
    "get_traditional_models",
    "train_and_evaluate_traditional",
    "run_traditional_experiments",
    "get_hybrid_models",
    "run_hybrid_experiments",
    "PhoBERTFineTuner",
    "finetune_phobert_end_to_end",
    "evaluate_predictions",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "perform_error_analysis",
]
