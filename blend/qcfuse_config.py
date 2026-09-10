"""QCFuse configuration for the SSD-backed blend runner."""

DIGEST_INDEX_METHOD = "kvzip"
DIGEST_RATIO = 0.15
DEFAULT_BLEND_RATIO = 0.5
DEFAULT_CONTEXT_N_SINK = 4
DEFAULT_CRITICAL_LAYERS = 3
FUSERAG_DIGEST_RATIO = 0.0
PROPHETKV_DIGEST_RATIO = 1.0
BLEND_BASELINES = ("ours", "fuserag", "prophetkv")
SUPPORTED_BASELINES = ("fullcomp",) + BLEND_BASELINES
BASELINE_DIGEST_RATIOS = {
    "ours": DIGEST_RATIO,
    "fuserag": FUSERAG_DIGEST_RATIO,
    "prophetkv": PROPHETKV_DIGEST_RATIO,
}

# Model-specific Top-10 critical layers. Values are 0-based layer ids and are
# consumed by the runtime critical_layers request argument.
MODEL_TOP10_CRITICAL_LAYERS = {
    "llama3.1-8b": [14, 13, 16, 18, 16, 20, 19, 10, 15, 22],
    "mistral-7b": [15, 19, 18, 16, 14, 17, 12, 11, 20, 9],
    "qwen3-14b": [24, 26, 21, 25, 18, 28, 29, 22, 23, 20],
    "qwen3-32b": [48, 45, 50, 42, 49, 47, 53, 52, 46, 51],
    "qwen3-8b": [20, 21, 19, 17, 24, 23, 26, 14, 22, 19],
}
