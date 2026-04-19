# src/utils/config_schema.py
"""
Pydantic v2 schema for the project configuration.

The DotDict-based loader in ``src/utils/config.py`` is convenient but
silently swallows typos: a misspelled key like ``lora.rnak: 8`` becomes
an attribute that doesn't exist, and the next attribute access raises
``AttributeError`` somewhere deep in the training loop.

This module defines an explicit Pydantic schema for every section the
project uses, plus a single ``validate_config(cfg_dict)`` entry point
that:

  1. Drops fields the schema does not know about into a side dict
     (allowed: keys like ``hardware`` and ``project_root`` that are
     injected at runtime, not authored by humans).
  2. Raises ``ConfigValidationError`` on any other typo, missing
     required field, or out-of-range value.
  3. Returns the validated, normalized dict — ready to be wrapped in a
     ``DotDict``.

The schema is intentionally permissive on optional sections (features,
quantization, etc.) and strict on the core ones (model, training, loss,
paths). Update it whenever you add a new top-level setting.

Usage::

    from src.utils.config_schema import validate_config
    validated = validate_config(raw_dict)  # raises on typos
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigValidationError(ValueError):
    """Raised when the project configuration fails schema validation."""


# ---------------------------------------------------------------------------
# Project / paths
# ---------------------------------------------------------------------------


class ProjectSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str | None = None
    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class PathsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str
    raw_dir: str
    processed_dir: str
    splits_dir: str
    models_dir: str
    reports_dir: str
    figures_dir: str | None = None
    mlflow_dir: str = "mlruns"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class DataSourceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organism_id: int
    reviewed: bool = True


class DataSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[DataSourceItem] = Field(default_factory=list)
    uniprot_api_url: str | None = None
    fields: str | None = None
    max_sequence_length: int = Field(default=2048, ge=1)
    min_sequence_length: int = Field(default=30, ge=1)


class ProcessingSection(BaseModel):
    # Permissive: location_groups is an arbitrary nested mapping that
    # users routinely extend, so we don't lock its shape.
    model_config = ConfigDict(extra="allow")

    min_samples_per_class: int = Field(default=50, ge=1)
    multi_label: bool = True
    valid_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"


class SplittingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["homology", "random"] = "homology"
    sequence_identity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    coverage_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    test_size: float = Field(default=0.15, ge=0.0, le=1.0)
    val_size: float = Field(default=0.15, ge=0.0, le=1.0)
    random_fallback: bool = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BackboneSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    embedding_dim: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    max_position_embeddings: int = Field(ge=1)
    use_sdpa_attention: bool = False


class LoRASection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1, le=512)
    alpha: int = Field(ge=1)
    dropout: float = Field(ge=0.0, le=1.0)
    target_modules: list[str]
    bias: Literal["none", "all", "lora_only"] = "none"


class ClassifierSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128])
    dropout: float = Field(default=0.3, ge=0.0, le=1.0)
    activation: Literal["gelu", "relu"] = "gelu"


class QuantizationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["nf4", "fp4", "int8"] = "nf4"
    compute_dtype: Literal["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"] = "bfloat16"
    double_quant: bool = True


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: BackboneSection
    lora: LoRASection
    classifier: ClassifierSection
    pooling: Literal["mean", "cls", "mean_cls", "light_attention"] = "mean"
    pooling_dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    quantization: QuantizationSection = Field(default_factory=QuantizationSection)


class MultiTaskSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    loss_weight: float = Field(default=0.2, ge=0.0, le=10.0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class FocalLossSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gamma: float = Field(default=2.0, ge=0.0)
    alpha: float | None = None  # None means auto-compute


class HierarchicalLossSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    weight: float = Field(default=0.1, ge=0.0)


class LossSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focal: FocalLossSection
    hierarchical: HierarchicalLossSection


# ---------------------------------------------------------------------------
# Features (external biology features)
# ---------------------------------------------------------------------------


class BiophysicalFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    properties: list[str] = Field(default_factory=list)


class SignalPeptideFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    binary_path: str | None = None


class TransmembraneFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    binary_path: str | None = None


class FeaturesSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    biophysical: BiophysicalFeatures = Field(default_factory=BiophysicalFeatures)
    signal_peptide: SignalPeptideFeatures = Field(default_factory=SignalPeptideFeatures)
    transmembrane: TransmembraneFeatures = Field(default_factory=TransmembraneFeatures)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class OptimizerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["adamw", "paged_adamw_8bit"] = "adamw"
    lr: float = Field(gt=0.0)
    head_lr: float | None = Field(default=None, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])


class SchedulerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "cosine_warmup"
    warmup_steps_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    min_lr: float = Field(default=1.0e-6, ge=0.0)


class ExperimentSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "protein-loc-v2"
    tracking_uri: str = "mlruns"
    log_every_n_steps: int = Field(default=10, ge=1)
    save_top_k: int = Field(default=3, ge=1)


class TrainingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_epochs: int = Field(ge=1)
    patience: int = Field(default=5, ge=1)
    gradient_clip_val: float = Field(default=1.0, ge=0.0)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    optimizer: OptimizerSection
    scheduler: SchedulerSection
    batch_size: int | None = Field(default=None, ge=1)
    max_sequence_length: int | None = Field(default=None, ge=1)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    use_length_bucketing: bool = False
    length_bucket_jitter: float = Field(default=0.05, ge=0.0, le=1.0)
    auto_batch_size: bool = False
    precision: Literal["bf16-mixed", "16-mixed", "32", "16", "bf16"] = "bf16-mixed"
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    experiment: ExperimentSection
    deterministic: bool = True


# ---------------------------------------------------------------------------
# Inference / serving (only validated when present)
# ---------------------------------------------------------------------------


class InferenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str | None = None
    device: Literal["auto", "cuda", "cpu"] = "auto"
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    batch_size: int = Field(default=32, ge=1)
    max_sequence_length: int = Field(default=1024, ge=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=3, ge=1)


class ServingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_requests_per_minute: int = Field(default=60, ge=1)
    max_sequence_length: int = Field(default=2048, ge=1)
    model_warmup: bool = True


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


class ProjectConfig(BaseModel):
    """The full validated project configuration."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectSection
    paths: PathsSection
    data: DataSection | None = None
    processing: ProcessingSection | None = None
    splitting: SplittingSection | None = None
    model: ModelSection
    loss: LossSection
    features: FeaturesSection = Field(default_factory=FeaturesSection)
    training: TrainingSection | None = None
    inference: InferenceSection | None = None
    serving: ServingSection | None = None
    multi_task: MultiTaskSection = Field(default_factory=MultiTaskSection)


# Keys that ``load_config`` injects at runtime and that are NOT part of
# the human-authored YAML. We pop them before validation and re-attach
# them afterwards.
_RUNTIME_KEYS = ("hardware", "project_root")


def validate_config(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the merged configuration dict against the project schema.

    Args:
        cfg_dict: The plain-dict configuration produced by
            ``load_config`` *before* it is wrapped in a DotDict.

    Returns:
        A new dict with the validated, normalized values plus any
        runtime keys that were stripped before validation.

    Raises:
        ConfigValidationError: If validation fails. The message lists
            every offending field with a human-readable path.
    """
    # Pop runtime-only keys
    runtime_payload = {k: cfg_dict[k] for k in _RUNTIME_KEYS if k in cfg_dict}
    payload = {k: v for k, v in cfg_dict.items() if k not in _RUNTIME_KEYS}

    try:
        validated = ProjectConfig.model_validate(payload)
    except ValidationError as e:
        raise ConfigValidationError(_format_validation_errors(e)) from e

    result = validated.model_dump(exclude_none=True)
    result.update(runtime_payload)
    return result


def _format_validation_errors(err: ValidationError) -> str:
    """Render a Pydantic ValidationError as a multi-line, human message."""
    lines = ["Configuration validation failed with the following errors:"]
    for e in err.errors():
        loc = ".".join(str(item) for item in e["loc"])
        msg = e["msg"]
        if e["type"] == "extra_forbidden":
            lines.append(
                f"  - Unknown key '{loc}'. Did you typo it? Add it to the schema if it is new."
            )
        else:
            lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)
