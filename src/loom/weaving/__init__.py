"""`loom.weaving`: Platform-grade 3-layer predictor-agnostic weaving & Grounded RSI architecture (`loom v2.0`)."""

from .conditioner import LoomConditioner
from .critic_bus import CompositeCriticBus
from .gp_atlas import HeteroscedasticGPAtlas1D, HeteroscedasticGPAtlas2D
from .grpo import (
    LoomGRPOTrainer,
    compute_disentangled_group_advantages,
    formal_charge_signature,
    is_clean_organic_charge_state,
    property_distance_reward,
    zero_step_physical_reward,
)
from .lora import LoRALinear, inject_uni3dar_lora, iter_lora_parameters
from .metrics import compute_loom_metrics
from .registry import (
    AssetCapabilities,
    AssetRegistry,
    BuiltInRuleAsset,
    CallablePredictorAsset,
    FingerprintQSARAsset,
    PhysicsConstrainedCoordRefiner,
    PredictorAsset,
    ResidualCalibratedAsset,
    SlotOracleAsset,
    UniMol3DAsset,
)
from .rsi import (
    IncrementalCalibrator,
    InnerRSIRunner,
    OuterRSIAcquisition,
    RSIDiagnoser,
    RSIDiagnosticThresholds,
)
from .slots import (
    AuditOracleSlot,
    GeneratorSlot,
    PotentialSlot,
    RewardOracleSlot,
    assert_not_audit_oracle,
    check_unimol2_support,
)
from .spec import (
    AuditorChannelSpec,
    ConstraintChannelSpec,
    ObjectiveChannelSpec,
    TaskWeavingSpec,
)
from .uni3dar_slot import HeavyAtomValenceHead, LoomUni3DARGenerator

__all__ = [
    "AssetCapabilities",
    "AssetRegistry",
    "AuditOracleSlot",
    "AuditorChannelSpec",
    "BuiltInRuleAsset",
    "CallablePredictorAsset",
    "CompositeCriticBus",
    "ConstraintChannelSpec",
    "FingerprintQSARAsset",
    "GeneratorSlot",
    "HeavyAtomValenceHead",
    "HeteroscedasticGPAtlas1D",
    "HeteroscedasticGPAtlas2D",
    "IncrementalCalibrator",
    "InnerRSIRunner",
    "LoRALinear",
    "LoomConditioner",
    "LoomGRPOTrainer",
    "LoomUni3DARGenerator",
    "ObjectiveChannelSpec",
    "OuterRSIAcquisition",
    "PhysicsConstrainedCoordRefiner",
    "PotentialSlot",
    "PredictorAsset",
    "RSIDiagnoser",
    "RSIDiagnosticThresholds",
    "ResidualCalibratedAsset",
    "RewardOracleSlot",
    "SlotOracleAsset",
    "TaskWeavingSpec",
    "UniMol3DAsset",
    "assert_not_audit_oracle",
    "check_unimol2_support",
    "compute_disentangled_group_advantages",
    "compute_loom_metrics",
    "formal_charge_signature",
    "inject_uni3dar_lora",
    "is_clean_organic_charge_state",
    "iter_lora_parameters",
    "property_distance_reward",
    "zero_step_physical_reward",
]
