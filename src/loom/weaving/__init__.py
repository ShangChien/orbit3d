"""`loom.weaving`: Platform-grade 3-layer predictor-agnostic weaving & Grounded RSI architecture (`loom v2.0`)."""

from .conditioner import LoomConditioner
from .critic_bus import CompositeCriticBus
from .flowmol_slot import FlowMolReplayDataset, LoomFlowMolGenerator
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
from .octree_process_critic import (
    NUM_COARSE_GEOM_FEATS,
    NUM_ELEM_CLOUD_FEATS,
    OctreeElementProcessCritic,
    OctreeProcessCritic,
    ProcessCriticTrainStats,
    TwoStageOctreeProcessCritic,
    compute_coarse_voxel_geometry,
    compute_element_cloud_features,
    fit_octree_element_process_critic,
    fit_octree_process_critic,
)
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
    build_generator_slot,
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
    "FlowMolReplayDataset",
    "GeneratorSlot",
    "HeavyAtomValenceHead",
    "HeteroscedasticGPAtlas1D",
    "HeteroscedasticGPAtlas2D",
    "IncrementalCalibrator",
    "InnerRSIRunner",
    "LoRALinear",
    "LoomConditioner",
    "LoomFlowMolGenerator",
    "LoomGRPOTrainer",
    "LoomUni3DARGenerator",
    "NUM_COARSE_GEOM_FEATS",
    "NUM_ELEM_CLOUD_FEATS",
    "ObjectiveChannelSpec",
    "OctreeElementProcessCritic",
    "OctreeProcessCritic",
    "OuterRSIAcquisition",
    "PhysicsConstrainedCoordRefiner",
    "PotentialSlot",
    "PredictorAsset",
    "ProcessCriticTrainStats",
    "RSIDiagnoser",
    "RSIDiagnosticThresholds",
    "ResidualCalibratedAsset",
    "RewardOracleSlot",
    "SlotOracleAsset",
    "TaskWeavingSpec",
    "TwoStageOctreeProcessCritic",
    "UniMol3DAsset",
    "assert_not_audit_oracle",
    "build_generator_slot",
    "check_unimol2_support",
    "compute_coarse_voxel_geometry",
    "compute_disentangled_group_advantages",
    "compute_element_cloud_features",
    "compute_loom_metrics",
    "fit_octree_element_process_critic",
    "fit_octree_process_critic",
    "formal_charge_signature",
    "inject_uni3dar_lora",
    "is_clean_organic_charge_state",
    "iter_lora_parameters",
    "property_distance_reward",
    "zero_step_physical_reward",
]
