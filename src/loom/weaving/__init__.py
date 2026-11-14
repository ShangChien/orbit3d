"""`loom.weaving`: Four-slot plug-and-play foundation model weaving architecture for `Uni-Loom`."""

from .conditioner import LoomConditioner
from .gp_atlas import HeteroscedasticGPAtlas1D, HeteroscedasticGPAtlas2D
from .grpo import (
    LoomGRPOTrainer,
    compute_disentangled_group_advantages,
    formal_charge_signature,
    property_distance_reward,
    zero_step_physical_reward,
)
from .lora import LoRALinear, inject_uni3dar_lora, iter_lora_parameters
from .metrics import compute_loom_metrics
from .slots import (
    AuditOracleSlot,
    GeneratorSlot,
    PotentialSlot,
    RewardOracleSlot,
    assert_not_audit_oracle,
    check_unimol2_support,
)
from .uni3dar_slot import LoomUni3DARGenerator

__all__ = [
    "AuditOracleSlot",
    "GeneratorSlot",
    "HeteroscedasticGPAtlas1D",
    "HeteroscedasticGPAtlas2D",
    "LoRALinear",
    "LoomConditioner",
    "LoomGRPOTrainer",
    "LoomUni3DARGenerator",
    "PotentialSlot",
    "RewardOracleSlot",
    "assert_not_audit_oracle",
    "check_unimol2_support",
    "compute_disentangled_group_advantages",
    "compute_loom_metrics",
    "formal_charge_signature",
    "inject_uni3dar_lora",
    "iter_lora_parameters",
    "property_distance_reward",
    "zero_step_physical_reward",
]
