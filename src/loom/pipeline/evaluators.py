"""Shared configured property/force evaluators for proposal workflows."""
import torch
from ..oracles.dpa4_oracle import InternalDPA4PropertyOracle
from ..oracles.fingerprint_oracle import FingerprintOracle
from ..oracles.applicability import MorganDomain
from ..physics.factory import build_forcefield


def load_evaluators(cfg, element_table, coordinate_dtype=torch.float32):
    paths, settings = cfg['mount_paths'], cfg['oracle_config']
    relaxation = cfg.get('stage5_relax_config', {})
    mode = settings['oracle_mode']
    backend = relaxation.get('mlip_backend', settings.get('mlip_backend',
        'hyff' if mode == 'internal_hyff' else 'dpa4'))
    if mode in {'internal_dpa4', 'internal_hyff'} and mode != 'internal_' + backend:
        raise ValueError('Shared property Oracle and forcefield must use the same backend')
    ff = build_forcefield(backend, paths['forcefield_path'], element_table, cfg.get('device', 'cuda:0'),
                         tier=relaxation.get('hyff_tier', settings.get('hyff_tier', 'vivace')),
                         route=relaxation.get('hyff_route', 'reference'))
    if mode in {'internal_dpa4', 'internal_hyff'}:
        oracle = InternalDPA4PropertyOracle.load_from_dir(paths['property_checkpoint_dir'], ff)
        oracle.coordinate_dtype = coordinate_dtype
    elif mode == 'fingerprint':
        oracle = FingerprintOracle.load_from_dir(paths['property_checkpoint_dir'])
    else:
        raise ValueError('Supported property modes: internal_dpa4, internal_hyff, fingerprint')
    if oracle.target_cols != settings['target_cols']:
        raise ValueError('Property target order mismatch')
    if 'target_units' in settings and list(settings['target_units']) != list(oracle.target_units):
        raise ValueError('Property target units mismatch')
    domain_path = settings.get('applicability_domain_path')
    domain = MorganDomain.load(domain_path) if domain_path else None
    return ff, oracle, domain
