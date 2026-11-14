"""A fixed-budget terminal gate; evaluate properties on the accepted geometry."""
import time
import numpy as np
from .standard_relaxer import relax_conformer_standard_energy
from ..generator.molecular_state import mol3d_to_validated_rdkit_mol


def property_snapshot(oracle, molecule):
    result=oracle.evaluate_3d_mols_batched([molecule])
    if result['errors'][0]:raise ValueError(result['errors'][0])
    prediction=np.asarray(result['mean'][0]);force=np.asarray(result['forces_eV_A'][0])
    if not np.isfinite(prediction).all() or not np.isfinite(force).all():
        raise ValueError('Nonfinite terminal evaluation')
    norms=np.linalg.norm(force,axis=-1)
    return {'prediction_eV':prediction.tolist(),'std_eV':result['std'][0].tolist(),
        'energy_eV':float(result['energy_eV'][0]),'rms_force_eVA':float(np.sqrt(np.mean(norms**2))),
        'max_force_eVA':float(norms.max())}


def evaluate_terminal(instance, forcefield, oracle, *, max_atoms=None):
    """148 optimization calls plus two endpoint measurements, without target guidance.

    Initial molecule validation belongs to the caller. Failed structures are not
    substituted; the caller retains the attempted proposal in its denominator.
    """
    started=time.perf_counter();history=[];row={'physical_accepted':False,'total_dpa_calls':0}
    final=instance
    if max_atoms is not None:
        if max_atoms < 1:raise ValueError('Terminal atom limit must be positive')
        row['scope_supported']=len(instance.coords)<=max_atoms
        if not row['scope_supported']:
            row.update(scope_exclusion='atom_limit',coordinates_A=instance.coords.tolist(),seconds=time.perf_counter()-started)
            return final,row
    try:
        row['total_dpa_calls']+=1
        try:row['initial']=property_snapshot(oracle,mol3d_to_validated_rdkit_mol(instance))
        except (ValueError,RuntimeError) as error:row['initial_property_error']=str(error)
        final,info=relax_conformer_standard_energy(instance,forcefield,'lbfgs',max_evaluations=148,fmax=.1,max_step=.2,history=history)
        info=dict(info);info.pop('history');row['optimization']=info
        row['physical_accepted']=bool(info['force_converged'] and info['geometry_valid'] and info['connectivity_preserved'])
        if info['geometry_valid'] and info['connectivity_preserved']:
            row['total_dpa_calls']+=1;row['final']=property_snapshot(oracle,mol3d_to_validated_rdkit_mol(final))
    except (ValueError,FloatingPointError,RuntimeError) as error:row['error']=str(error)
    row['total_dpa_calls']+=len(history)
    row['force_trace']=[{k:v for k,v in h.items() if k!='positions_A'} for h in history]
    row['coordinates_A']=final.coords.tolist();row['seconds']=time.perf_counter()-started
    return final,row
