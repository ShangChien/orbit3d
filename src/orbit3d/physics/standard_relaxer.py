"""ASE energy minimizers for free molecules, with raw forces and measured calls."""
from dataclasses import replace
import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes
from ase.optimize import FIRE,LBFGS
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from ..generator.molecular_state import mol3d_to_validated_rdkit_mol,mol_to_clean_smiles


def check_relaxed_geometry(initial_graph,instance):
    """Keep numerical convergence separate from graph/stereo admission."""
    try:
        graph=mol3d_to_validated_rdkit_mol(instance)
        inferred=Chem.Mol(graph)
        rdDetermineBonds.DetermineConnectivity(inferred)
        edges=lambda m:{tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in m.GetBonds()}
        same=edges(initial_graph)==edges(inferred)
        return {'geometry_valid':True,'connectivity_preserved':same,
                'geometry_error':'' if same else 'Coordinate-inferred connectivity changed'}
    except ValueError as error:
        return {'geometry_valid':False,'connectivity_preserved':None,'geometry_error':str(error)}


class _ForcefieldCalculator(Calculator):
    implemented_properties=['energy','forces']

    def __init__(self,instance,forcefield,max_evaluations,history):
        super().__init__()
        self.instance=instance;self.forcefield=forcefield
        self.max_evaluations=max_evaluations;self.history=history
        self.coordinate_dtype=torch.float64 if instance.coords.dtype==np.float64 else torch.float32
        self.atom_types=torch.tensor([forcefield.atom_type_map.index(s) for s in instance.symbols],
                                     device=forcefield.device_str,dtype=torch.long)
        self.last_positions=None

    def calculate(self,atoms=None,properties=('energy','forces'),system_changes=all_changes):
        if len(self.history)>=self.max_evaluations:
            raise RuntimeError('Optimizer exceeded its force-evaluation budget')
        super().calculate(atoms,properties,system_changes)
        x=torch.tensor(self.atoms.positions,dtype=self.coordinate_dtype,device=self.forcefield.device_str)
        entry={'call':len(self.history)+1,'positions_A':x.cpu().numpy().tolist(),'completed':False}
        self.history.append(entry)
        energy,forces=self.forcefield.compute_energy_and_forces(x,self.atom_types,
            clip_forces=False,zero_com_forces=False,charges=[self.instance.charge],multiplicities=[self.instance.multiplicity])
        energy=float(torch.as_tensor(energy).sum().detach().cpu())
        forces=torch.as_tensor(forces).detach().cpu().numpy()
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise FloatingPointError('Nonfinite energy-minimization evaluation')
        norms=np.linalg.norm(forces,axis=-1)
        entry.update(completed=True,energy_eV=energy,rms_force_eVA=float(np.sqrt(np.mean(norms**2))),
                     max_force_eVA=float(norms.max()))
        self.last_positions=x.detach().cpu().numpy().copy()
        self.results={'energy':energy,'forces':np.asarray(forces,dtype=float)}


def relax_conformer_standard_energy(instance,forcefield,method='lbfgs',max_evaluations=150,
                                    fmax=.1,max_step=.2,history=None):
    """Run standard unconstrained geometry steps; report final chemical admission separately."""
    if method not in {'lbfgs','fire'} or max_evaluations<1 or fmax<=0 or max_step<=0:
        raise ValueError('Invalid energy-minimization settings')
    initial=mol3d_to_validated_rdkit_mol(instance)
    instance=replace(instance,graph=initial,smiles=mol_to_clean_smiles(initial))
    history=[] if history is None else history
    if history:raise ValueError('Supply an empty evaluation history for a fresh minimization')
    atoms=Atoms(instance.symbols,positions=instance.coords,pbc=False)
    calculator=_ForcefieldCalculator(instance,forcefield,max_evaluations,history);atoms.calc=calculator
    if method=='lbfgs':
        optimizer=LBFGS(atoms,logfile=None,maxstep=max_step,memory=100,alpha=70.,damping=1.,use_line_search=False)
    else:
        optimizer=FIRE(atoms,logfile=None,maxstep=max_step,dt=.1,dtmax=1.,Nmin=5,
                       finc=1.1,fdec=.5,astart=.1,fa=.99,a=.1,downhill_check=False)
    # Initial force plus one new evaluation per step for these non-line-search variants.
    converged=bool(optimizer.run(fmax=fmax,steps=max_evaluations-1))
    result=replace(instance,coords=calculator.last_positions)
    admission=check_relaxed_geometry(initial,result)
    info={'method':method,'force_converged':converged,
          'status':'force_converged' if converged else 'evaluation_budget',
          'evaluation_calls':len(history),'optimizer_steps':optimizer.nsteps,
          'maximum_evaluations':max_evaluations,'maximum_atom_force_threshold_eVA':fmax,
          'maximum_atom_step_A':max_step,'coordinate_dtype':str(calculator.coordinate_dtype),
          'initial_energy_ev':history[0]['energy_eV'],'final_energy_ev':history[-1]['energy_eV'],
          'init_force_eva':history[0]['rms_force_eVA'],'final_force_eva':history[-1]['rms_force_eVA'],
          'init_max_force_eva':history[0]['max_force_eVA'],'final_max_force_eva':history[-1]['max_force_eVA'],
          **admission,'history':history}
    if not admission['geometry_valid'] or not admission['connectivity_preserved']:info['status']='graph_not_preserved'
    return result,info
