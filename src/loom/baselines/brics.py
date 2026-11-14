"""A training-only, single-cut BRICS fragment recombination baseline."""
from collections import defaultdict
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS


def build_library(molecules):
    sources=defaultdict(set);cut_counts={}
    for molecule in molecules:
        if molecule is None or not molecule.HasProp('SourceID'):
            raise ValueError('BRICS training structures require readable molecules with SourceID')
        identifier=molecule.GetProp('SourceID')
        graph=Chem.RemoveHs(molecule);cuts=list(BRICS.FindBRICSBonds(graph));cut_counts[identifier]=len(cuts)
        for cut in cuts:
            fragments=Chem.GetMolFrags(BRICS.BreakBRICSBonds(graph,bonds=[cut]),asMols=True)
            if len(fragments)!=2:
                raise ValueError('A single BRICS cut did not produce two fragments')
            for fragment in fragments:
                if sum(a.GetAtomicNum()==0 for a in fragment.GetAtoms())!=1:
                    raise ValueError('A single-cut fragment must have one attachment point')
                sources[Chem.MolToSmiles(fragment,isomericSmiles=True)].add(identifier)
    fragments=[{'smiles':smiles,'source_ids':sorted(ids)} for smiles,ids in sorted(sources.items())]
    if not fragments:raise ValueError('No single-cut BRICS fragments in the supplied training set')
    return {'fragments':fragments,'source_cut_counts':cut_counts}


class FragmentSampler:
    def __init__(self, library):
        self.records=library['fragments']
        self.molecules=[Chem.MolFromSmiles(r['smiles']) for r in self.records]
        self.groups=defaultdict(list)
        for index,molecule in enumerate(self.molecules):
            atoms=[a for a in molecule.GetAtoms() if a.GetAtomicNum()==0]
            if len(atoms)!=1:raise ValueError('BRICS library requires single-port fragments')
            self.groups[atoms[0].GetIsotope()].append(index)
        pairs={tuple(sorted((int(re.sub(r'\D','',left)),int(re.sub(r'\D','',right)))))
               for rules in BRICS.reactionDefs for left,right,_ in rules}
        self.pairs=sorted(pair for pair in pairs if all(label in self.groups for label in pair))
        if not self.pairs:raise ValueError('No compatible BRICS pairs in the supplied library')
        weights=np.array([len(self.groups[a])*len(self.groups[b]) for a,b in self.pairs],dtype=float)
        self.probability=weights/weights.sum()

    def sample(self, seed):
        rng=np.random.default_rng(seed)
        labels=self.pairs[int(rng.choice(len(self.pairs),p=self.probability))]
        indices=[int(rng.choice(self.groups[label])) for label in labels]
        left,right=[self.molecules[i] for i in indices]
        # Exactly one selected seed and one partner; there is no recursive search.
        products=list(BRICS.BRICSBuild([right],seeds=[left],maxDepth=0,
                                      onlyCompleteMols=True,uniquify=True,scrambleReagents=False))
        if not products:raise ValueError('The selected compatible BRICS pair produced no complete graph')
        by_smiles={}
        for mol in products:
            Chem.SanitizeMol(mol);by_smiles[Chem.MolToSmiles(mol,isomericSmiles=True)]=mol
        choices=sorted(by_smiles);smiles=choices[int(rng.integers(len(choices)))]
        return smiles,{'fragment_indices':indices,'attachment_labels':list(labels),
                       'fragment_smiles':[self.records[i]['smiles'] for i in indices],
                       'enumerated_complete_products':len(choices)}
