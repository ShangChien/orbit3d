"""Necessary electron-count constraints applied before categorical sampling."""
import torch
from rdkit import Chem


def atom_electron_parities(model, device):
    parity = torch.zeros(model.num_atom, dtype=torch.long, device=device)
    table = Chem.GetPeriodicTable()
    for symbol in model.args.allow_atoms.split(','):
        parity[model.dictionary[symbol]] = table.GetAtomicNumber(symbol) % 2
    return parity  # The null atom contributes no electrons.


def mask_terminal_element_logits(logits, accumulated, terminal, desired, element_parity):
    if desired not in (0, 1):
        raise ValueError('Electron parity must be 0 or 1')
    supported = (~terminal[:, None]) | ((accumulated[:, None] + element_parity[None]) % 2 == desired)
    return logits.masked_fill(~supported, -torch.inf)
