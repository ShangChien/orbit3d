"""Extend named property interfaces without erasing learned common-target features."""
import torch


def expand_interval_conditioner(source, destination, old_names, new_names):
    if len(set(new_names)) != len(new_names) or not set(old_names) <= set(new_names):
        raise ValueError('Target expansion must retain each existing target exactly once')
    if source.num_props != len(old_names) or destination.num_props != len(new_names):
        raise ValueError('Target names and conditioner dimensions differ')
    old, new = source.state_dict(), destination.state_dict()
    mapping = {i: new_names.index(name) for i, name in enumerate(old_names)}
    with torch.no_grad():
        for key, value in old.items():
            if key.startswith('prop_encoders.'):
                _, index, suffix = key.split('.', 2)
                new[f'prop_encoders.{mapping[int(index)]}.{suffix}'].copy_(value)
            elif key == 'prop_id_emb.weight':
                for before, after in mapping.items():
                    new[key][after].copy_(value[before])
            elif key == 'global_mlp.0.weight':
                new[key].zero_()
                width = source.emb_dim
                for before, after in mapping.items():
                    new[key][:, after*width:(after+1)*width].copy_(value[:, before*width:(before+1)*width])
            else:
                new[key].copy_(value)


def expand_property_readout(source, destination, old_names, new_names):
    """Existing targets retain their normalization; new rows initialize to their mean."""
    destination.net[:-1].load_state_dict(source.net[:-1].state_dict(), strict=True)
    with torch.no_grad():
        destination.net[-1].weight.zero_()
        destination.net[-1].bias.zero_()
        for before, name in enumerate(old_names):
            after = new_names.index(name)
            destination.net[-1].weight[after].copy_(source.net[-1].weight[before])
            destination.net[-1].bias[after].copy_(source.net[-1].bias[before])
