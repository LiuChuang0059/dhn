def get_model_hamiltoinian(model_config, dtype):
  
  if model_config.hamiltonian == 'default':
    from .hamiltonian import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)

  elif model_config.hamiltonian == 'ar':
    from .hamiltonian_ar import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)

  else:
    raise NotImplementedError('Hamiltonian model not implemented.')
  
  return model