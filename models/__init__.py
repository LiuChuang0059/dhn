def get_model(model_config):
  
  if model_config.name == 'simple_transformer':
    from .simple_transformer import SimpleTransformer
    model = SimpleTransformer(model_config)
  
  else:
    raise NotImplementedError('Model not implemented.')
  
  return model