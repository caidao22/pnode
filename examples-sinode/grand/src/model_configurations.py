from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_laplacian_diffusion import LaplacianODEFunc
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock
from block_mixed import MixedODEblock
from block_transformer_hard_attention import HardAttODEblock
from block_transformer_rewiring import RewireAttODEblock

from block_pnode import PNODEblock
from function_mytransformer_attention import myODEFuncTransformerAtt

class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass


def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'mixed':
    block = MixedODEblock
  elif ode_str == 'attention':
    block = AttODEblock
  elif ode_str == 'hard_attention':
    block = HardAttODEblock
  elif ode_str == 'rewire_attention':
    block = RewireAttODEblock
  elif ode_str == 'constant':
    block = ConstantODEblock
  elif ode_str == 'heavyball':
    from block_heavyball import HBNODEblock
    block = HBNODEblock
  elif ode_str == 'pnode':
    block = PNODEblock
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'GAT':
    f = ODEFuncAtt
  elif ode_str == 'transformer':
    f = ODEFuncTransformerAtt
  elif ode_str == 'hbnode':
    from function_hbnode import HeavyBallNODEFunc
    f = HeavyBallNODEFunc
  elif ode_str == 'mytransformer':
    f = myODEFuncTransformerAtt
  else:
    raise FunctionNotDefined
  return f
