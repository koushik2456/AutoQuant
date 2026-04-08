import torch
import gc

gc.collect()                 # clear Python memory
torch.cuda.empty_cache()     # clear GPU cache