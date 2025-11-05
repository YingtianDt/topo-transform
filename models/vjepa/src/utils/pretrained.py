import logging

import torch

def load_checkpoint(r_path, module):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        msg = module.load_state_dict(checkpoint, strict=True)
        print(f'Loaded pretrained component with msg: {msg}')

        del checkpoint
    except Exception as e:
        print(f'Encountered exception when loading checkpoint {e}')
        raise Exception()
    
    return module