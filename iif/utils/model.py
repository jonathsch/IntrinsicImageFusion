import copy
import hydra
from omegaconf import DictConfig, OmegaConf
import torch


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def freeze_model_params(model, freezing_info=True):
    if isinstance(freezing_info, bool):
        if freezing_info:
            # Freeze all params
            for param in model.parameters():
                param.requires_grad = False
        
    elif isinstance(freezing_info, (dict, DictConfig)):
        # Freeze specific params
        for name, param in model.named_parameters():
            should_freeze = False
            for key, value in freezing_info.items():
                if name.startswith(key):
                    should_freeze = value

            if should_freeze:
                param.requires_grad = False
    else:
        raise ValueError(f"freezing_info must be either a boolean or a dictionary, but got {type(freezing_info)}")
    
    return model

def load_model(**model_info):
    # Get the model config
    model_cfg = model_info["cfg"]

    # Instantiate the model
    if isinstance(model_cfg, (dict, DictConfig)):
        model = hydra.utils.instantiate(model_cfg)
    else:
        # Has been already instantiated
        model = model_cfg

    # Load the parameters
    if model_info["pt"] is not None:
        model.load_state_dict(torch.load(model_info["pt"], weights_only=True))

    # Freeze the model if requested
    if "freeze" in model_info:
        freeze_model_params(model, model_info["freeze"])

    return model

def get_config(model_info):
    if model_info is None:
        return None
    
    model_info = copy.deepcopy(model_info)

    # Get the cfg of the children as well
    for k in model_info:
        v = model_info[k]
        if k == "_target_":
            if v == "iif.utils.model.load_model":
                subconf = get_config(model_info["cfg"])
                assert isinstance(subconf, (dict, DictConfig)), "cfg must be a dict or DictConfig"
                return subconf
        elif isinstance(v, (dict, DictConfig)):
            subconf = get_config(v)
            model_info[k] = subconf

    return model_info
