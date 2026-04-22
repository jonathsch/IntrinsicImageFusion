from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf

def range2list(config, max_length=None):
    if config is None:
        return list(range(max_length))
    elif isinstance(config, int):
        return [config]
    elif isinstance(config, list):
        return config
    elif isinstance(config, ListConfig):
        return config
    elif isinstance(config, str):
        slice_def = config.split(":")

        start = 0
        if len(slice_def) >= 1:
            if slice_def[0] != "":
                start = int(slice_def[0])

        end = max_length
        if len(slice_def) >= 2:
            if slice_def[1] != "":
                end = int(slice_def[1])
        assert end is not None, f"End must be defined for config {config}"

        step = 1
        if len(slice_def) == 3:
            if slice_def[2] != "":
                step = int(slice_def[2])

        return list(range(start, end, step))
    
# Dynamic config parsing utility
def parse_dynamic_cfg(cfg):
    cfg_items = list(cfg.items())
    for k, v in cfg_items:
        if isinstance(v, (dict, DictConfig)):
            if k == "_override_":
                cfg["cfg"] = OmegaConf.merge(cfg["cfg"], parse_dynamic_cfg(v))
            else:
                cfg[k] = parse_dynamic_cfg(v)
        elif isinstance(v, str):
            if k == "_cfg_":
                cfg["cfg"] = OmegaConf.load(v)
                del cfg["_cfg_"]
    return cfg
            

# WORKAROUND: Not possible to stop hydra from instantiating, just if _target_ is present
def native_dict(**kwargs):
    return dict(**kwargs)

