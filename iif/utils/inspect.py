import torch

from iif.utils.attribute import rgetattr


def inspect_layer(model,
                  layer_path,
                  input_storage_dict: dict = None,
                  output_storage_dict: dict = None,
                  name: str = None,
                  unsqueeze: bool = True,
                  index_to_inspect: int = None,
                  skip_if_no_grad: bool = False):
    if name is None:
        name = layer_path

    def hook(layer, input, output):
        if skip_if_no_grad and not torch.is_grad_enabled():
            return

        if index_to_inspect is not None:
            output = output[index_to_inspect]
            
        if unsqueeze:
            output = output.unsqueeze(0)

        def add_new_values_to_storage(name, storage, values):
            if name in storage:
                # storage[name] = torch.cat((storage[name], values), dim=0)
                storage[name].append(values)
            else:
                storage[name] = [values]

        if output_storage_dict is not None:
            add_new_values_to_storage(f"{name}", output_storage_dict, output)

        if input_storage_dict is not None:
            add_new_values_to_storage(f"{name}", input_storage_dict, input)

    return rgetattr(model, layer_path).register_forward_hook(hook)
