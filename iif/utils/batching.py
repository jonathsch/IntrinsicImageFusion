import torch
import einops
from iif.utils.datastructure import Batch


def batched_average(fn, input, total_size, batch_size, **kwargs):
    def eval_sliced_input(input, i, j):
        if isinstance(input, torch.Tensor):
            input_batch = input[:, i:j]
            input_batch = einops.rearrange(input_batch.clone(), 'b spp ... -> (b spp) ...').contiguous()
            result = fn(input_batch, **kwargs)
        elif isinstance(input, Batch):
            # print(f"i: {i}, j: {j}, input: {input}")
            input_batch = input[:, i:j]
            # print(f"input batch: {input_batch}")
            input_batch = input_batch.map(lambda x: einops.rearrange(x.clone(), 'b spp ... -> (b spp) ...').contiguous())
            result = fn(**input_batch.to_dict(), **kwargs)
        else:
            raise NotImplementedError(f"Type {type(input)} not supported in batched_average")
        return result
        
    def average_result(result):
        if isinstance(result, dict):
            result = Batch(result)
            result = result.map(lambda x: einops.rearrange(x, '(b spp) ... -> b spp ...', spp=batch_size).float())
            result = result.mean(dim=1)
        elif isinstance(result, Batch):
            result = result.map(lambda x: einops.rearrange(x, '(b spp) ... -> b spp ...', spp=batch_size).float())
            result = result.mean(dim=1)
        elif isinstance(result, torch.Tensor):
            result = einops.rearrange(result, '(b spp) ... -> b spp ...', spp=batch_size).float()
            result = result.mean(1)
        elif isinstance(result, (list, tuple)):
            result = [average_result(r) for r in result]
        else:
            raise NotImplementedError(f"Type {type(result)} not supported in batched_average")
        return result
    
    def aggregate_result(results, result):
        if isinstance(result, torch.Tensor):
            results = results + result
        elif isinstance(result, (tuple, list)):
            results = [r1 + r2 for r1, r2 in zip(results, result)]
        elif isinstance(result, dict):
            result = Batch(result)
            results = Batch(results)
            results = results + result
        else:
            raise NotImplementedError(f"Type {type(result)} not supported in batched_average")
        return results
    
    def collate_results(results):
        if isinstance(results, torch.Tensor):
            results = results / (total_size // batch_size)
        elif isinstance(results, (tuple, list)):
            results = [r / (total_size // batch_size) for r in results]
        elif isinstance(results, dict):
            results = Batch(results)
            results = results / (total_size // batch_size)
        else:
            raise NotImplementedError(f"Type {type(results)} not supported in batched_average")
        return results
    
    # Trivial case
    if total_size == batch_size:
        result = eval_sliced_input(input, 0, total_size)
        result = average_result(result)
        return result

    results = None
    assert total_size % batch_size == 0, f"Input dimension  with shape {total_size} is not divisible by batch size {batch_size}"
    
    for i in range(0, total_size, batch_size):
        j = min(i + batch_size, total_size)

        result = eval_sliced_input(input, i, j)

        result = average_result(result)

        if results is None:
            results = result
        else:
            results = aggregate_result(results, result)

    results = collate_results(results)
    return results


def batched_eval(fn, input, total_size, batch_size, **kwargs):
    def eval_sliced_input(input, i, j):
        if isinstance(input, torch.Tensor):
            input_batch = input[:, i:j]
            input_batch = einops.rearrange(input_batch, 'b spp ... -> (b spp) ...')
            result = fn(input_batch, **kwargs)
        elif isinstance(input, Batch):
            # print(f"i: {i}, j: {j}, input: {input}")
            input_batch = input[:, i:j]
            # print(f"input batch: {input_batch}")
            input_batch = input_batch.map(lambda x: einops.rearrange(x, 'b spp ... -> (b spp) ...'))
            result = fn(**input_batch.to_dict(), **kwargs)
        else:
            raise NotImplementedError(f"Type {type(input)} not supported in batched_average")
        return result
        
    def reshape_result(result):
        if isinstance(result, dict):
            result = Batch(result)
            result = result.map(lambda x: einops.rearrange(x, '(b spp) ... -> b spp ...', spp=batch_size))
        elif isinstance(result, Batch):
            result = result.map(lambda x: einops.rearrange(x, '(b spp) ... -> b spp ...', spp=batch_size))
        elif isinstance(result, torch.Tensor):
            result = einops.rearrange(result, '(b spp) ... -> b spp ...', spp=batch_size)
        elif isinstance(result, (list, tuple)):
            result = [reshape_result(r) for r in result]
        else:
            raise NotImplementedError(f"Type {type(result)} not supported in batched_average")
        return result
    
    def aggregate_result(results, result):
        if isinstance(result, torch.Tensor):
            results = results.append(result)
        elif isinstance(result, (tuple, list)):
            results = [r1 + [r2] for r1, r2 in zip(results, result)]
        elif isinstance(result, dict):
            result = Batch(result)
            results.append(result)
        else:
            raise NotImplementedError(f"Type {type(result)} not supported in batched_average")
        return results
    
    def collate_results(results):
        if isinstance(results, torch.Tensor):
            results = torch.cat(results, dim=1)
        elif isinstance(results, (tuple, list)):
            results = [torch.cat(r, dim=1) for r in results]
        elif isinstance(results, dict):
            results = Batch(results)
            results = results.map(lambda x: torch.cat(x, dim=1))
        else:
            raise NotImplementedError(f"Type {type(results)} not supported in batched_average")
        return results
    
    # Trivial case
    if total_size == batch_size:
        result = eval_sliced_input(input, 0, total_size)
        result = reshape_result(result)
        return result

    results = None
    assert total_size % batch_size == 0, f"Input dimension  with shape {total_size} is not divisible by batch size {batch_size}"
    
    for i in range(0, total_size, batch_size):
        j = min(i + batch_size, total_size)

        result = eval_sliced_input(input, i, j)

        result = reshape_result(result)

        if results is None:
            if isinstance(result, dict):
                results = Batch(result)
            if isinstance(result, Batch):
                results = result.map(lambda x: [x])
            elif isinstance(result, torch.Tensor):
                results = [result]
            elif isinstance(result, (list, tuple)):
                results = [[r] for r in result]
            else:
                raise NotImplementedError(f"Type {type(result)} not supported in batched_average")
        else:
            results = aggregate_result(results, result)

    results = collate_results(results)
    return results