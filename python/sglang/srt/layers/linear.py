from vllm.model_executor.layers.linear import *

for module in [MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear]:
    original_weight_loader = module.weight_loader

    def create_new_weight_loader(original_weight_loader):
        def new_weight_loader(
            self,
            param: Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: Optional[int] = None,
        ):
            print("loaded weight", loaded_weight)
            if param.data.dtype != loaded_weight.dtype:
                param.data = torch.empty_like(
                    param.data, dtype=loaded_weight.dtype, device="cuda"
                )
            original_weight_loader(param, loaded_weight, loaded_shard_id)

        return new_weight_loader

    new_weight_loader = create_new_weight_loader(original_weight_loader)
    setattr(module, "weight_loader", new_weight_loader)
