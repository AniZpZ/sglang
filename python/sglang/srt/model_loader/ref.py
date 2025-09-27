    @staticmethod
    def rebinding_and_load_weights(model, first_time_load_weights, weights):
        """Reload weights with proper state management for multiple loading scenarios."""
        # Reset the model state to allow re-quantization
        DefaultModelLoader.reset_model_weights_state(model)

        # Preserve workspace if exists
        for _, module in model.named_modules():
            if torch.is_tensor(getattr(module, "workspace", None)):
                setattr(module, f"preserved_workspace", getattr(module, "workspace"))

        existing_params = dict(model.named_parameters())

        # Preserve original data, so that after parameter loading, we can put the parameter
        # in the original tensor
        original_param_dict = {}
        for name, p in existing_params.items():
            original_param_dict[name] = p.data

        # Recover the parameter to the state before first loading
        for name, rebuild_info in model.original_weights_rebuild_keys.items():
            if name in existing_params:
                existing_params[name].data = torch.empty(
                    rebuild_info["shape"],
                    dtype=rebuild_info["dtype"],
                    device=existing_params[name].device,
                )

        # Restore weight loaders
        for k, loader_k in model.recorded_loader.items():
            for n, loader in loader_k.items():
                if not hasattr(existing_params[n], k):
                    # Simple binding for now - in a full implementation you might need
                    # a more sophisticated binding mechanism
                    setattr(existing_params[n], k, loader)

        # After recovering, the weight loading can be called as usual
        updated_params = first_time_load_weights(weights)

        # Manually conducting process weights after loading
        # Note: We don't need to call load_weights_and_postprocess here because
        # first_time_load_weights already loaded the weights, we just need to process them
        target_device = next(model.parameters()).device
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        # Mark as already called
        model.process_weights_after_loading_already_called = True

        # Put the value of the newly created tensor to the original tensor
        for name, p in model.named_parameters():
            assert (
                name in original_param_dict
            ), f"param {name} is not in original_param_dict"
            assert (
                original_param_dict[name].dtype == p.data.dtype
            ), f"param {name} dtype mismatch: {original_param_dict[name].dtype} vs {p.data.dtype}"
            assert (
                original_param_dict[name].numel() == p.data.numel()
            ), f"param {name} numel() mismatch: {original_param_dict[name].numel()} vs {p.data.numel()}"

            if name in updated_params:
                strided_data = torch.as_strided(
                    p.data,
                    original_param_dict[name].shape,
                    original_param_dict[name].stride(),
                )
                original_param_dict[name].copy_(strided_data)

            del p.data
            p.data = original_param_dict[name]

        del original_param_dict
        del existing_params

        gc.collect()

        # Restore workspace
        for _, module in model.named_modules():
            if torch.is_tensor(getattr(module, "workspace", None)):
                setattr(module, "workspace", getattr(module, "preserved_workspace"))
                delattr(module, "preserved_workspace")

        return updated_params