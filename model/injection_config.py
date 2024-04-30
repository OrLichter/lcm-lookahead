# Handles the selection of which layers are used in extended self-attention + which layers get LoRA training.
# Currently uses all self-attention layers for shared attention.

def create_injection_module_list(type, block_indices, num_attentions, num_layers):
    module_list = []

    for entry_idx, block_idx in enumerate(block_indices):

        if len(block_indices) > 1:
            type_qualifier = f"{type}_blocks.{block_idx}"
        else:
            type_qualifier = f"{type}_block"

        for attn_idx in range(num_attentions):
            attn_qualifier = f"attentions.{attn_idx}"

            for layer_idx in range(num_layers[entry_idx]):
                layer_qualifier = f"transformer_blocks.{layer_idx}"

                module_list.append(f"{type_qualifier}.{attn_qualifier}.{layer_qualifier}")

    return module_list


def get_injection_modules():
    injection_modules_up = create_injection_module_list("up", [0, 1], 3, [10, 2])
    injection_modules_down = create_injection_module_list("down", [1, 2], 2, [2, 10])
    injection_modules_mid = create_injection_module_list("mid", [0], 1, [10])

    return injection_modules_up + injection_modules_down + injection_modules_mid


def get_lora_injection_modules():
    """ Returns the modules in which we will inject LoRA layers and tune the encoder weights """

    injection_modules_up = create_injection_module_list("up", [0, 1], 3, [10, 2])

    # drop layers after the last self attention. These will not affect any extracted KVs and not get gradients anyway.
    filtered_injection_up = [module for module in injection_modules_up if "up_blocks.1.attentions.2.transformer_blocks.1" not in module]
    
    injection_modules_down = create_injection_module_list("down", [1, 2], 2, [2, 10])
    injection_modules_mid = create_injection_module_list("mid", [0], 1, [10])

    return filtered_injection_up + injection_modules_down + injection_modules_mid

def get_kvcopy_lora_injection_modules():
    """ Returns the subset of the kvcopy modules where LoRA will be used, restricts training only to the LoRA params """

    # NOTE: Right now this is the same as get_injection_modules, but is provided in a separate function in case you want to change it.
    lora_injection_modules = get_injection_modules()

    # Make sure we only inject into the kvcopy modules
    kv_modules = get_injection_modules()
    if not all([module in kv_modules for module in lora_injection_modules]):
        raise ValueError("Not all injection modules are in the kvcopy modules")
    return lora_injection_modules
