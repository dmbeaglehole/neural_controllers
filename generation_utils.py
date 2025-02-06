def generate_on_text(model, tokenizer, input_text, **kwargs):
        
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    # Generate output
    outputs = model.generate(
        **inputs,
        **kwargs,
    )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0])
    return generated_text
    
def hook_model(model, directions, layers_to_control, control_coef, component_idx=0):
    hooks = {}
    for layer_idx in layers_to_control:
        control_vec = directions[layer_idx][component_idx]
        if len(control_vec.shape)==1:
            control_vec = control_vec.reshape(1,1,-1)
               
               
        block = model.model.layers[layer_idx]

        def block_hook(module, input, output, control_vec=control_vec, control_coef=control_coef):
            """
            note that module, input are unused, but are
            required by torch.
            """ 
            
            new_output = output[0]

            new_output = new_output + control_coef*control_vec.to(dtype=new_output.dtype, device=new_output.device)
            
            if isinstance(output, tuple):
                new_output = (new_output,) + output[1:] 
            
            return new_output
        
        hook_handle = block.register_forward_hook(block_hook)
        hooks[layer_idx] = hook_handle
    
    return hooks

def clear_hooks(hooks) -> None:
    for hook_handle in hooks.values():
        hook_handle.remove()