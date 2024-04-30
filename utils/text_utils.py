import torch

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def add_tokens(tokenizers, tokens, text_encoders):
    new_token_indices = {}
    for idx, tokenizer in enumerate(tokenizers):
        for token in tokens:
            num_added_tokens = tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {token}. Please pass a different"
                    " `placeholder_token` that is not already in the tokenizer."
                )
            
            new_token_indices[f"{idx}_{token}"] = num_added_tokens
        # resize embedding layers to avoid crash. We will never actually use these.    
        text_encoders[idx].resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

    return new_token_indices
        
            
def patch_embedding_forward(embedding_layer, new_tokens, new_embeddings):
    
    def new_forward(input):
        embedded_text = torch.nn.functional.embedding(
            input, embedding_layer.weight, embedding_layer.padding_idx, embedding_layer.max_norm,
            embedding_layer.norm_type, embedding_layer.scale_grad_by_freq, embedding_layer.sparse)
        
        replace_indices = (input == new_tokens)

        if torch.count_nonzero(replace_indices) > 0:
            embedded_text[replace_indices] = new_embeddings

        return embedded_text
    
    embedding_layer.forward = new_forward