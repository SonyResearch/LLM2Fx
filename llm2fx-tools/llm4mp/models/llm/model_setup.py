from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_tokenizer(model_path, cache_dir, use_number_token_loss=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        pad_token = tokenizer.eos_token
        tokenizer.pad_token = pad_token
        print("tokenizer.pad_token: ", tokenizer.pad_token)
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        print(f"Added pad token: {tokenizer.pad_token}")
    return tokenizer


def setup_llm(model_path, attn_implementation, cache_dir, dtype):
    tokenizer = setup_tokenizer(model_path, cache_dir)
    lm = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return lm, tokenizer
