import logging
import torch
import torch.nn as nn

# Create a logger for this module
logger = logging.getLogger(__name__)

def get_vocabulary_indexes(tokenizer, targets):
    """
    Get the indexes of vocabulary tokens that match the target words.
    This function tokenizes the target words using the provided tokenizer and 
    finds all vocabulary entries that contain the target words. It returns a 
    list of unique token indexes that either match the target words directly 
    or contain them as substrings.
    Args:
        tokenizer: A tokenizer object that provides methods `encode`, 
                   `convert_ids_to_tokens`, and supports iteration over its 
                   vocabulary.
        targets (list of str): A list of target words to search for in the 
                               tokenizer's vocabulary.
    Returns:
        list of int: A list of unique token indexes that match or contain the 
                     target words.
    """
    # Check how each string is tokenized
    tokenized_indexes = []
    for test_str in targets:
        tokens = tokenizer.encode(test_str)
        token_strings = tokenizer.convert_ids_to_tokens(tokens)
        logger.debug(f"'{test_str}' → {list(zip(tokens, token_strings))}")
        tokenized_indexes.extend(tokens)

    # Find all vocabulary entries that contain target words
    related_tokens = []
    for i in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(i)
        for target in targets:
            if target.lower() in token.lower():
                related_tokens.append((i, token))
                logger.debug(f"Related token found - ID: {i}, Token: '{token}'")
                break  # No need to check other targets if one matches

    token_indexes = list(set(tokenized_indexes + [token_id for token_id, _ in related_tokens]))
    
    logger.debug("\nAll vocabulary tokens containing target words:")
    for token_id in token_indexes:
        token = tokenizer.convert_ids_to_tokens(token_id)
        logger.debug(f"ID: {token_id}, Token: '{token}'")

    return token_indexes

def get_layers(model, num_layers):
    """Return names of the last `num_layers` transformer blocks (e.g. `model.layers.N`)."""
    candidates = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            if i + 2 == len(parts) and parts[i + 1].isdigit():
                if any(isinstance(sm, nn.Linear) for sm in module.modules()):
                    candidates.append((int(parts[i + 1]), name))

    candidates.sort(key=lambda x: x[0])
    return [name for _, name in candidates[-num_layers:]]

def apply_weight_masks(model):
    """Apply pruning permanently by removing pruning reparameterizations."""
    import torch.nn.utils.prune as prune

    for name, module in model.named_modules():
        # torch.nn.utils.prune adds weight_orig/weight_mask when a parameter is pruned.
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
            logger.info(f"Made pruning permanent for {name}.weight")