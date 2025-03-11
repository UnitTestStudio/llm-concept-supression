from .pruning_operations import get_vocabulary_indexes
from .utils import load_examples
import torch
import torch.nn.utils.prune as prune
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

class Pruner:
    def __init__(self, model, tokenizer, concept_definition, config):
        self.model = model
        self.tokenizer = tokenizer
        self.concept_definition = concept_definition
        self.config = config
        if config["neural_pruning"]["prune_concept"]:
            self.concept_examples, self.background_examples = load_examples(config)

    def prune_vocabulary(self):
        """
        Prune the model's vocabulary by zeroing out the weights of specified token IDs in the model's head layer.
        """
        logger.info("Pruning the model's vocabulary...")
        try:
            token_indexes = get_vocabulary_indexes(self.tokenizer, 
                                                   self.concept_definition)
        except Exception as e:
            logger.error(f"Error generating token indexes: {e}")
            raise

        # Create a mask with the same shape as lm_head's weight
        mask = torch.ones_like(self.model.lm_head.weight)

        # Set the mask to zero for the specified token rows
        for token_id in token_indexes:
            mask[token_id, :] = 0.0
            logger.debug(f"Modified token ID {token_id}")

        # Apply pruning using PyTorch's pruning method with the custom mask
        prune.custom_from_mask(self.model.lm_head, name="weight", mask=mask)

    # def prune_concept(self):