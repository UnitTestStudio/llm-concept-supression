import traceback
from .neuron_saliency_analysis import ConceptNeuronSaliencyAnalyzer
from .pruning_operations import get_vocabulary_indexes, get_layers
from .utils import load_examples
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import logging
import os

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

    def prune_concept(self):
        layer_names = get_layers(self.model, self.config["neural_pruning"]["num_layers"])
        logger.debug(f"Selected layers: {layer_names}")

        # If no activation file path is supplied, assume we want to test a random pruning
        if self.config["neural_pruning"]["activations_file_path"] is None:
            prune_percentage = self.config["neural_pruning"].get("prune_percentage", 0.5)
            for layer_name in layer_names:
                submodule = dict(self.model.named_modules())[layer_name]
                for submodule_name, submodule in submodule.named_modules():
                    if isinstance(submodule, nn.Linear):
                        mask = torch.ones_like(submodule.weight)
                        num_elements_to_prune = int(prune_percentage * mask.numel())
                        indices = torch.randperm(mask.numel())[:num_elements_to_prune]
                        mask.view(-1)[indices] = 0.0
                        prune.custom_from_mask(submodule, name="weight", mask=mask)
                        logger.debug(f"Applied mask with {prune_percentage*100}% zeros to {layer_name}.{submodule_name}")
        else:
            # Initialize the ConceptNeuronSaliencyAnalyzer
            analyzer = ConceptNeuronSaliencyAnalyzer(self.model, 
                                                    self.tokenizer, 
                                                    self.config["base_model"]["device"])

            # Check if activations file exists, if not, extract activations
            if not os.path.isfile(self.config['neural_pruning']['activations_file_path']):
                try:
                    logger.info("Extracting activations...")
                    analyzer.extract_activations(self.concept_examples,
                                                self.background_examples,
                                                self.config['neural_pruning']['activations_file_path'])
                    logger.info(f"Activations saved to {self.config['neural_pruning']['activations_file_path']}")
                except Exception as e:
                    logger.error(f"An error occurred: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.info(f"Activations file already exists at {self.config['neural_pruning']['activations_file_path']}")

                        
            # Analyze concept saliency for the top 10 layers
            results = analyzer.analyze_concept_saliency(
                activations_path = self.config['neural_pruning']['activations_file_path'],
                num_layers = 32,
                top_k = 8000,
                regularisation_strength = 100,
                statistical_test = True
            )

            # Create a mask with the same shape as the model's weights
            mask = {}
            for submodule_name, neurons in results.items():
                submodule = dict(self.model.named_modules())[submodule_name]
                mask[submodule_name] = torch.ones_like(submodule.weight)
                for neuron_idx, _ in neurons:
                    mask[submodule_name][neuron_idx, :] = 0.0
                    logger.debug(f"Modified neuron index {neuron_idx} in {submodule_name}")

            # Apply pruning using PyTorch's pruning method with the custom mask
            for submodule_name, submodule_mask in mask.items():
                submodule = dict(self.model.named_modules())[submodule_name]
                prune.custom_from_mask(submodule, name="weight", mask=submodule_mask)