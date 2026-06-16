# Data directory

This repository includes only small reproducibility metadata and sample data.
Large generated artefacts and full training/example files are intentionally
excluded from Git.

Included:

- `chair-concept-def.json`: target terms and test prompts used by the study
  model configuration.
- `chair-simplified-sample.json`: a minimal schema example for the full
  `chair-simplified-10k.json` examples file.

Not included:

- `chair-simplified-10k.json`, the full concept/background examples file.
- `chair-3B-simplified-activations-10k.npz`, the activation cache generated
  from the base model and examples.
- model checkpoints or GGUF files.

The activation cache is a derived intermediate artefact. It should be
regenerated from the base model, final config, example data, and code rather
than committed to the repository.
