# Set Operations Experiment Configuration Generator

This tool allows you to generate a configuration file for running set operations experiments. The configuration file controls various aspects of the experiment, including the types of operations, set characteristics, and prompting strategies.

## How to Generate Your Configuration File

1. Create a new file named `config.yaml` in your project directory.

2. Copy the template below into your `config.yaml` file:

```yaml
# Prompt generation configuration
N_RUN: 100
RANDOM_SEED_VAL: 292
OP_LIST:
  - "union"
  - "intersection"
  - "difference"
  - "symmetric difference"
  - "cartesian product"

# Sampler/Sets configuration
SET_TYPES:
  - "numbers"
  - "words"
N:
  - 10000
M:
  - 2
  - 4
  - 8
  - 16
ITEM_LEN:
  - None
  - 1
  - 2
  - 3
  - 4
OVERLAP_FRACTION:
  - None
DECILE_NUM:
  - None

# Prompt configuration
K_SHOT:
  - 0
  - 1
  - 3
  - 5
PROMPT_TYPE:
  - "formal_language"
  - "plain_language"
  - "functional_language"
  - "pythonic_language"
  - "iterative_accumulation"
PROMPT_APPROACH:
  - "baseline"
  - "baseline_allow_empty"
  - "composite"
  - "composite_allow_empty"
  - "domain_expertise"
  - "self_recitation"
  - "chain_of_thought"
IS_FIX_SHOT:
  - True
```

## Configuration Options Explained

### Prompt Generation Configuration

* `N_RUN`: Number of prompt-examples to generate
* `RANDOM_SEED_VAL`: Seed for reproducibility
* `OP_LIST`: Set operations to perform (e.g., union, intersection)

### Sampler/Sets Configuration

* `SET_TYPES`: Types of sets to use ("numbers" or "words")
* `N`: Maximum range for sampling numbers/words
* `M`: Number of members in each set
* `ITEM_LEN`: Length of items in sets (None for no restriction, 1 for single-digit/character)
* `OVERLAP_FRACTION`: Fraction of overlap between sets (if applicable)
* `DECILE_NUM`: Decile number for set generation (if applicable)

### Prompt Configuration

* `K_SHOT`: Number of examples in the prompt
* `PROMPT_TYPE`: Type of prompt language ("formal_language" or "plain_language")
* `PROMPT_APPROACH`: Prompting strategy (e.g., "baseline", "chain_of_thought")
* `IS_FIX_SHOT`: Whether to use fixed pre-defined sets across all experiments

### Customization Tips

1. To disable options, add a # at the beginning of the line.
2. You can add multiple values to lists to test different configurations.
3. Ensure that your modifications maintain the YAML format.
