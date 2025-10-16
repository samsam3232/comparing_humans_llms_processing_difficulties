# Comparing humans and LLMs

This repository contains experimental code and data for our paper [Comparing Human and Language Models Sentence Processing Difficulties on Complex Structures](https://arxiv.org/pdf/2510.07141).

## Overview

We test reading comprehension on sentences with challenging syntax (e.g., garden-path sentences, center embeddings, reduced relatives) across:

- **Multiple LLMs**: DeepSeek, Qwen, Gemma, Llama, OpenAI models
- **Syntactic constructions**: NPS, NPVP, reduced relatives, subject/object relatives, double center embedding, depth charge, similarity interference,

## Repository Structure

```
├── configs/              # Model-specific experiment configurations
│   ├── family_name       # Some family of models we want to test
├── data/                 # CSV files with experimental stimuli
├── human_experiments/    # Prolific study creation and results parsing for human experiments
├── inference/            # LLM inference code
│   └── textgen_inference/
├── prefixes/             # Few-shot example templates (JSON)
├── results/              # Experimental results (CSV)
├── constants.py          # Prompt templates and examples
└── global_utils.py       # Helper functions
```

## Installation

```bash
pip install -e requirements.txt
# Additional dependencies for LLM inference (transformers, openai, etc.)
```

## Data

Stimuli are stored as CSV files in `data/` with columns:
- `sentence` - The test sentence
- `question` - Comprehension question
- `correct_answer` - Correct response
- `incorrect_answer` - Foil/distractor
- Metadata: `sent_type`, `quest_type`, etc.

Example datasets:
- `nps_human_base_data.csv`
- `npvp_human_base_data.csv`
- `reduced_relative_human_base_data.csv`
- `double_center_human_base_data.csv`
- `depth_charge_human_base_data.csv`

## Running LLM Experiments

### Configuration

Each experiment uses a JSON config file specifying:

```json
{
    "data_path": "data/nps_human_base_data.csv",
    "keys_to_add": ["sent_type", "quest_type"],
    "model_args": [{
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "open_source": true
    }],
    "prefix_path": "prefixes/prefixes.json",
    "results_path": "results/llm_results/deepseek_large/nps.csv"
}
```

### Execution

```bash
python inference/textgen_inference/base_inference.py -c path/to/config
```

## Running Human Experiments

### Creating Prolific Studies

```bash
python human_experiments/create_many_experiments.py \
    --pcibex_url "https://your-pcibex-url.com/experiment" \
    --completion_code "YOUR_PROLIFIC_CODE" \
    --num_groups 5
```

**Parameters:**
- `--pcibex_url` - PCIBex experiment URL
- `--completion_code` - Prolific completion code for payment
- `--num_groups` - Number of between-subjects conditions
- `--missing_info` - (Optional) JSON with missing participant counts per group

The script automatically:
- Creates Prolific studies for each experimental group
- Adds cross-blocking to prevent repeat participation
- Publishes studies

### Parsing Results

```bash
python human_experiments/human_results_parser.py
```

Parsing results will create a dictionary with the groups with a missing number of participants, that can be used to create new prolific studies.


## Citation

If you use this code or data, please cite our paper:

```
@misc{amouyal2025comparinghumanlanguagemodels,
      title={Comparing human and language models sentence processing difficulties on complex structures}, 
      author={Samuel Joseph Amouyal and Aya Meltzer-Asscher and Jonathan Berant},
      year={2025},
      eprint={2510.07141},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07141}, 
}
```