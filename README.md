<div align="center">

# MentalBench: A Benchmark for Evaluating Psychiatric Diagnostic Capability of Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2510.18383-b31b1b.svg)](https://arxiv.org/abs/2602.12871)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

[Hoyun Song](https://github.com/HoyunS)¹\*, [Migyeong Kang](https://github.com/gyeong707)²\*, Jisu Shin¹, Jihyun Kim², Chanbi Park³,
Hangyeol Yoo⁴,  Jihyun An⁵, Alice Oh¹, Jinyoung Han²†, KyungTae Lim¹†

¹KAIST, ²Sungkyunkwan University, ³Dongguk University Medical Center, ⁴Seoul National University of Science and Technology, ⁵Samsung Medical Center

\*Equal contribution, †Corresponding author

</div>

## 📁 Project Structure

```
MENTALBENCH/
├── scripts/                                # Core evaluation and preprocessing scripts
│   ├── eval_type12.py                      # Evaluation script for Type 1 & 2 questions
│   ├── eval_type34.py                      # Evaluation script for Type 3 & 4 questions
│   ├── knowledge_graph.py                  # Knowledge graph construction and utilities
│   ├── models.py                           # LLM inference utilities (API & vLLM)
│   ├── prompts.py                          # Prompt templates for different question types
│   ├── demographics.py                     # Patient demographic generation
│   ├── run_script_type12.sh                # Batch evaluation script for Type 1 & 2
│   ├── run_script_type34.sh                # Batch evaluation script for Type 3 & 4
│   └── preprocess_dataset/                 # Dataset preprocessing utilities
│       ├── select_feature.py               # Feature selection for questions
│       ├── select_feature_diff_diag.py     # Differential diagnosis feature selection
│       ├── merge_feature.py                # Feature merging and validation
│       └── variate_feature.py              # Feature variation generation
│
└── resources/                              # Data resources
    ├── knowledge_graph/                    # Psychiatric knowledge graph
    │   └── EN/                             # English version
    │       ├── disorder.json               # Mental disorder definitions
    │       ├── diagnostic_criteria.json    # DSM-based diagnostic criteria
    │       ├── symptom/                    # Symptom definitions
    │       └── differential_diagnosis/     # Differential diagnosis rules
    ├── features/                           # Sampled and validated features
    └── dataset/                            # Generated benchmark dataset
        ├── low/                            # Low difficulty (clinical summaries)
        ├── medium/                         # Medium difficulty (patient vignettes)
        └── high/                           # High difficulty (differential diagnosis)
```

## 🎯 Question Types

| Type | Description | Difficulty |
|------|-------------|------------|
| **Type 1** | Clinical case summary → Single diagnosis | Low |
| **Type 2** | Patient vignette → Single diagnosis | Medium |
| **Type 3** | Ambiguous presentation → Multiple possible diagnoses | High |
| **Type 4** | Differential diagnosis → Distinguish between similar disorders | High |

## 🚀 Getting Started

### Prerequisites

```bash
pip install openai vllm transformers tqdm pandas networkx matplotlib
```

### Running Evaluations

We provide shell scripts for batch evaluation. Before running, configure the model list and API keys in the scripts.

#### For Local Models (vLLM) - Type 1 & 2 Questions

Edit `scripts/run_script_type12.sh` to configure models:

```bash
model_list=(
    "Qwen/Qwen3-8B"
    # "Qwen/Qwen3-14B"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # Add more models as needed
)
```

Run the evaluation:

```bash
cd scripts
bash run_script_type12.sh
```

#### For API Models (OpenAI, Gemini, Claude) - Type 3 & 4 Questions

Edit `scripts/run_script_type34.sh` to configure models and API keys:

```bash
model_list=(
    "google/gemini-2.5-flash"
    "google/gemini-2.5-pro"
    "anthropic/claude-haiku-4.5"
    "anthropic/claude-sonnet-4.5"
)

# Set your API keys in the script
--api_key "YOUR_OPENROUTER_API_KEY"  # For Gemini/Claude via OpenRouter
--api_key "YOUR_OPENAI_API_KEY"      # For GPT models
```

Run the evaluation:

```bash
cd scripts
bash run_script_type34.sh
```

Logs will be saved in `scripts/logs/` and `scripts/logs_clear/` directories.

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Model name (HuggingFace path or API model name) |
| `--tp` | Tensor parallelism for vLLM (default: 1) |
| `--difficulty` | Question difficulty: `low`, `medium`, or `high` |
| `--prompt_style` | Prompt style: `default`, `single`, or `clear` |
| `--output_dir` | Output directory for results |
| `--api_key` | API key for OpenAI/OpenRouter |

## 📊 Supported Models

### API Models
- OpenAI: `gpt-4o`, `gpt-5-mini`, `gpt-5.1`
- Google: `google/gemini-2.5-flash`, `google/gemini-2.5-pro`
- Anthropic: `anthropic/claude-haiku-4.5`, `anthropic/claude-sonnet-4.5`

### Local Models (via vLLM)
- Qwen: `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen2.5-7B-Instruct`, etc.
- LLaMA: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`
- Gemma: `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`
- MentaLLaMA: `klyang/MentaLLaMA-chat-7B`, `klyang/MentaLLaMA-chat-13B`

## 🧠 Knowledge Graph

The benchmark is built upon a psychiatric knowledge graph containing:
- **23 Mental Disorders** based on DSM-5 criteria
- **Symptom Definitions** with detailed descriptions and subtypes
- **Diagnostic Criteria** including duration, functional impairment, and stressor requirements
- **Differential Diagnosis Rules** for distinguishing between similar disorders

## 📈 Evaluation Metrics

Results are reported as accuracy scores per disorder and overall:
- Per-disease accuracy breakdown
- Total accuracy across all questions
- Support for multi-label evaluation (Type 3 questions)

## 📄 Output Format

Evaluation results are saved as JSON files containing:
- Question and options
- Ground truth answer
- Model prediction
- Evaluation result (correct/incorrect)
