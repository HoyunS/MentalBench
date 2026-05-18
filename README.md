<div align="center">

# MentalBench: A Benchmark for Evaluating Psychiatric Diagnostic Capability of Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.12871-b31b1b.svg)](https://arxiv.org/abs/2602.12871)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MentalBench-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/datasets/hysong/MentalBench)

[Hoyun Song](https://github.com/HoyunS)¹\*, [Migyeong Kang](https://github.com/gyeong707)²\*, Jisu Shin¹, Jihyun Kim², Chanbi Park³,

Hangyeol Yoo⁴,  Jihyun An⁵, Alice Oh¹, Jinyoung Han²†, KyungTae Lim¹†

¹KAIST, ²Sungkyunkwan University, ³Dongguk University Medical Center, ⁴Seoul National University of Science and Technology, ⁵Samsung Medical Center

\*Equal contribution, †Corresponding author

</div>

## 🌟 Overview

**MentalBench** is a comprehensive benchmark designed to evaluate the psychiatric diagnostic capabilities of Large Language Models (LLMs). As the application of LLMs in healthcare expands, ensuring their reliability in sensitive domains like psychiatry is crucial. 

MentalBench provides a robust evaluation framework, grounded in real-world psychiatric knowledge. To facilitate deeper reasoning and grounded evaluation, this benchmark is integrated with **MentalKG**, a specialized knowledge graph structured for psychiatric domain knowledge.

### 1. MENTALKG: The Logical Backbone
At the core of our benchmark is MentalKG, a psychiatrist-built and validated knowledge graph that encodes DSM-5 diagnostic criteria and differential diagnostic rules for **23 psychiatric disorders**.

<p align="center">
  <img width="600"  alt="Image" src="https://github.com/user-attachments/assets/cfabe646-82a4-47d3-b79a-11eeb1e9a61f" />
  <br>
  <em>Figure 1: Example of the MentalKG schema. It demonstrates the relational dependencies among disorders, symptom groups, and symptoms, as well as directional differential diagnoses. It also details the granular clinical attributes embedded in nodes and edges, including discriminating rules, symptom subtypes, and diagnostic constraints (e.g., duration and thresholds).</em>
</p>

The benchmark is built upon a psychiatric knowledge graph containing:
- **23 Mental Disorders** based on DSM-5 criteria
- **Symptom Definitions** with detailed descriptions and subtypes
- **Diagnostic Criteria** including duration, functional impairment, and stressor requirements
- **Differential Diagnosis Rules** for distinguishing between similar disorders


### 2. MENTALBENCH: Clinical Case Generation
Using MENTALKG as a golden-standard logical backbone, we generated **24,750 synthetic clinical cases**. These cases systematically vary in information completeness (from structured medical charts to incomplete patient self-reports) and diagnostic complexity (from single-disorder to challenging differential diagnosis scenarios).

<p align="center">
  <img width="800" alt="Image" src="https://github.com/user-attachments/assets/78d6b670-1464-4e75-94f2-b6956a3fb60f" />
  <br>
  <em>Figure 2: Overview of the clinical case generation framework for constructing MentalBench across Single-Disease Identification and Differential Diagnosis scenarios.</em>
</p>



## 👨‍⚕️ Expert Validation
To ensure rigorous clinical reliability, the entire framework was developed and evaluated in close collaboration with mental health professionals, including a board-certified psychiatrist and a licensed clinical psychologist.

* 🧠 **MentalKG Validation:** Experts thoroughly verified the alignment of our formalized diagnostic criteria with DSM-5 standards and the clinical validity of the complex differential diagnosis logic.
* 📝 **Clinical Case Evaluation:** A random sample of 220 generated clinical scenarios was blindly evaluated by the experts on a 5-point Likert scale. The benchmark achieved exceptionally high scores across all key dimensions:
  * **Linguistic Naturalness:** 4.95 / 5.0
  * **Clinical Realism:** 4.89 / 5.0
  * **Diagnostic Validity:** 4.44 / 5.0


## 🎯 Question Types

| Type | Description | Difficulty | Number of Samples |
|------|-------------|------------|-------------------|
| **Type 1** | Medical Chart → Single Answer | Low | 1,725 |
| **Type 2** | Patient Self-Report → Single Answer | Medium | 3,450 |
| **Type 3** | Ambiguous Type → Multiple Answer | High | 6,525 |
| **Type 4** | Clear Type → Single Answer | High | 13,050 |


## 📊 Key Results

Our experiments reveal critical insights into the diagnostic capabilities of current state-of-the-art LLMs:

**Performance Highlights:**
* **Top Performers:** Claude Sonnet-4.5 and GPT-5.1 achieved the highest overall accuracy of 62.69% and 62.17%, respectively.
* **The Calibration Gap:** While models perform well on structured queries probing DSM-5 knowledge, they struggle to calibrate confidence in diagnostic decision-making when distinguishing between clinically overlapping disorders.
* **Over-diagnosis vs. Under-diagnosis:** Open-source models exhibit excessive commitment leading to over-diagnosis, whereas proprietary models show rigid strictness resulting in under-diagnosis.


---


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

---

## 📝 Citation

If you find MentalBench and MentalKG useful for your research, please cite our paper:

```bibtex
@article{song2026mentalbench,
    title={MentalBench: A Benchmark for Evaluating Psychiatric Diagnostic Capability of Large Language Models},
    author={Song, Hoyun and Kang, Migyeong and Shin, Jisu and Kim, Jihyun and Park, Chanbi and Yoo, Hangyeol and An, Jihyun and Oh, Alice and Han, Jinyoung and Lim, KyungTae},
    journal={arXiv preprint arXiv:2602.12871},
    year={2026}
  }
```
