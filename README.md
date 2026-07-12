# Representation Alignment and Model Stitching Across Modalities

This project studies how internal representations can be compared, aligned, and transferred across neural models.

The main goals are to understand:

- whether different models learn similar internal representations,
- where weaker and stronger models begin to diverge,
- whether representation differences relate to hallucinations or missing knowledge,
- whether lightweight stitching methods can transfer useful information between models.

## Team

### Jahnavi Sharma: LLM Representation Similarity

Jahnavi's work focuses on comparing hidden-state representations across language models through layer-wise Linear CKA and similarity visualizations.

Current model comparisons include:

- TinyLlama-1.1B vs. Qwen2.5-0.5B
- Llama-3.2-1B vs. Llama-3.2-3B

### Palaash Bhathena: Direct Matching and Model Stitching

Palaash's work focuses on aligning representations between smaller and larger language models using Direct Matching.

The main questions include where hallucination-related divergence begins, whether weaker models contain latent knowledge they fail to express, and whether lightweight adapters can translate between model representation spaces.

### Ismail Jamal: Audio and Time-Series Alignment

Ismail's work extends representation alignment and stitching to speech, audio-language models, and physiological time-series data.

A key challenge is aligning different temporal resolutions before applying stitching methods.

## Repository Structure

```text
llm_experiment/
├── data/
│   └── factual_qa/
│       └── factual_qa.json
├── experiments/
│   ├── jahnavi/
│   ├── palaash/
│   └── ismail/
├── results/
│   ├── tinyllama/
│   ├── qwen/
│   ├── llama1b/
│   └── llama3b/
├── figures/
│   ├── tinyllama_qwen/
│   └── llama1b_llama3b/
├── README.md
├── requirements.txt
└── .gitignore
```

## Datasets

- TruthfulQA, loaded dynamically through the Hugging Face `datasets` library
- Small custom factual QA dataset stored at `data/factual_qa/factual_qa.json`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Some Llama models require Hugging Face authentication:

```bash
huggingface-cli login
```

### Running Jahnavi's Experiments

Run commands from the repository root.

**Extract hidden states**

```bash
python experiments/jahnavi/hidden_states.py
python experiments/jahnavi/hidden_states_qwen.py
python experiments/jahnavi/hidden_states_llama1b.py
python experiments/jahnavi/hidden_states_llama3b.py
```

**Compute CKA**

```bash
python experiments/jahnavi/cka.py
python experiments/jahnavi/cka_llama.py
```

**Generate heatmaps**

```bash
python experiments/jahnavi/heatmap.py
python experiments/jahnavi/heatmap_llama.py
```

## Collaboration Guidelines

- Place experiment-specific code inside your folder under `experiments/`.
- Store datasets in `data/`.
- Store generated outputs in `results/`.
- Store plots in `figures/`.
- Do not commit virtual environments, model checkpoints, `.pt` files, or large generated tensors.

Before pushing changes:

```bash
git pull --rebase origin main
git add .
git commit -m "Describe your changes"
git push origin main
```

## Contributors

- Jahnavi Sharma
- Palaash Bhathena
- Ismail Jamal
- Krishna Praneet Gudipaty, research supervisor

University of Massachusetts Amherst  
Summer Research Project
