# Divergence Analysis

This folder contains code for reproducing the divergence analysis experiments and figures from the paper.

## Setup

### Hugging Face API Key

Before running any experiments replicating Figure 2 and parts of Figure 3, you need to add your Hugging Face API key to `your_token.txt`. This file is used to authenticate with Hugging Face when downloading models and datasets.

1. Obtain your Hugging Face API token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Add the token to `divergence/your_token.txt` (one line, just the token string) (don't commit this if you fork this repo...)

## Generating Divergence Figures

The data for the divergence figures can be generated using the following notebooks:

- **`CL_Boundless_DAS.ipynb`**: Generates data for CL (Contrastive Learning) Boundless DAS divergence experiments
- **`fengandsteinhardt_replication.ipynb`**: Replicates experiments from Feng & Steinhardt's work
- **`sae_divergence.ipynb`**: Generates data for SAE (Sparse Autoencoder) divergence experiments

After generating the data, the figures can be created using:

- **`emd_vis.ipynb`**: Creates visualization figures from the collected divergence data

## Section 4.2 Replication

The mean-difference intervention experiments described in Section 4.2 can be replicated using:

- **`mean_diff_section_4.2_sanity_check.py`**: A sanity check script that verifies the mean-difference intervention calculations and demonstrates the key results from Section 4.2.1

Run with:
```bash
python mean_diff_section_4.2_sanity_check.py
```

## Counterfactual Latent (CL) Figures

The CL loss figures can be recreated using: 

- **`collect_cl_boundless_das.py`**: performs Boundless DAS trainings for different weightings of the CL loss using the experimental setup from Wu et al. 2023.
- **`clloss_synthetic_task.ipynb`**: Notebook for performing the modified, standalone CL loss on synthetic tasks

Run the collection script with:
```bash
python divergence/collect_cl_boundless_das.py
```

## Output Files

- **`csvs/`**: Contains CSV files with collected metrics and divergence measurements
- **`figs/`**: Contains generated visualization figures
- **`data/`**: Contains preprocessed data files used in experiments
- **`results/`**: Contains structured results organized by model and experiment type

