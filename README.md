# Denoising Hamiltonian Network for Physical Reasoning

Created by
[Congyue Deng](https://cs.stanford.edu/~congyue/),
[Brandon Y. Feng](https://brandonyfeng.github.io/),
[Cecilia Garraffo](https://www.ceciliagarraffo.com/),
[Alan Garbarz](https://scholar.google.com/citations?user=vz9CBV0AAAAJ&hl=en),
[Robin Walters](https://www.robinwalters.com/),
[William T. Freeman](https://billf.mit.edu/),
[Leonidas Guibas](https://geometry.stanford.edu/?member=guibas), and
[Kaiming He](https://people.csail.mit.edu/kaiming/)

[**Paper**](https://arxiv.org/pdf/2503.07596) | [**Project**](https://cs.stanford.edu/~congyue/dhn/)

<img src='visuals/teaser.png' width=90%>

\
This repository provides a simple implementation of DHN to help you get started quickly. For the full code for all experiments in the paper, check out [**this repo**](https://github.com/FlyingGiraffe/dhn-exps).

## Data Preparation

Download the data from [**Google Drive**](https://drive.google.com/file/d/1ry6Lz1Wa77UZmApEghfm9aSydXU2vWWr/view?usp=sharing) and upzip it to the folder `data` into the following format:
```
data/
 ├── single_pendulum/
 │     ├── train/
 │     └── test
 └── double_pendulum/
       ├── train/
       └── test
```

You can also generate the data yourself by running
```
bash scripts/data_gen_train.sh
bash scripts/data_gen_test.sh
```
Change the variable `DATA_NAME` to generate simulated data for different physical systems (`single_pendulum` or `double_pendulum`).

## Experiments

Change the variable `EXP_NAME` in the scripts to run with different config files.
All experimental results, including logs and checkpoints, will be under the directory `results/${EXP_NAME}`.

### Forward Simulation

**Fitting known trajectories**\
Step 1: Run `train.sh`.\
Step 2: Run `generate.sh`.\
The generated sequences will be in a subfolder named `gen_sequence`.

**Completion on novel trajectories**\
Step 1: Run `train.sh`.\
Step 2: Run `extract_partial.sh`.\
Step 3: Run `generate_partial.sh`.\
The generated sequences will be in a subfolder named `extract/gen_sequence`.

## License
MIT License