# PixelCNN for MCMC *Ansatz*

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/code-Lightning-blueviolet"></a>
    <a href="https://hydra.cc/"><img alt="Conf: hydra" src="https://img.shields.io/badge/conf-hydra-blue"></a>
    <a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow"></a>
    <a href="https://dvc.org/"><img alt="Conf: hydra" src="https://img.shields.io/badge/data-dvc-9cf"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://github.com/lucmos/nn-template"><img alt="Template" src="https://shields.io/badge/-nn--template-emerald?style=flat&logo=github&labelColor=gray"></a>
</p>

This is a custom implementation using [PyTorch Lightning](https://pytorchlightning.ai/) of the autoregressive PixelCNN model, see also:
* [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)
* [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)
* [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications](https://arxiv.org/abs/1701.05517)
* [PixelSNAIL: An Improved Autoregressive Generative Model](https://arxiv.org/abs/1712.09763)

## Build the env
We highly recommend to create an isolated environment, e.g., using conda `conda create -n <env> python==3.9` and then install all the dependencies running `pip install -r requirements.txt`.

## Usage
To use, create your user in [Weights and Biases](https://wandb.ai/home) (if you don't have one) and login typing `wandb login`.

Then you have to add train, validation and test datasets, which are `.npy` array of size `[N,L,L]` (`L` is the lattice size), in the correct folder, which is `data/`.

As last step remember to create a `.env` file (using the template `template.env`) that should look as
```bash
export YOUR_TRAIN_DATASET_PATH="/your/root/to/train/dataset"
export YOUR_VAL_DATASET_PATH="/your/root/to/validation/dataset"
export YOUR_TEST_DATASET_PATH="/your/root/to/test/dataset"
export PROJECT_ROOT="/your/project/root"
```
So now you add to your repository the following files
```bash
.             
├── data                    # dataset dir
│   ├── train_dataset.npy
│   ├── validation_dataset.npy      
│   ├── test_dataset.npy
├── .env                    # system-specific env variables, e.g. PROJECT_ROOT
│
```
To run training just type `python src/run.py`.
If you need to modify default parameters just change the `conf/*` files according to your prefereces. 

Best 2 checkpoints, i.e., best two models according to the validation loss, are saved in `/your/project/root/wandb/online-run-YYYYMMDD_HHMMSS-hash/files/pixel-cnn/hash/checkpoints/epoch=XX-step=XXXX.ckpt`. 

To load and use the model to generate new sample run `python src/generate.py --ckpt_path <ckpt_path> --num_sample <num>`.

# About <a href="https://github.com/lucmos/nn-template"><img alt=![] src="https://shields.io/badge/-nn--template-emerald?style=flat&logo=github&labelColor=gray"></a>


Generic template to bootstrap your [PyTorch](https://pytorch.org/get-started/locally/) project. Click on [![](https://img.shields.io/badge/-Use_this_template-success?style=flat)](https://github.com/lucmos/nn-template/generate) and avoid writing boilerplate code for:

- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), lightweight PyTorch wrapper for high-performance AI research.
- [Hydra](https://github.com/facebookresearch/hydra), a framework for elegantly configuring complex applications.
- [DVC](https://dvc.org/doc/start/data-versioning), track large files, directories, or ML models. Think "Git for data".
- [Weights and Biases](https://wandb.ai/home), organize and analyze machine learning experiments. *(educational account available)*
- [Streamlit](https://streamlit.io/), turns data scripts into shareable web apps in minutes.

*`nn-template`* is opinionated so you don't have to be.
If you use this template, please add
[![](https://shields.io/badge/-nn--template-emerald?style=flat&logo=github&labelColor=gray)](https://github.com/lucmos/nn-template)
to your `README`.


### Usage Examples

Checkout the [`mwe` branch](https://github.com/lucmos/nn-template/tree/mwe) to view a minimum working example on MNIST.

## Structure

```bash
.
├── .cache              
├── conf                # hydra compositional config 
│   ├── data
│   ├── default.yaml    # current experiment configuration        
│   ├── hydra
│   ├── logging
│   ├── model
│   ├── optim
│   └── train
├── data                # datasets
├── .env                # system-specific env variables, e.g. PROJECT_ROOT
├── requirements.txt    # basic requirements
├── src
│   ├── common          # common modules and utilities
│   ├── pl_data         # PyTorch Lightning datamodules and datasets
│   ├── pl_modules      # PyTorch Lightning modules
│   ├── run.py          # entry point to run current conf
│   └── ui              # interactive streamlit apps
└── wandb               # local experiments (auto-generated)
```

## Data Version Control

DVC runs alongside `git` and uses the current commit hash to version control the data.

Initialize the `dvc` repository:

```bash
$ dvc init
```

To start tracking a file or directory, use `dvc add`:

```bash
$ dvc add data/ImageNet
```

DVC stores information about the added file (or a directory) in a special `.dvc` file named `data/ImageNet.dvc`, a small text file with a human-readable format.
This file can be easily versioned like source code with Git, as a placeholder for the original data (which gets listed in `.gitignore`):

```bash
git add data/ImageNet.dvc data/.gitignore
git commit -m "Add raw data"
```

### Making changes

When you make a change to a file or directory, run `dvc add` again to track the latest version:

```bash
$ dvc add data/ImageNet
```

### Switching between versions

The regular workflow is to use `git checkout` first to switch a branch, checkout a commit, or a revision of a `.dvc` file, and then run `dvc checkout` to sync data:

```bash
$ git checkout <...>
$ dvc checkout
```

---

Read more in the [docs](https://dvc.org/doc/start/data-versioning)!


## Weights and Biases

Weights & Biases helps you keep track of your machine learning projects. Use tools to log hyperparameters and output metrics from your runs, then visualize and compare results and quickly share findings with your colleagues.

[This](https://wandb.ai/gladia/nn-template?workspace=user-lucmos) is an example of a simple dashboard.

### Quickstart

Login to your `wandb` account, running once `wandb login`.
Configure the logging in `conf/logging/*`.


---


Read more in the [docs](https://docs.wandb.ai/). Particularly useful the [`log` method](https://docs.wandb.ai/library/log), accessible from inside a PyTorch Lightning module with `self.logger.experiment.log`.

> W&B is our logger of choice, but that is a purely subjective decision. Since we are using Lightning, you can replace
`wandb` with the logger you prefer (you can even build your own).
 More about Lightning loggers [here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html).

## Hydra

Hydra is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads.

The basic functionalities are intuitive: it is enough to change the configuration files in `conf/*` accordingly to your preferences. Everything will be logged in `wandb` automatically.

Consider creating new root configurations `conf/myawesomeexp.yaml` instead of always using the default `conf/default.yaml`.


### Sweeps

You can easily perform hyperparameters [sweeps](https://hydra.cc/docs/advanced/override_grammar/extended), which override the configuration defined in `/conf/*`.

The easiest one is the grid-search. It executes the code with every possible combinations of the specified hyperparameters:

```bash
PYTHONPATH=. python src/run.py -m optim.optimizer.lr=0.02,0.002,0.0002 optim.lr_scheduler.T_mult=1,2 optim.optimizer.weight_decay=0,1e-5
```

You can explore aggregate statistics or compare and analyze each run in the W&B dashboard.

---

We recommend to go through at least the [Basic Tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli), and the docs about [Instantiating objects with Hydra](https://hydra.cc/docs/patterns/instantiate_objects/overview).


## PyTorch Lightning

Lightning makes coding complex networks simple.
It is not a high level framework like `keras`, but forces a neat code organization and encapsulation.

You should be somewhat familiar with PyTorch and [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/index.html) before using this template.

## Environment Variables

System specific variables (e.g. absolute paths to datasets) should not be under version control, otherwise there will be conflicts between different users.

The best way to handle system specific variables is through environment variables.

You can define new environment variables in a `.env` file in the project root. A copy of this file (e.g. `.env.template`) can be under version control to ease new project configurations.

To define a new variable write inside `.env`:

```bash
export MY_VAR=/home/user/my_system_path
```

You can dynamically resolve the variable name from Python code with:

```python
get_env('MY_VAR')
```

and in the Hydra `.yaml` configuration files with:

```yaml
${oc.env:MY_VAR}
```
