# nanoGPT fork for Dickinson COMP560, spring 2026

In this fork we implement minimal changes to [nanoGPT](https://github.com/karpathy/nanoGPT), allowing it to be used for experiments from other repositories. At the time of writing (December 2025), there is only one such change, described in the following section. It is anticipated that other changes will be needed.

## NANOGPT_CONFIG environment variable

The `train.py` and `sample.py` both employ a new environment variable called `NANOGPT_CONFIG`. It holds the filepath (absolute or relative) to the configurator file e.g. `../../comp560-nanoGPT/configurator.py`. If `NANOGPT_CONFIG` is not defined, `configurator.py` is assumed to be in the current working directory.

The motivation for this is that we can run `train.py` and `sample.py` from outside the nanoGPT repo. For example, suppose we have the following layout:
```
git
├── comp560-nanoGPT
│   ├── sample.py
│   └── train.py
└── comp560-jmac
    └── alphabet
        ├── README.md
        ├── config
        │   ├── basic.py
        │   ├── mixed.py
        │   └── reverse.py
        ├── data
        ├── out
        └── useful-commands.md
```


Then we can run the following bash command from the working directory `alphabet` inside the `comp560-jmac` repo:
```
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py  python -u ../../comp560-nanoGPT/train.py config/basic.py
```
