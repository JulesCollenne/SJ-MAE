# SJ-MAE: Joint Masked, Jigsaw, and Contrastive Pretraining for Vision Transformers

Welcome to the official repository for the paper "SJ-MAE: Joint Masked, Jigsaw, and Contrastive Pretraining for Vision Transformers"!
Our approach is inspired by recent advancements in visual representation learning, including Masked AutoEncoders, Jigsaw-ViT, and SimSiam. It aims to create a more comprehensive model by integrating all these tasks.

## Abstract
Multi-task self-supervised learning (SSL) remains mostly underexplored despite the recent advances in SLL and their diverse set of pretext objectives. Combining multiple objectives during pretraining could open new possibilities and make learnt representations more robust. Following this idea, we present SJ-MAE, a unified framework that combines masked image modeling, spatial reasoning through jigsaw puzzles, and contrastive alignment within a single ViT encoder. This joint training enforces both low-level reconstruction and high-level semantic discrimination, leading to stronger and more versatile representations. We analyze the interplay between tasks and show how loss balancing and positional encoding design are critical to success. Our results suggest that a well-designed multi-task SSL framework can surpass the limitations of single-task pretext learning at the cost of more heavy computations.


## Repository Structure
The code will generate missing folders if they do not exist.
```
/
|- configs/               # Directory for configuration files
|- models/                # Directory containing model implementations
|- util/                  # Utility code directory
|- main_pretrain.py       # Script to run pretraining (usage: `python main_pretrain.py --config configs/config.yaml`)
|- main_linprobe.py       # Script to run linear probing
|- main_finetune.py       # Script to run finetuning
|- engine_finetune.py     # Utility functions for linear probing and finetuning
|- engine_pretrain.py     # Utility functions for pretraining
|- LICENSE                # License file
|- README.md              # Readme file
|- requirements.txt       # File listing the project dependencies
```

## Getting Started


To get started with this repository, first clone it and install the required dependencies:

```
git clone https://github.com/JulesCollenne/SJ-MAE.git
cd SJ-MAE
pip install -r requirements.txt
```
Ensure you customize the configuration files according to your needs. Then, you can begin the pretraining process by running:
```
python main_pretrain.py --config configs/config.yaml
```

##  Citation
Soon!

[//]: # (If you find our work useful in your research, please consider citing:)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # (@article{collenne2024,)

[//]: # ()
[//]: # (  title={ReSet: A Residual Set-Transformer approach to tackle the ugly-duckling sign in melanoma detection},)

[//]: # ()
[//]: # (  author={Collenne, Jules and Iguernaissi, Rabah and Dubuisson, Severine and Merad, Djamal},)

[//]: # ()
[//]: # (  journal={},)

[//]: # ()
[//]: # (  year={2024})

[//]: # ()
[//]: # (})

[//]: # ()
[//]: # (```)

[//]: # (And the SetTransformer paper:)

[//]: # (```)

[//]: # (@InProceedings{lee2019set,)

[//]: # (    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},)

[//]: # (    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},)

[//]: # (    booktitle={Proceedings of the 36th International Conference on Machine Learning},)

[//]: # (    pages={3744--3753},)

[//]: # (    year={2019})

[//]: # (})

[//]: # (```)