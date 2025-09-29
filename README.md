# Accompanying code for exploring representational divergence from causal interventions

This code base contains the code to replicate the experiments presented in the paper
"Addressing divergent representations from causal interventions on neural networks."

The code to replicate each experiment can be found in the folder `divergence`. 

## Installation

You will need to install the version of pyvene contained in this repo in order
to replicate the DAS results. To install the required version, first download/clone
this repo, start
a new virtual environment, and then you can run the following:

```
git clone git@github.com:grantsrb/rep_divergence.git
cd rep_divergence
pip install -e .
```

## Citation
If you use this repository, you can cite the paper:
```bibtex
@misc{grant2024repdivergence,
    title = "Addressing divergent representations from causal interventions on neural networks",
    author = "Grant, Satchel and Han, Simon Jerome and Tartaglini, Alexa and Potts, Christopher",
    month = oct,
    year = "2025",
    publisher = "arXiv",
}
```

Please also consider to citing the pyvene library:
```bibtex
@inproceedings{wu-etal-2024-pyvene,
    title = "pyvene: A Library for Understanding and Improving {P}y{T}orch Models via Interventions",
    author = "Wu, Zhengxuan and Geiger, Atticus and Arora, Aryaman and Huang, Jing and Wang, Zheng and Goodman, Noah and Manning, Christopher and Potts, Christopher",
    editor = "Chang, Kai-Wei and Lee, Annie and Rajani, Nazneen",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-demo.16",
    pages = "158--165",
}
```

