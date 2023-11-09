## Can Brain Signals Reveal Inner Alignment with Human Languages?
Code for our paper [Can Brain Signals Reveal Inner Alignment with Human Languages?](https://arxiv.org/abs/2208.06348).

In EMNLP Findings 2023.

## Usage

### Set up the Environment

Create a virtual environment and activate it. 

```
python -m venv .env

source .env/bin/activate
```

Install basic requirements.

```
pip install -r requirements.txt
```

### Download Datasets

Download K-EmoCon Dataset [here.](https://zenodo.org/record/3931963)

Download ZuCo Dataset [here.](https://osf.io/q3zws/)

For ZuCo Dataset, please only download task1 and task3.

### Preprocessing data

In the preprocessed folder, preprocessed data is readily available for usage. 
For K-EmoCon, df.csv is used. For ZuCo sentiment analysis, we provide the sentence-level csv in the preprocessed folder.

Preprocessing scripts are provided as well. 

### Training 

The `main_new.py` file is used for training selected models. Arguments are provided for selecting datasets, modalities, models, levels, and tasks. 
Please view the `config.py` file in tandem and customize it as necessary. 


### Plotting

The `plot.py` file is used for plotting TSNE, alignment, and brain topological figures. Arguments are also provided for selecting datasets, modalities, models, levels, tasks, and types of plots. Additionally, `plot_new.py` is used to plot the learning curves. 


### Citation

```
@misc{han2023brain,
      title={Can Brain Signals Reveal Inner Alignment with Human Languages?}, 
      author={William Han and Jielin Qiu and Jiacheng Zhu and Mengdi Xu and Douglas Weber and Bo Li and Ding Zhao},
      year={2023},
      eprint={2208.06348},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```

### Contact

If you have any questions, please contact wjhan@andrew.cmu.edu, jielinq@andrew.cmu.edu.
