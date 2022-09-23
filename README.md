## ECG_Language_Alignment

An Empirical Exploration of Cross-domain Alignment between Language and Electroencephalogram

[William Han<sup>*</sup>](https://willxxy.github.io/), [Jielin Qiu<sup>*</sup>](https://www.cs.cmu.edu/~jielinq/), [Jiacheng Zhu](https://jiachengzhuml.github.io/), [Mengdi Xu](https://mxu34.github.io/), [Douglas Weber](https://www.meche.engineering.cmu.edu/directory/bios/weber-douglas.html), [Bo Li](https://aisecure.github.io/), [Ding Zhao](https://safeai-lab.github.io/)

Under Review / [arxiv](https://arxiv.org/abs/2208.06348)

## Usage

### Set up Environment

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


### Set up directories

From the root directory, please create directories for results like so:
```
results
  |-biLSTM_eeg
  |-biLSTM_text
  |-CCA_ds_eeg
  |-CCA_ds_text
  |-CCA_fusion_fusion
  |-fusion_fusion
  |-MLP_eeg
  |-MLP_text
  |-resnet_eeg
  |-resnet_text
  |-transformer_eeg
  |-transformer_text
  |-WD_ds_eeg
  |-WD_ds_text
  |-WD_fusion_fusion
```

### Preprocessing data

In the preprocessed folder, preprocessed data is readily available for usage. 
For K-EmoCon, df.csv is used. For ZuCo sentiment analysis and relation detection, respective csv's for sentence-level, word-level, and concat word-level are provided. 

Preprocessing scripts are provided as well. 

### Training 

The `main.py` file is used for training selected models. Arguments are provided for selecting datasets, modalities, models, levels, and tasks. 
Please view the `config.py` file in tandem and customize as necessary. 


### Plotting

The `plot.py` file is used for plotting TSNE, alignment, and brain topological figures. Arguments are also provided for selecting datasets, modalities, models, levels, tasks, and type of plots. 


### Citation

```
@article{han_2022_an,
	author = {Han, William and Qiu, Jielin and Zhu, Jiacheng and Xu, Mengdi and Weber, Douglas and Li, Bo and Zhao, Ding},
	journal = {arXiv:2208.06348 [cs, q-bio]},
	month = {08},
	title = {An Empirical Exploration of Cross-domain Alignment between Language and Electroencephalogram},
	url = {https://arxiv.org/abs/2208.06348},
	urldate = {2022-08-17},
	year = {2022},
	bdsk-url-1 = {https://arxiv.org/abs/2208.06348}}
```

### Reference

This is the same codebase as in: https://github.com/willxxy/EEG_Text_alignment

### Contact

If you have any question, please contact willhan327@outlook.com, jielinq@andrew.cmu.edu.
