# Portrait-Mode Video Recognition

<a href='https://mingfei.info/PMV/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://github.com/bytedance/Portrait-Mode-Video/blob/master/DATA.md'><img src='https://img.shields.io/badge/Github-Data-red'></a>
<a href='https://arxiv.org/abs/2312.13746'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://github.com/bytedance/Portrait-Mode-Video'><img src='https://img.shields.io/badge/Github-Code-green'></a>

We are releasing our code and dataset regarding Portrait-Mode Video Recognition research. The videos are sourced from [Douyin platform](https://www.douyin.com). We distribute video content through the provision of links. Users are responsible for downloading the videos independently. 

## Videos
The high-quality videos are filtered by humans, with human activities across wide-spread categories.

🚀🚀 Thanks for the support from the community. Please check the issue [here](https://github.com/bytedance/Portrait-Mode-Video/issues/7) for cached videos on [OneDrive](https://1drv.ms/f/c/8d9d5fbede2ace9d/Ep3OKt6-X50ggI2MAAAAAAABV0VlHe1CPMEbHIJ1ytZYZA?e=d1LJkF). 

🚀🚀 Please check the annotation at `Uniformer/data_list/PMV/` and text description of the categories at `data/class_name_mapping.csv`

## Taxonomy
Please check our released taxonomy [here](./data/class_name_mapping.csv). There is also an interactive demo of the taxonomy [here](https://mingfei.info/PMV/PMV_400_taxonomy.html).

## Usage
We assume two directories for this project. `{CODE_DIR}` for the code respository; `{PROJ_DIR}` for the model logs, checkpoints and dataset. 

To start with, please clone our code from Github
```bash
git clone https://github.com/bytedance/Portrait-Mode-Video.git {CODE_DIR}
```

### Python environment
We train our model with Python 3.7.3 and Pytorch 1.10.0. Please use the following command to install the packages used for our project.
First install pytorch following the [official instructions](https://pytorch.org/get-started/previous-versions/#v1100). Then install other packages by
```bash
pip3 install -r requirements.txt
```

### Data downloading
Please refer to [DATA.md](./DATA.md) for data downloading. We assume the videos are stored under `{PROJ_DIR}/PMV_dataset`. Category IDs for the released videos are under `{CODE_DIR}/MViT/data_list/PMV` and `{CODE_DIR}/Uniformer/data_list/PMV`.

### Training
We provide bash scripts for training models using our PMV-400 data, as in `exps/PMV/`. A demo running script is
```bash
bash exps/PMV/run_MViT_PMV.sh
```
For each model, e.g., `MViT`, we provide the scripts for different training recipes in a single bash scripts, e.g., `exps/PMV/run_MViT_PMV.sh`. Please choose the one suiting your purpose.

Note that you should set some environment variables in the bash scripts, such as `WORKER_0_HOST`, `WORKER_NUM` and `WORKER_ID` in `run_SlowFast_MViTv2_S_16x4_PMV_release.sh`; `PROJ_DIR` in `run_{model}_PMV.sh`.

### Inference
We provide inference scripts for obtaining the report results in our paper. We also provide the trained model checkpoints.


## License
Our code is licensed under an [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).
Our data is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). The data is released for non-commercial research purposes only. 

By engaging in the downloading process, users are considered to have agreed to comply with our distribution license terms and conditions.

---
We would like to extend our thanks to the teams behind [SlowFast code repository](https://github.com/facebookresearch/SlowFast), [3Massiv](https://github.com/ShareChatAI/3MASSIV), [Kinetics](https://research.google/pubs/the-kinetics-human-action-video-dataset/) and [Uniformer](https://github.com/Sense-X/UniFormer). Our work builds upon their valuable contributions. Please acknowledge these resources in your work.
