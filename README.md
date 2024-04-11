<div style="display: flex; align-items: center;">
  <img src="assets/galaxy_brain.gif" alt="galaxy brain" style="height: 2em; margin-right: 20px;">
  <h1 >MindBridge: A Cross-Subject <br> Brain Decoding Framework</h1>
</div>

<!-- # MindBridge: A Cross-Subject Brain Decoding Framework -->

![teasor](assets/MindBridge_teaser.png)

[Shizun Wang](https://littlepure2333.github.io/home/), [Songhua Liu](http://121.37.94.87/), [Zhenxiong Tan](https://github.com/Yuanshi9815), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)    
National University of Singapore

**CVPR 2024 Highlight**   
[Project](https://littlepure2333.github.io/MindBridge/) | [Arxiv]() 

## News
**[2024.04.12]** MindBridge's paper, project and code are released.    
**[2024.04.05]** MindBridge is selected as CVPR 2024 Highlight paper!    
**[2024.02.27]** MindBridge is accepted by CVPR 2024!

## Overview
![method](assets/MindBridge_method.png)

> we present a novel approach, MindBridge, that achieves cross-subject brain decoding by employing only one model. Our proposed framework establishes a generic paradigm capable of addressing these challenges: **1) the inherent variability** in input dimensions across subjects due to differences in brain size; **2) the unique intrinsic neural patterns**, influencing how different individuals perceive and process sensory information; **3) limited data availability for new subjects** in real-world scenarios hampers the performance of decoding models. 
Notably, by cycle reconstruction, MindBridge can enable novel brain signals synthesis, which also can serve as pseudo data augmentation. Within the framework, we can adapt a pretrained MindBridge to a new subject using less data.

## Installation

1. Agree to the Natural Scenes Dataset's [Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) and fill out the [NSD Data Access form](https://forms.gle/xue2bCdM9LaFNMeb7)

2. Download this repository: ``git clone https://github.com/littlepure2333/MindBridge.git``

3. Create a conda environment and install the packages necessary to run the code.

```bash
conda create -n mindbridge python=3.10.8 -y
conda activate mindbridge
pip install -r requirements.txt
```

## Data

Download the essential files we used from [NSD dataset](https://natural-scenes-dataset.s3.amazonaws.com/index.html), which contains `nsd_stim_info_merged.csv`, `captions_train2017.json` and `captions_val2017.json`.
We use the same preprocessed data as [MindEye's](https://github.com/MedARC-AI/fMRI-reconstruction-NSD), which can be downloaded from [Huggingface](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/webdataset_avg_split), and extract all files from the compressed tar files.
Then organize the data as following:

```
data/natural-scenes-dataset
├── nsddata
│   └── experiments
│       └── nsd
│           └── nsd_stim_info_merged.csv
├── nsddata_stimuli
│   └── stimuli
│       └── nsd
│           └── annotations
│              ├── captions_train2017.json
│              └── captions_val2017.json
└── webdataset_avg_split
    ├── test
    │   ├── subj01
    │   │   ├── sample000000349.coco73k.npy
    │   │   ├── sample000000349.jpg
    │   │   ├── sample000000349.nsdgeneral.npy
    │   │   └── ...
    │   └── ...
    ├── train
    │   ├── subj01
    │   │   ├── sample000000300.coco73k.npy
    │   │   ├── sample000000300.jpg
    │   │   ├── sample000000300.nsdgeneral.npy
    │   │   └── ...
    │   └── ...
    └── val
        ├── subj01
        │   ├── sample000000000.coco73k.npy
        │   ├── sample000000000.jpg
        │   ├── sample000000000.nsdgeneral.npy
        │   └── ...
        └── ...
```


## Training on single subject
This script contains training the per-subject-per-model version of MindBridge (which refers to "Vanilla" in the paper) on one subject (e.g. subj01). The training progress can be monitored through [wandb](https://wandb.ai/). You can also indicate which subject in the script.

```bash
bash scripts/train_single.sh
```

## Training on multi-subjects
This script contains training MindBridge on multi-subjects (e.g. subj01, 02, 05, 07). The training progress can be monitored through [wandb](https://wandb.ai/). You can also indicate which subjects in the script.

```bash
bash scripts/train_bridge.sh
```

## Adapting to a new subject
Once the MindBridge is trained on some known "source subjects" (e.g. subj01, 02, 05), you can adapt the MindBridge to a new "target subject" (e.g. subj07) based on limited data volume (e.g. 4000 data points). The training progress can be monitored through [wandb](https://wandb.ai/). You can also indicate which source subjects, which target subject, or data volume (length) in the script.

```bash
bash scripts/adapt_bridge.sh
```

## Reconstructing and evaluating
This script will reconstruct one subject's images (e.g. subj01) on the test set from a MindBridge model (e.g. subj01, 02, 05, 07), then calculate all the metrics. The evaluated metrics will be saved in a csv file. You can indicate which MindBridge model and which subject in the script.

```bash
bash scripts/inference.sh
```


## TODO List
- [ ]  Release pretrained checkpoints.
- [ ]  Training MindBridge on all 8 subjects in NSD dataset.

## Citation
```
@inproceedings{wang2024mindbridge,
  author    = {Shizun Wang, Songhua Liu, Zhenxiong Tan, Xinchao Wang},
  title     = {MindBridge: A Cross-Subject Brain Decoding Framework},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024}
}
```

## Acknowledgement
We extend our gratitude to [MindEye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD) and [nsd_access](https://github.com/tknapen/nsd_access) for generously sharing their codebase, upon which ours is built. We are indebted to the [NSD dataset](https://natural-scenes-dataset.s3.amazonaws.com/index.html) for providing access to high-quality, publicly available data. 
Our appreciation also extends to the [Accelerate](https://huggingface.co/docs/accelerate/index) and [DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) for simplifying the process of efficient multi-GPU training, enabling us to train on the 24GB vRAM GPU, NVIDIA A5000.
Special thanks to [Xingyi Yang](https://adamdad.github.io/) and [Yifan Zhang](https://sites.google.com/view/yifan-zhang) for their invaluable discussions.

<!-- <div align="center">
    <img src="assets/galaxy_brain.gif" alt="galaxy brain" height=100 />
</div> -->


