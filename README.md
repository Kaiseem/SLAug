# SLAug

## Rethinking Data Augmentation for Single-source Domain Generalization in Medical Image Segmentation, AAAI 2023. [ArXiv](https://arxiv.org/pdf/2211.14805.pdf)

**Abstract**

Single-source domain generalization (SDG) in medical image segmentation is a challenging yet essential task as domain shifts are quite common among clinical image datasets. Previous attempts most conduct global-only/random augmentation.  Their augmented samples are usually insufficient in diversity and informativeness, thus failing to cover the possible target domain distribution. In this paper, we rethink the data augmentation strategy for SDG in medical image segmentation. Motivated by the class-level representation invariance and style mutability of medical images, we hypothesize that  unseen target data can be sampled from a linear combination of C (the class number) random variables, where each variable follows a location-scale distribution at the class level. Accordingly, data augmented can be readily made by sampling the random variables through a general form. On the empirical front, we implement such strategy with constrained Bezier transformation on both  global and  local (i.e. class-level) regions, which can largely increase the augmentation diversity.  A Saliency-balancing Fusion mechanism is further proposed to enrich the informativeness by engaging the gradient information, guiding augmentation with proper orientation and magnitude. As an important contribution, we prove theoretically that our proposed augmentation can lead to an upper bound of the generalization risk on the unseen  target domain, thus confirming our hypothesis. Combining the two strategies, our Saliency-balancing Location-scale Augmentation (SLAug) exceeds the state-of-the-art works by a large margin in two challenging SDG tasks.

## News:
\[2022/12/1\] We release the training and inference code, even including the pretrained checkpoints and the processed dataset!

\[2022/11/19\] Our paper "Rethinking Data Augmentation for Single-source Domain Generalization in Medical Image Segmentation" accepted by AAAI2023!


## 1. Installation

Clone this repo.
```bash
git clone https://github.com/Kaiseem/SLAug.git
cd SLAug/
```

This code requires PyTorch 1.10 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```


## 2. Data preparation

We conduct datasets preparation following [CSDG](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization)

<details>
  <summary>
    <b>1) Abdominal MRI</b>
  </summary>

0. Download [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/) and put the `/MR` folder under `./data/CHAOST2/` directory

1. Converting downloaded data (T2 SPIR) to `nii` files in 3D for the ease of reading.

run `./data/abdominal/CHAOST2/s1_dcm_img_to_nii.sh` to convert dicom images to nifti files.

run `./data/abdominal/CHAOST2/png_gth_to_nii.ipynp` to convert ground truth with `png` format to nifti.

2. Pre-processing downloaded images

run `./data/abdominal/CHAOST2/s2_image_normalize.ipynb`

run `./data/abdominal/CHAOST2/s3_resize_roi_reindex.ipynb`

The processed dataset is stored in `./data/abdominal/CHAOST2/processed/`

</details>

<details>
  <summary>
    <b>1) Abdominal CT</b>
  </summary>

0. Download [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and put the `/img` and `/label` folders under `./data/SABSCT/CT/` directory

1.Pre-processing downloaded images

run `./data/abdominal/SABS/s1_intensity_normalization.ipynb` to apply abdominal window.

run `./data/abdominal/SABS/s2_remove_excessive_boundary.ipynb` to remove excessive blank region. 

run `./data/abdominal/SABS/s3_resample_and_roi.ipynb` to do resampling and roi extraction.
</details>

The details for cardiac datasets will be given later.

We also provide the [processed datasets](https://drive.google.com/file/d/1WlXGt3Nffzu1bn6co-qaidHjqWH51smU/view?usp=share_link). Download and unzip the file where the folder structure should look this:

```none
SLAug
├── ...
├── data
│   ├── abdominal
│   │   ├── CHAOST2
│   │   │   ├── processed
│   │   ├── SABSCT
│   │   │   ├── processed
│   ├── cardiac
│   │   ├── processed
│   │   │   ├── bSSFP
│   │   │   ├── LGE
├── ...
```

## 3. Inference Using Pretrained Model
Download the [pretrained model](https://drive.google.com/file/d/10VnqWWgiqsU4c5bTz77GKgEtASdlXd29/view?usp=share_link) and unzip the file where the folder structure should look this:

```none
SLAug
├── ...
├── logs
│   ├── 2022-xxxx-xx-xx-xx
│   │   ├── checkpoints
│   │   │   ├── latest.pth
│   │   ├── configs
│   │   │   ├── xx.yaml
├── ...
```

<details>
  <summary>
    <b>1) Cross-modality Abdominal Dataset</b>
  </summary>

For direction CT -> MRI (DICE 88.63), run the command 
```bash
python test.py -r logs/2022-08-06T15-20-35_seed23_efficientUnet_SABSCT
```

For direction MRI -> CT (DICE 83.05), run the command 
```bash
python test.py -r logs/2022-08-06T11-03-14_seed23_efficientUnet_CHAOS
```


</details>

<details>
  <summary>
    <b>2)  Cross-sequence Cardiac Dataset</b>
  </summary>
  
For direction bSSFP -> LEG (DICE 86.69), run the command 
```bash
python test.py -r logs/2022-08-05T21-44-50_seed23_efficientUnet_bSSFP_to_LEG
```

For direction LEG -> bSSFP (DICE 87.67), run the command 
```bash
python test.py -r logs/2022-08-06T00-20-02_seed23_efficientUnet_LEG_to_bSSFP
```
</details>


## 4. Training the model
To reproduce the performance, you need one 3090 GPU


<details>
  <summary>
    <b>1) Cross-modality Abdominal Dataset</b>
  </summary>
  
For direction CT -> MRI, run the command 
```bash
python main.py --base configs/efficientUnet_SABSCT_to_CHAOS.yaml --seed 23
```

For direction MRI -> CT, run the command 
```bash
python main.py --base configs/efficientUnet_CHAOS_to_SABSCT.yaml --seed 23
```
</details>

<details>
  <summary>
    <b>2)  Cross-sequence Cardiac Dataset</b>
  </summary>
  
For direction bSSFP -> LEG, run the command 
```bash
python main.py --base configs/efficientUnet_bSSFP_to_LEG.yaml --seed 23
```

For direction LEG -> bSSFP, run the command 
```bash
python main.py --base configs/efficientUnet_LEG_to_bSSFP.yaml --seed 23
```
</details>

## Acknowledgements

Our codes are built upon [CSDG](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization), thanks for their contribution to the community and the development of researches!

## Citation
If our work or code helps you, please consider to cite our paper. Thank you!

```
@inproceedings{su2023slaug,
  title={Rethinking Data Augmentation for Single-source Domain Generalization in Medical Image Segmentation},
  author={Zixian Su, Kai Yao, Xi Yang, Qiufeng Wang, Jie Sun, Kaizhu Huang},
  booktitle={AAAI},
  year={2023},
}
```
