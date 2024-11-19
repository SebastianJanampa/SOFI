# SOFI
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SebastianJanampa/SOFI/blob/main/online_demo.ipynb) [![arXiv](https://img.shields.io/badge/arXiv-2108.03144-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2409.15553) 
 
This repository contains the official code and pretrained models for **SOFI** (multi-**S**cale def**O**rmable trans**F**ormer for camera calibrat**I**on with enhanced line queries)

Camera calibration estimates the camera parameters, such as the zenith vanishing point and horizon line. In addition, estimating the camera parameters allows other tasks like 3D rendering, artificial reality effects, and object insertion in an image. Transformer-based models have provided promising results; however, they lack cross-scale interaction. In this work, we introduce multi-**S**cale def**O**rmable trans**F**ormer for camera calibrat**I**on with enhanced line queries, SOFI. SOFI improves the line queries used in CTRL-C and MSCC by using line content and geometric features. Moreover, SOFI's line queries allow transformer models to adopt the multi-scale deformable attention mechanism to promote cross-scale interaction between the feature maps produced by the backbone. SOFI outperforms existing methods on the *Google Street View*, *Horizon Line in the Wild*, and *Holicity* datasets while keeping a competitive inference speed.

<img src="figs/architecture.png" alt="Model Architecture"/>

## Results & Checkpoints

### Google Street View Dataset (Training)
|Model| Up Dir (◦) | Pitch (◦) | Roll (◦) | FoV (◦) | AUC (%) | URL |
| --- | --- | --- | --- | --- | --- | --- |
**Official Implementation**
|CTRL-C | 1.80 | 1.58 | 0.66 | 3.59 | 87.29 | 
|MSCC| 1.72 | 1.50 | 0.62 | 3.21 |
**Ours** 
|CTRL-C | 1.71 | 1.52 | 0.57 | 3.38 | 87.16 | 
|MSCC| 1.75 | 1.56 | 0.58 | 3.04 | 87.63 |
|SOFI| 1.64 | 1.51 | 0.54 | 3.09 | 87.87 | [!link](https://www.dropbox.com/scl/fi/1dwdn9sepyyj818ri5ml9/sofi.pth?rlkey=zwldsnj0vk7tb8px4hid2xwf8&st=s7uq8s2n&dl=0)

### Holicity Dataset (Testing)
|Model| Up Dir (◦) | Pitch (◦) | Roll (◦) | FoV (◦) | AUC (%) |
| --- | --- | --- | --- | --- | --- |
**Ours** 
|CTRL-C | 2.66 | 2.26 | 1.09 | 3.38 | 72.31 | 
|MSCC| 2.28 | 1.87 | 1.08 | 1.08 | 77.43 |
|SOFI| 2.23 | 1.75 | 1.16 | 1.16 | 82.96 |

### Horizon Line in the Wild  Dataset (Testing)
|Model| AUC (%) |
| --- | --- |
**Ours** 
|CTRL-C | 46.37 | 
|MSCC| 47.28 |
|SOFI| 49.69 |


## Installation


1. Clone this repository.
   ```sh
   git clone https://github.com/SebastianJanampa/SOFI.git
   cd SOFI
   ```

2. Install Pytorch and torchvision

   Follow the instructions on https://pytorch.org/get-started/locally/.
   ```sh
   # an example:
   conda install -c pytorch pytorch torchvision
   ```

3. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```
      
4. Compiling CUDA operators
   ```sh
   cd models/sofi/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```

## Dataset

   ```sh
   mkdir dataset data_csv
   ```
* Please download [Google Street View dataset](https://drive.google.com/file/d/1o_831g-3NDnhR94MEwDS2MFvAwpGmVXN/view?usp=share_link), [Horizon Line in the Wild (HLW)](https://mvrl.cse.wustl.edu/datasets/hlw/) , and [Holicity](https://holicity.io/) datasets and organize them as following:
```
SOFI/
├── data/
│    ├── google_street_view_191210
│    ├── hlw
│    └── holicity
│    
└── data_csv/
     ├── gsv_train_20210313.csv
     ├── gsv_val_20210313.csv
     ├── gsv_test_20210313.csv
     ├── hlw_test.csv
     ├── holicity-test-split.csv
     └── gsv_train_20210313
```
For Holicity, you need to download the files: image, camera, vanishing points.

* Then, run the following command
  ```sh
  python scripts/dataset/extract_segments.py
  ```

## RUN 

1. Training
```sh
bash scripts/train/sofi.sh 
```

2. Testing
```sh
bash scripts/train/sofi.sh dataset
```
Supported datasets: gsv, hlw and holicity

3. Compute metrics
```sh
bash results.py --dataset dataset
```

## Citation
If you use this code for your research, please cite our paper:

```sh
@InProceedings{Janampa_BMVC2024,
    Title     = {{SOFI: Multi-Scale Deformable Transformer for Camera Calibration with Enhanced Line Queries}},
    Author    = {Sebastian Janampa Student and Marios Pattichis},    
    Booktitle = {35th British Machine Vision Conference 2025, {BMVC} 2025, Glasgow, UK, November 25-28, 2024},
    Year      = {2024},
}
   ```

