# SOFI
 ========
 
This repository contains the official code and pretrained models for **SOFI** (multi-**S**cale def**O**rmable trans**F**ormer for camera calibrat**I**on with enhanced line queries)


## Installation


1. Clone this repository.
   ```sh
   git clone https://github.com/SebastianJanampa/SOFI.git
   cd SOFI
   ```

2. Install Pytorch and torchvision

   Follow the instruction on https://pytorch.org/get-started/locally/.
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
   cd models/dino/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```

## Training Dataset

   ```sh
   mkdir dataset data_csv
   ```
* Please download [Google Street View dataset](https://drive.google.com/file/d/1o_831g-3NDnhR94MEwDS2MFvAwpGmVXN/view?usp=share_link), [Horizon Line in the Wild (HLW)](https://mvrl.cse.wustl.edu/datasets/hlw/) datasets and organize them as following:
```
SOFI/
├── data/
│    ├── google_street_view_191210
│    ├── hlw
│    └── holicity
│    
└── data_csv/
     ├── gsv_train_20210313
     └── gsv_test_20210313
```

## RUN 

1. Training
```sh
bash scripts/train/model_name.sh 
```
We support training for SOFI, CTRL-C and MSCC

2. Testing
```sh
bash scripts/train/model_name.sh dataset
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

