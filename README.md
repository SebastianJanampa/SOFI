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
* [Google Street View dataset](https://drive.google.com/file/d/1o_831g-3NDnhR94MEwDS2MFvAwpGmVXN/view?usp=share_link)
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
bash scripts/train/#model_name
```
We support training for SOFI, CTRL-C and MSCC


