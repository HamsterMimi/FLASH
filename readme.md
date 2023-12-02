

# FLASH: A Fast and Unified Zero-Shot NAS with Greedy Strategy

## Installation

```
Python >= 3.6
PyTorch >= 2.0.0
nas-bench-201
```

## Preparation

1. Download three datasets (CIFAR-10, CIFAR-100, ImageNet16-120) from [Google Drive](https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4),  place them into the directory `./_dataset`
2. Download the [`data` directory](https://drive.google.com/drive/folders/18Eia6YuTE5tn5Lis_43h30HYpnF9Ynqf?usp=sharing) and save it to the root folder of this repo. 
3. Download the benchmark files of NAS-Bench-201 from [Google Drive](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view) , put them into the directory `./data`

 ```bash
cd zero-cost-nas
pip install .
cd ..
 ```

## Usage/Examples

### Correlation Experiment

```bash
cd correlation
python NAS_Bench_101.py
python NAS_Bench_201.py
```




### Experiments on NAS-Bench-201

1. Run Zero-Cost-PT with appointed zero-cost proxy:

```bash
cd exp_scripts
bash zerocostpt_nb201_pipline.sh --metric [metric] --batch_size [batch_size] --seed [seed]
```

You can choice metric from `['snip', 'fisher', 'synflow', 'grad_norm', 'grasp', 'jacob_cov','tenas', 'zico', 'meco'] `

### Experiments on DARTS-CNN Space

```bash
cd exp_scripts
bash zerocostpt_darts_pipline.sh --metric [metric] --batch_size [batch_size] --seed [seed]
```


## Reference

Our code is based on [Zero-Cost-PT](https://github.com/zerocostptnas/zerocost_operation_score) and [Zero-Cost-NAS](https://github.com/SamsungLabs/zero-cost-nas).
