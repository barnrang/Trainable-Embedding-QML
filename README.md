# Trainable-Embedding-QML
Implementation of Quantum Random Access Coding (QRAC) and Trainable Embedding (TE) based on our preprinted paper [here](https://arxiv.org/abs/2106.09415)

# Run experiment
To run the experiment provided in the paper, please take a look at `configs/expriment_config.yaml`.
In `experiment_config.yaml`, the structure is 
```
<dataset 1 (exp_name)>
    - <method 1 (sub_exp_name)>
        - param 1
        - param 2
    - <method 2>
        - param 1
        - param 2

<dataset 2>
...
```
which there are 2 levels, `exp_name` and `sub_exp_name`. To run all expriment in the dataset, please run the python script `run_exp.py` with the following argument
```bash
python run_exp --exp_name <dataset_name>
```
For example, to run all breast cancer dataset experiments
```bash
python run_exp --exp_name bc
```

If you want to run a single experiment, please specify the `sub_exp_name`. For example, we run the QRAC method on breast cancer. 
```bash
python run_exp.py --exp_name bc --sub_exp_name qrac
```

# Run custom experiment
Ther are 4 files for different experiment
```
parity_check.py - Parity function problem
bc.py - Breast cancer
titanic.py - Titanic Survival
mnist.py - MNIST handwritten digit
```
The parameter for each experiment can be listed using `--help`. 

# Run onreal device (NEW!)
6 bits parity check
```
run_exp.py --exp_name pc_6 --sub_exp_name te_pc_real_device
```
Please update detail at this [line](https://github.com/barnrang/Trainable-Embedding-QML/blob/7ab2f9c6f0d39daecf4237c077fd9c3c451833cc/parity_check.py#L110)


100 epochs Titanic Survival
```
run_exp.py --exp_name ts --sub_exp_name te_reg_ts_real
```
Please update detail at this [line](https://github.com/barnrang/Trainable-Embedding-QML/blob/7ab2f9c6f0d39daecf4237c077fd9c3c451833cc/titanic.py#L138)

You can change the parameters in `config/experiment_config.yaml`.

# Citation
```
@article{Thumwanit2021TrainableDF,
  title={Trainable Discrete Feature Embeddings for Variational Quantum Classifier},
  author={Napat Thumwanit and Chayaphol Lortararprasert and Hiroshi Yano and Raymond H. Putra},
  journal={ArXiv},
  year={2021},
  volume={abs/2106.09415}
}
```
