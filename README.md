# Trainable-Embedding-QML
Implementation of Quantum Random Access Coding (QRAC) and Trainable Embedding (TE)

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