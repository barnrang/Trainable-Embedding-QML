from math import exp
import pickle
import yaml
import argparse

from bc import run_exp as bc_run_exp
from titanic import run_exp as ts_run_exp
from parity_check import run_exp as pc_run_exp
from mnist import run_exp as mnist_run_exp

parser = argparse.ArgumentParser(description='Run an experiment')
parser.add_argument('--exp_name', dest='exp_name', type=str)
parser.add_argument('--sub_exp_name', dest="sub_exp_name", type=str, default=None)
args = parser.parse_args()
exp_name = args.exp_name
sub_exp_name = args.sub_exp_name

run_exp_dataset = {
    'bc': bc_run_exp,
    'ts': ts_run_exp,
    'pc_3': pc_run_exp,
    'pc_6': pc_run_exp,
    'pc_9': pc_run_exp,
    'mnist': mnist_run_exp
}

def run_single_exp(exp_configs, exp_name, sub_exp_name):
    exp_config = exp_configs[exp_name][sub_exp_name]
    if isinstance(exp_config['seed'], list):
        seeds = exp_config['seed']
        exp_config.pop('seed', None)
        result = None
        n = len(seeds)
        for seed in seeds:
            tmp_result = run_exp_dataset[exp_name](seed=seed, **exp_config)
            if result is None:
                result = tmp_result
            else:
                for key in tmp_result:
                    result[key] += tmp_result[key]

        # Take mean of all seeds
        for key in result:
            result[key] /= n
    else:
        result = run_exp_dataset[exp_name](**exp_config)

    return result

#read yaml file
with open('configs/experiment_config.yaml') as file:
    exp_configs = yaml.safe_load(file)

print(exp_configs)

# Run all experiments for a dataset
if sub_exp_name is None:
    results = {}
    exp_dataset = exp_configs[exp_name]
    print(exp_dataset)
    for sub_exp in exp_dataset:
        results[sub_exp] = run_single_exp(exp_configs, exp_name, sub_exp)
else:
    results = run_single_exp(exp_configs, exp_name, sub_exp_name)

if sub_exp_name is None:
    with open(f"results/{exp_name}_exp.pkl", 'wb') as f:
        pickle.dump(results, f)
else:
    with open(f"results/{exp_name}_{sub_exp_name}_exp.pkl", 'wb') as f:
        pickle.dump(results, f)
print(results)

