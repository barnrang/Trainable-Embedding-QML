import yaml
import argparse

parser = argparse.ArgumentParser(description='Run an experiment')
parser.add_argument('--exp_name', dest='exp_name', type=str)
args = parser.parse_args()
exp_name = args.exp_name

#read yaml file
with open('configs/experiment_config.yaml') as file:
    exp_configs = yaml.safe_load(file)

print(exp_configs[exp_name])

