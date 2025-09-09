import argparse

from train import rl_logging
from train.rl_noisy_hf import train, evaluate
from train.utils import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Noise RL using a YAML configuration."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the YAML configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default='test_run',
        help='Name of the experiment'
    )
    args = parser.parse_args()
    rl_logging.setup_logging(args.run_name)
    config = load_config(args.config)
    if config['run_type'] == 'train':
        train(config, args.run_name)
    elif config['run_type'] == 'eval':
        evaluate(config, args.run_name)
    else:
        raise ValueError('Only train or eval run_types supported')
