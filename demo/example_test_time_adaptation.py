import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import gc

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
print(f"Current sys.path: {sys.path}")

from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.preprocess import FeatureProcessor
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.preprocess.build_dataset import build_dataset

# Import model dynamically based on experiment_id
import importlib

if __name__ == '__main__':
    # Load configuration from yaml file
    config_dir = './model_zoo/DCN/DCN_torch/config/'
    config_dir = os.path.abspath(config_dir)
    experiment_id = 'DCN_frappe'
    
    # Load params from config file
    params = load_config(config_dir, experiment_id)
    
    # Add test time adaptation parameters
    params.update({
        "enable_adaptation": True,
        "adaptation_lr": 1e-4,
        "adaptation_steps": 5,
        "adaptation_method": "entropy_minimization"  # ["entropy_minimization", "self_training", "tent"]
    })
    
    # Set GPU
    params['gpu'] = 0 if torch.cuda.is_available() else -1
    
    # Set logger
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])
    
    # Load data
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    
    # Build feature map
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    # Dynamically import the model class based on experiment_id
    model_name = experiment_id.split('_')[0]  # Extract model name from experiment_id
    logging.info(f"Dynamically loading model: {model_name}")
    
    try:
        # Import the module containing the model
        model_module = importlib.import_module(f"model_zoo.{model_name}.{model_name}_torch.src")
        # Get the model class
        model_class = getattr(model_module, model_name)
        # Initialize model
        model = model_class(feature_map, **params)
        model.count_parameters()
    except (ImportError, AttributeError) as e:
        logging.error(f"Failed to import model {model_name}: {e}")
        sys.exit(1)
    
    # Load training data
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    
    # Train model
    logging.info("Start training model...")
    model.fit(train_gen, validation_data=valid_gen, **params)
    
    # Load test data
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
    
    # Evaluate without test time adaptation
    logging.info("Evaluating without test time adaptation...")
    model.enable_adaptation = False
    test_result_no_adaptation = model.evaluate(test_gen)
    
    # Reload test data
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
    
    # Evaluate with test time adaptation
    logging.info("Evaluating with test time adaptation...")
    model.enable_adaptation = True
    test_result_with_adaptation = model.evaluate(test_gen)
    
    # Compare results
    logging.info("Results comparison:")
    for metric in params["metrics"]:
        if metric in test_result_no_adaptation and metric in test_result_with_adaptation:
            improvement = test_result_with_adaptation[metric] - test_result_no_adaptation[metric]
            logging.info(f"{metric}: No adaptation = {test_result_no_adaptation[metric]:.6f}, "
                        f"With adaptation = {test_result_with_adaptation[metric]:.6f}, "
                        f"Improvement = {improvement:.6f}") 