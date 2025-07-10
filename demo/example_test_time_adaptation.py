import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import gc

sys.path.append('../')
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.preprocess import FeatureProcessor
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.preprocess.build_dataset import build_dataset

# Import model
from model_zoo.DeepFM.DeepFM_torch.src import DeepFM

if __name__ == '__main__':
    # Set parameters
    params = {
        "model": "DeepFM",
        "dataset_id": "tiny_parquet",
        "loss": "binary_crossentropy",
        "metrics": ["logloss", "AUC"],
        "task": "binary_classification",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "embedding_dim": 10,
        "net_dropout": 0,
        "batch_norm": False,
        "batch_size": 128,
        "epochs": 1,
        "seed": 2023,
        "gpu": 0 if torch.cuda.is_available() else -1,
        "data_format": "parquet",
        "data_root": "../data/",
        "model_root": "../checkpoints/",
        "verbose": 1,
        "save_best_only": True,
        "monitor": "AUC",
        "monitor_mode": "max",
        
        # Test time adaptation parameters
        "enable_adaptation": True,
        "adaptation_lr": 1e-4,
        "adaptation_steps": 5,
        "adaptation_method": "entropy_minimization"  # ["entropy_minimization", "self_training", "tent"]
    }
    
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
    
    # Initialize model
    model = DeepFM(feature_map, **params)
    model.count_parameters()
    
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