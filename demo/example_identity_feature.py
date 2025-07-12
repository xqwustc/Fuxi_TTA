import os
import sys
import logging
import numpy as np
import torch
import gc

# Set up proper path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
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
    experiment_id = 'DCN_frappe'
    model_name = experiment_id.split('_')[0]  # Extract model name from experiment_id
    
    # Define relative config directory path
    relative_config_dir = f'model_zoo/{model_name}/{model_name}_torch/config/'
    
    # Combine parent directory with relative config path to get absolute path
    config_dir = os.path.join(root_dir, relative_config_dir)
    print(f"Using config directory: {config_dir}")
    
    # Load params from config file
    params = load_config(config_dir, experiment_id)
    
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
    
    # Print original features
    logging.info("Original features: " + print_to_json(feature_map.features))
    
    # Dynamically import the model class based on experiment_id
    logging.info(f"Dynamically loading model: {model_name}")
    
    try:
        # Import the module containing the model
        model_module_path = f"model_zoo.{model_name}.{model_name}_torch.src"
        logging.info(f"Importing module: {model_module_path}")
        model_module = importlib.import_module(model_module_path)
        
        # Get the model class
        model_class = getattr(model_module, model_name)
        logging.info(f"Model class loaded: {model_class.__name__}")
        
        # Initialize model
        model = model_class(feature_map, **params)
        model.count_parameters()
        
        # Check if the model's embedding layer has the identity feature
        has_identity = hasattr(model.embedding_layer.embedding_layer, 'has_identity_feature') and \
                      model.embedding_layer.embedding_layer.has_identity_feature
        
        logging.info(f"Model has identity feature '#': {has_identity}")
        
        # Load training data
        train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
        
        # Train model
        logging.info("Start training model...")
        model.fit(train_gen, validation_data=valid_gen, **params)
        
        # Load test data
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        
        # Evaluate model
        logging.info("Evaluating model...")
        test_result = model.evaluate(test_gen)
        
        # Print test results
        logging.info("Test results: " + print_to_json(test_result))
        
        # Analyze feature importance through cross features with identity
        logging.info("Identity feature allows the model to capture individual feature importance")
        logging.info("When a feature interacts with the identity feature '#', it represents the importance of that feature itself")
        
    except (ImportError, AttributeError) as e:
        logging.error(f"Failed to import or initialize model {model_name}: {e}")
        sys.exit(1) 