# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
print(f"Current sys.path: {sys.path}")
import logging
import sys
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src
import gc
import argparse
import os
from pathlib import Path


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DCN_frappe', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    # Add Test Time Adaptation related arguments
    parser.add_argument('--enable_tta', default=False, type=bool, help='Enable Test Time Adaptation')
    parser.add_argument('--tta_lr', type=float, default=1e-4, help='Learning rate for Test Time Adaptation')
    parser.add_argument('--tta_steps', type=int, default=5, help='Number of adaptation steps per batch')
    parser.add_argument('--tta_method', type=str, default='entropy_minimization', 
                       choices=['entropy_minimization', 'self_training', 'tent'],
                       help='Method for Test Time Adaptation')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    
    # Add Test Time Adaptation parameters to params
    if args['enable_tta']:
        params['adaptation_lr'] = args['tta_lr']
        params['adaptation_steps'] = args['tta_steps']
        params['adaptation_method'] = args['tta_method']
        
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    test_result = {}
    if params["test_data"]:
        # Evaluate without test time adaptation
        logging.info('******** Test evaluation without adaptation ********')
        model.enable_adaptation = False
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result_no_adaptation = model.evaluate(test_gen)
        test_result = test_result_no_adaptation  # Keep original result for backward compatibility
        
        # Evaluate with test time adaptation if enabled
        if args['enable_tta']:
            logging.info('******** Test evaluation with adaptation ********')
            model.enable_adaptation = True
            test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
            test_result_with_adaptation = model.evaluate(test_gen)
            
            # Compare results
            logging.info("******** Results comparison ********")
            for metric in params["metrics"]:
                if metric in test_result_no_adaptation and metric in test_result_with_adaptation:
                    improvement = test_result_with_adaptation[metric] - test_result_no_adaptation[metric]
                    logging.info(f"{metric}: No adaptation = {test_result_no_adaptation[metric]:.6f}, "
                                f"With adaptation = {test_result_with_adaptation[metric]:.6f}, "
                                f"Improvement = {improvement:.6f}")
            
            # Update test_result to include both results
            for metric in params["metrics"]:
                if metric in test_result_with_adaptation:
                    test_result[f"{metric}_with_adaptation"] = test_result_with_adaptation[metric]
    
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))
