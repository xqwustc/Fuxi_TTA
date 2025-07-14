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
import importlib


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DCN_frappe', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    parser.add_argument('--has_identity_feature', type=bool, default=False, help='Whether to use identity feature.')
    parser.add_argument('--mode', type=str, default='recon', help='The mode to run the model.')
    parser.add_argument('--checkpoint', type=str, default="default", help='The checkpoint to load.')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    params['has_identity_feature'] = args['has_identity_feature']
    params['mode'] = args['mode']
        
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    if params.get('spe_processor'):
        module_name = f"fuxictr.datasets.{params['spe_processor']}"
        fp_module = importlib.import_module(module_name)
        assert hasattr(fp_module, 'FeatureProcessor')
        FeatureProcessor = getattr(fp_module, 'FeatureProcessor')
    else:
        from fuxictr.preprocess import FeatureProcessor

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

    # model.embedding_layer.plot_embedding_each_field()
    
    # check point in model.checkpoint
    if args.get('checkpoint') is None:
        train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
        model.fit(train_gen, validation_data=valid_gen, **params)

        logging.info('****** Validation evaluation ******')
        valid_result = model.evaluate(valid_gen)
        del train_gen, valid_gen
        gc.collect()
    elif args.get('checkpoint') == "default":
        model.load_weights(model.checkpoint)
    else:
        # concat the path
        checkpoint_path = os.path.join(model.checkpoint, args['checkpoint'])
        model.load_weights(checkpoint_path)

    model.embedding_layer.cluster_embedding()
    model.embedding_layer.plot_embedding_each_field()
    
    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)
    

    # 如果 valid_result 这个符号存在
    if 'valid_result' in locals():
        result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
        with open(result_filename, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                        ' '.join(sys.argv), experiment_id, params['dataset_id'],
                        "N.A.", print_to_list(valid_result), print_to_list(test_result)))
