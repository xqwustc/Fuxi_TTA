# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


import torch.nn as nn
import numpy as np
import torch
import os, sys
import logging
from fuxictr.pytorch.layers import FeatureEmbeddingDict
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from fuxictr.utils import Monitor, not_in_whitelist
from tqdm import tqdm


class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 task="binary_classification", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 early_stop_patience=2, 
                 eval_steps=None, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True,
                 # Test time adaptation parameters
                 enable_adaptation=False,
                 adaptation_lr=1e-4,
                 adaptation_steps=10,
                 adaptation_method="entropy_minimization",  # ["entropy_minimization", "self_training", "tent"]
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]
        
        # Test time adaptation parameters
        self.enable_adaptation = enable_adaptation
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.adaptation_method = adaptation_method
        self.adaptation_optimizer = None
        
        # Initialize adaptation optimizer if needed
        if self.enable_adaptation:
            logging.info("Test time adaptation enabled with method: {}".format(self.adaptation_method))
            logging.info("Adaptation learning rate: {}, steps: {}".format(self.adaptation_lr, self.adaptation_steps))
            
    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss)

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            emb_params = set()
            for m_name, module in self.named_modules():
                if type(module) == FeatureEmbeddingDict:
                    for p_name, param in module.named_parameters():
                        if param.requires_grad:
                            emb_params.add(".".join([m_name, p_name]))
                            for emb_p, emb_lambda in emb_reg:
                                reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name not in emb_params:
                        for net_p, net_lambda in net_reg:
                            reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def add_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        return loss

    def compute_loss(self, return_dict, y_true):
        loss = self.add_loss(return_dict, y_true) + self.regularization_loss()
        return loss

    def reset_parameters(self):
        def default_reset_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def custom_reset_params(m):
            # initialize layers with customized init_weights()
            if hasattr(m, 'init_weights'):
                m.init_weights()
        self.apply(default_reset_params)
        self.apply(custom_reset_params)

    def get_inputs(self, inputs, feature_source=None):
        X_dict = dict()
        for feature in inputs.keys():
            if feature in self.feature_map.labels:
                continue
            spec = self.feature_map.features[feature]
            if spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(spec["source"], feature_source):
                continue
            X_dict[feature] = inputs[feature].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[self.feature_map.group_id]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr
           
    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch
        
        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        if return_dict.get("add_loss", None):
            loss += return_dict["add_loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

    def predict(self, data_generator):
        if self.enable_adaptation:
            return self.predict_with_adaptation(data_generator)
        else:
            # Original predict method
            self.eval()  # set to evaluation mode
            with torch.no_grad():
                y_pred = []
                if self._verbose > 0:
                    data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
                for batch_data in data_generator:
                    return_dict = self.forward(batch_data)
                    y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_pred = np.array(y_pred, np.float64)
                return y_pred
    
    def predict_with_adaptation(self, data_generator):
        """
        Predict with test-time adaptation
        """
        # Initialize adaptation optimizer if not already done
        if self.adaptation_optimizer is None:
            self.adaptation_optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.adaptation_lr
            )
        
        # Set to evaluation mode but enable gradients for adaptation
        self.eval()
        
        y_pred = []
        if self._verbose > 0:
            data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
        
        for batch_data in data_generator:
            # Perform test-time adaptation on this batch
            self._adapt_on_batch(batch_data)
            
            # After adaptation, make prediction
            with torch.no_grad():
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
        
        y_pred = np.array(y_pred, np.float64)
        return y_pred
    
    def _adapt_on_batch(self, batch_data):
        """
        Perform adaptation on a single batch
        """
        # Enable gradients for adaptation
        for param in self.parameters():
            param.requires_grad = True
            
        # Perform adaptation steps
        for _ in range(self.adaptation_steps):
            self.adaptation_optimizer.zero_grad()
            
            # Forward pass
            return_dict = self.forward(batch_data)
            y_pred = return_dict["y_pred"]
            
            # Compute adaptation loss based on the selected method
            if self.adaptation_method == "entropy_minimization":
                # Entropy minimization: minimize prediction uncertainty
                loss = self._entropy_loss(y_pred)
            elif self.adaptation_method == "self_training":
                # Self-training: generate pseudo-labels and train on them
                pseudo_labels = (y_pred > 0.5).float().detach()
                loss = nn.BCELoss()(y_pred, pseudo_labels)
            elif self.adaptation_method == "tent":
                # Tent: normalize and minimize entropy (Test Entropy minimization)
                # Described in: https://arxiv.org/abs/2006.10726
                loss = self._entropy_loss(y_pred)
            else:
                raise NotImplementedError(f"Adaptation method {self.adaptation_method} not implemented")
            
            # Backward pass and optimization
            loss.backward()
            self.adaptation_optimizer.step()
        
        # Disable gradients after adaptation
        for param in self.parameters():
            param.requires_grad = False
    
    def _entropy_loss(self, y_pred):
        """
        Compute entropy loss for binary predictions
        """
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-12
        entropy = -y_pred * torch.log(y_pred + eps) - (1 - y_pred) * torch.log(1 - y_pred + eps)
        return entropy.mean()
    
    def evaluate(self, data_generator, metrics=None):
        if self.enable_adaptation:
            return self.evaluate_with_adaptation(data_generator, metrics)
        else:
            # Original evaluate method
            self.eval()  # set to evaluation mode
            with torch.no_grad():
                y_pred = []
                y_true = []
                group_id = []
                if self._verbose > 0:
                    data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
                for batch_data in data_generator:
                    return_dict = self.forward(batch_data)
                    y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                    y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                    if self.feature_map.group_id is not None:
                        group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
                y_pred = np.array(y_pred, np.float64)
                y_true = np.array(y_true, np.float64)
                group_id = np.array(group_id) if len(group_id) > 0 else None
                if metrics is not None:
                    val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
                else:
                    val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
                logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                return val_logs
    
    def evaluate_with_adaptation(self, data_generator, metrics=None):
        """
        Evaluate with test-time adaptation
        """
        # Initialize adaptation optimizer if not already done
        if self.adaptation_optimizer is None:
            self.adaptation_optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.adaptation_lr
            )
            
        self.eval()  # set to evaluation mode
        
        with torch.set_grad_enabled(True):  # Enable gradients for adaptation
            y_pred = []
            y_true = []
            group_id = []
            
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
                
            for batch_data in data_generator:
                # Perform test-time adaptation on this batch
                self._adapt_on_batch(batch_data)
                
                # After adaptation, make prediction
                with torch.no_grad():
                    return_dict = self.forward(batch_data)
                    y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                    y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                    if self.feature_map.group_id is not None:
                        group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
                    
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
                
            logging.info('[Metrics with adaptation] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))

