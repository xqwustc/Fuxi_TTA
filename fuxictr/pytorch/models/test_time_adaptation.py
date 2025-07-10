import os
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import sys
from ...pytorch.models.rank_model import BaseModel

class TestTimeAdaptation(BaseModel):
    """
    Test Time Adaptation for CTR models
    This class wraps around a BaseModel and provides test time adaptation capabilities.
    """
    def __init__(self, 
                 base_model,
                 adaptation_lr=1e-4,
                 adaptation_steps=10,
                 adaptation_batch_size=128,
                 adaptation_loss="binary_crossentropy",
                 adaptation_method="entropy_minimization",  # ["entropy_minimization", "self_training", "tent"]
                 **kwargs):
        """
        Initialize the TestTimeAdaptation wrapper
        
        Args:
            base_model: The pre-trained base model to adapt
            adaptation_lr: Learning rate for adaptation
            adaptation_steps: Number of adaptation steps per batch
            adaptation_batch_size: Batch size for adaptation
            adaptation_loss: Loss function for adaptation
            adaptation_method: Method for adaptation (entropy_minimization, self_training, tent)
        """
        # We don't call super().__init__() because we're wrapping an existing model
        self.base_model = base_model
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.adaptation_batch_size = adaptation_batch_size
        self.adaptation_loss = adaptation_loss
        self.adaptation_method = adaptation_method
        
        # Copy attributes from base_model
        self.device = base_model.device
        self.feature_map = base_model.feature_map
        self.model_id = base_model.model_id + "_tta"
        self.output_activation = base_model.output_activation
        self.checkpoint = base_model.checkpoint
        self._verbose = base_model._verbose
        
        # Initialize optimizer for adaptation
        self.adaptation_optimizer = torch.optim.Adam(
            self.base_model.parameters(), 
            lr=self.adaptation_lr
        )
        
        # Set model to eval mode by default
        self.base_model.eval()
        
    def forward(self, inputs):
        """
        Forward pass through the base model
        """
        return self.base_model.forward(inputs)
    
    def get_labels(self, inputs):
        """
        Get labels from inputs
        """
        return self.base_model.get_labels(inputs)
    
    def get_inputs(self, inputs):
        """
        Get inputs from batch data
        """
        return self.base_model.get_inputs(inputs)
    
    def predict(self, data_generator):
        """
        Predict with test-time adaptation
        """
        # Set to evaluation mode but enable gradients for adaptation
        self.base_model.eval()
        
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
        for param in self.base_model.parameters():
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
        for param in self.base_model.parameters():
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
        """
        Evaluate with test-time adaptation
        """
        self.base_model.eval()  # set to evaluation mode
        
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
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
                    
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            
            if metrics is not None:
                val_logs = self.base_model.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.base_model.evaluate_metrics(y_true, y_pred, self.base_model.validation_metrics, group_id)
                
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs
    
    def save_weights(self, checkpoint):
        """
        Save model weights
        """
        self.base_model.save_weights(checkpoint)
    
    def load_weights(self, checkpoint):
        """
        Load model weights
        """
        self.base_model.load_weights(checkpoint) 