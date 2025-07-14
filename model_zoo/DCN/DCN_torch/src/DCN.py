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

import logging
import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNet, ReconEmbedding


class DCN(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="DCN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DCN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)

        self.mode = kwargs.get('mode', None)
        self.loss_on = kwargs.get('loss_on', 'embeds')
        # assert self.loss_on in ['batch', 'embeds']
        assert self.loss_on in ['batch', 'embeds']

        # --- update for tayfs retrain start ---
        if self.mode and self.mode in ['recon', 'pre']: # infer with the reconstructed embedding
            self.embedding_layer = ReconEmbedding(feature_map, embedding_dim, mode=self.mode)
        else: # use the original embedding
            self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
                                                            
        ## Update
        input_dim = feature_map.sum_emb_out_dim()
        input_dim += kwargs.get("has_identity_feature", False)*embedding_dim
        ## Update end

        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, num_cross_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1) # [cross_part, dnn_part] -> logit
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        
        # 1. 计算批次内所有样本的“平均字段嵌入” (Mean Field Embedding)
        #    结果形状为 (field_num, embedding_dim)
        if self.loss_on == 'batch':
            mean_field_emb = torch.mean(feature_emb, dim=0)

            # 2. 计算每个样本到“平均字段嵌入”的整体平方距离
            #    diff 的形状是 (batch_size, field_num, embedding_dim)
            diff = feature_emb - mean_field_emb
            #    per_sample_dist_sq 的形状是 (batch_size,)
            #    它现在扮演了新 "dist" 的角色，但它是一个向量而非矩阵
            per_sample_dist_sq = diff.pow(2).sum(dim=[1, 2])

            # 3. 将这个新的距离向量应用到 exp 损失函数中
            t = 0.1 # 温度参数
            #    torch.exp 会逐元素地应用在 per_sample_dist_sq 向量上
            #    然后 .mean() 会计算这 batch_size 个 exp 值的平均值
            center_based_uniformity_loss = torch.exp(-per_sample_dist_sq / t).mean()
        elif self.loss_on == 'embeds':
            center_based_uniformity_loss = 0
            tot_num = 0

            for i, (feature, embedding_layer) in enumerate(self.embedding_layer.embedding_layer.embedding_layers.items()):
                mean_field_emb = torch.mean(embedding_layer.weight, dim=0)
                diff = embedding_layer.weight - mean_field_emb
                per_sample_dist_sq = diff.pow(2).sum(dim=1)
                t = 0.1 # 温度参数
                feature_uniformity_loss += torch.exp(-per_sample_dist_sq / t).sum()
                tot_num += feature_uniformity_loss.shape[0]
            center_based_uniformity_loss = feature_uniformity_loss / tot_num
        else:
            raise ValueError(f"Invalid loss_on: {self.loss_on}")
                
        
        # 后续网络部分
        feature_emb = feature_emb.view(feature_emb.size(0), -1)

        cross_out = self.crossnet(feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        
        # 在返回的字典中使用新的loss
        return_dict = {"y_pred": y_pred, "add_loss": 3 * center_based_uniformity_loss}
        return return_dict
        
        return_dict = {"y_pred": y_pred, "uniformity_loss": uniformity_loss}
        return return_dict

