import torch
from torch import nn
import h5py
import os
import numpy as np
from collections import OrderedDict
from fuxictr.pytorch.torch_utils import get_initializer
from fuxictr.pytorch import layers
import random
import pandas as pd
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 确保你已经安装了 umap-learn 和 hdbscan
# pip install umap-learn hdbscan
import umap
import hdbscan


class ReconEmbedding(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 cross_num=0,
                 mode='pre', # in [pre, recon]
                 paths=None):
        super(ReconEmbedding, self).__init__()
        self.embedding_layer = FeatureEmbeddingDict(feature_map,
                                                    embedding_dim,
                                                    embedding_initializer=embedding_initializer,
                                                    required_feature_columns=required_feature_columns,
                                                    not_required_feature_columns=not_required_feature_columns,
                                                    use_pretrain=use_pretrain,
                                                    use_sharing=use_sharing,
                                                    cross_num=cross_num,
                                                    mode=mode,
                                                    paths=paths)

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb

    def plot_embedding_each_field(self):
        self.embedding_layer.plot_embedding_each_field(self.embedding_layer.embedding_layers)

    def cluster_embedding(self):
        self.embedding_layer.cluster_embedding()

class FeatureEmbeddingDict(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 cross_num=0,
                 method='gncfs',
                 mode='weight',
                 paths=None):
        super(FeatureEmbeddingDict, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.use_pretrain = use_pretrain
        self.embedding_initializer = embedding_initializer
        self.embedding_layers = nn.ModuleDict()
        self.feature_encoders = nn.ModuleDict()
        for feature, feature_spec in self._feature_map.features.items():
            if self.is_required(feature):
                if not (use_pretrain and use_sharing) and embedding_dim == 1:
                    feat_emb_dim = 1  # in case for LR
                    if feature_spec["type"] == "sequence":
                        self.feature_encoders[feature] = layers.MaskedSumPooling()
                else:
                    feat_emb_dim = feature_spec.get("embedding_dim", embedding_dim)
                    if feature_spec.get("feature_encoder", None):
                        self.feature_encoders[feature] = self.get_feature_encoder(feature_spec["feature_encoder"])

                # Set embedding_layer according to share_embedding
                if use_sharing and feature_spec.get("share_embedding") in self.embedding_layers:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue

                if feature_spec["type"] == "numeric":
                    self.embedding_layers[feature] = nn.Linear(1, feat_emb_dim, bias=False)
                elif feature_spec["type"] == "categorical":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = nn.Embedding(feature_spec["vocab_size"],
                                                    feat_emb_dim,
                                                    padding_idx=padding_idx)
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix,
                                                                          feature_map,
                                                                          feature,
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
                elif feature_spec["type"] == "sequence":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = nn.Embedding(feature_spec["vocab_size"],
                                                    feat_emb_dim,
                                                    padding_idx=padding_idx)
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix,
                                                                          feature_map,
                                                                          feature,
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
        # --- update for LiteFCS ---
        self.cross_num = cross_num
        if cross_num:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if type(paths) == dict:
                if paths.get('score_path'):
                    assert os.path.exists(paths.get('score_path')), f"score_path {paths.get('score_path')} does not exist"
            else:
                tpath = {}
                tpath['score_path'] = paths
                paths = tpath
            feature_cross_importance = pd.read_csv(paths.get('score_path'))

            # log the top-10 feature crosses
            logging.info(f'Feature crosses importance: {feature_cross_importance}')

            self.feature_cross_choice = []

            feature_cross_importance = feature_cross_importance.sort_values(by='score', ascending=False)
            feature_cross_name = feature_cross_importance["combination"].tolist()
            # # manually add some third and forth order features
            # feature_cross_name = ["user & item & city" , "user & item & city & country"] + feature_cross_name[2:]
            # feature_cross_name = ['user & item', 'user & city', 'city & country', 'daytime & user', 'cost & country', 'country & city']
            # feature_cross_name = ['C14 & C17', 'site_id & app_category', 'device_ip & banner_pos', 'device_ip & banner_pos', 'device_model & device_ip', 'app_domain & site_domain', 'C20 & C14', 'app_id & C20', 'app_domain & app_id', 'site_id & site_domain', 'C20 & C14', 'app_id & app_category', 'C17 & site_id', 'device_id & C1', 'hour & device_conn_type', 'device_conn_type & hour', 'C19 & C21', 'device_id & C1']

            upper_bound = 5000000  # for hash table size
            max_size = 5e20   # for feature used size
            if mode.endswith('hash'):
                upper_bound = 5e6
            self.cross_vocabs = {}

            if mode.startswith('lre'):
                if feature_map.dataset_id == 'ipinyou_x1':
                    upper_bound = upper_bound // 5
                logging.info('Use LRE to select feature combinations')
                # eliminated features
                ele_path = paths.get('ele_path')
                assert os.path.exists(ele_path), f"ele_path {ele_path} does not exist"
                ele_feat = pd.read_csv(ele_path)
                # only get whose AUC > 0
                # ele_feat = ele_feat[ele_feat['AUC'] > 0]
                ele_feat = ele_feat[ele_feat['logloss'] < 0]
                # if os.getenv('MODE') == 'random':
                #     logging.info('Randomly select feature crosses')
                #     upper_bound = upper_bound // 2
                #     feature_cross_name = random.sample(feature_cross_name, len(feature_cross_name))
                #     # feature_cross_name = random.sample(feature_cross_name, len(feature_cross_name))
                # if os.getenv('MODE') == 'autofield':
                #     logging.info('Use autofield-like method to select features')
                #     feature_cross_name = \
                #         feature_cross_name[:1] + random.sample(feature_cross_name[1:45], len(feature_cross_name[1:45])) \
                #         + feature_cross_name[45:]
                #     upper_bound = upper_bound // 1.5
                # if os.getenv('MODE') == 'pfi':
                #     logging.info('Use pfi to select feature crosses')
                #     feature_cross_name = \
                #         feature_cross_name[:1] + random.sample(feature_cross_name[1:20], len(feature_cross_name[1:20])) \
                #         + feature_cross_name[20:]
                #     upper_bound = upper_bound // 1.5
                #     # eliminated features
                #     ele_feat_name = f'feat_taylor/elem_{feature_map.dataset_id}_use.csv'
                #     current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                #     ele_feat = pd.read_csv(os.path.join(current_dir, ele_feat_name))
                # if os.getenv('MODE') == 'lre':
                #     # TODO: change it
                #     pass
                # if os.getenv('MODE') == 'expert':
                #     logging.info('Use expertise to select feature combinations')
                #     # eliminated features
                #     current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                #     ds_name = feature_map.dataset_id.split('_')[0]
                #     importance_filename = f'feat_taylor/taylorfs_score_{ds_name}_expert.csv'
                #     logging.info(f'Reading file {importance_filename}')
                #     feature_cross_importance = pd.read_csv(os.path.join(current_dir, importance_filename))
                #     # feature_cross_importance = feature_cross_importance.sort_values(by='score', ascending=False)
                #     feature_cross_name = feature_cross_importance["combination"].tolist()
                #     logging.info(f'Expert top-20 feature crosses: {feature_cross_name[:20]}')

            # if method.startswith('random'):
            #     logging.info('Randomly select feature crosses')
            #     max_size = 5e6
            #     feature_cross_name = random.sample(feature_cross_name, len(feature_cross_name))


            for feature_cross in feature_cross_name:
                feature_son = feature_cross.split(" & ")
                if len(feature_son) == 1:
                    # Do not add single feature
                    continue

                if 'ele_feat' in locals():
                    if feature_cross in ele_feat['feature_name'].tolist():
                        continue
                if any(f in ['artist_name', 'genre_ids'] for f in feature_son):
                    continue

                tot_vocab_size = 1
                for cur_feature in feature_son:
                    tot_vocab_size *= self._feature_map.features[cur_feature]["vocab_size"]
                # if self._feature_map.features[feature_son[0]]["vocab_size"] * \
                #         self._feature_map.features[feature_son[1]]["vocab_size"] < 100000000:
                if max_size > tot_vocab_size > 0:
                    self.feature_cross_choice.append(feature_cross)
                    if len(self.feature_cross_choice) == cross_num:
                        break


            logging.info(f'Add new crosses:{self.feature_cross_choice}')
            self.cross_emb_dims = 0
            for feature_cross in self.feature_cross_choice:
                feature_son = feature_cross.split(" & ")
                # embedding_matrix = nn.Embedding(self._feature_map.features[feature_son[0]]["vocab_size"] *
                #                                 self._feature_map.features[feature_son[1]]["vocab_size"],
                #                                 feat_emb_dim,
                #                                 padding_idx=padding_idx)
                # self.embedding_layers[feature_cross] = embedding_matrix
                tot_vocab_size = 1
                for cur_feature in feature_son:
                    tot_vocab_size *= self._feature_map.features[cur_feature]["vocab_size"]

                if mode.endswith('hash'):
                    tot_vocab_size = min(tot_vocab_size, upper_bound) + 1
                self.cross_vocabs[feature_cross] = int(tot_vocab_size)

                cur_dim = embedding_dim

                embedding_matrix = nn.Embedding(self.cross_vocabs[feature_cross], cur_dim, padding_idx=padding_idx)
                logging.info(f'Add new cross:{feature_cross} with feat_emb_dim:{cur_dim} and '
                             f'tot_vocab_size:{self.cross_vocabs[feature_cross]}')
                self.cross_emb_dims += cur_dim
                self.embedding_layers[feature_cross] = embedding_matrix


        # --- update for LiteFCS ---
        self.reset_parameters()

    def get_feature_encoder(self, encoder):
        try:
            if type(encoder) == list:
                encoder_list = []
                for enc in encoder:
                    encoder_list.append(eval(enc))
                encoder_layer = nn.Sequential(*encoder_list)
            else:
                encoder_layer = eval(encoder)
            return encoder_layer
        except:
            raise ValueError("feature_encoder={} is not supported.".format(encoder))

    def reset_parameters(self):
        self.embedding_initializer = get_initializer(self.embedding_initializer)
        for k, v in self.embedding_layers.items():
            if self.use_pretrain and k in self._feature_map.features and "pretrained_emb" in self._feature_map.features[
                k]:  # skip pretrained
                continue
            if k in self._feature_map.features and "share_embedding" in self._feature_map.features[
                k] and v.weight.requires_grad == False:
                continue
            if type(v) == nn.Embedding:
                if v.padding_idx is not None:  # using 0 index as padding_idx
                    self.embedding_initializer(v.weight[1:, :])
                else:
                    self.embedding_initializer(v.weight)

    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.features[feature]
        if feature_spec["type"] == "meta":
            return False
        elif self.required_feature_columns and (feature not in self.required_feature_columns):
            return False
        elif self.not_required_feature_columns and (feature in self.not_required_feature_columns):
            return False
        else:
            return True

    def get_pretrained_embedding(self, pretrained_path, feature_name):
        with h5py.File(pretrained_path, 'r') as hf:
            embeddings = hf[feature_name][:]
        return embeddings

    def load_pretrained_embedding(self, embedding_matrix, feature_map, feature_name, freeze=False, padding_idx=None):
        pretrained_path = os.path.join(feature_map.data_dir, feature_map.features[feature_name]["pretrained_emb"])
        embeddings = self.get_pretrained_embedding(pretrained_path, feature_name)
        if padding_idx is not None:
            embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
        assert embeddings.shape[-1] == embedding_matrix.embedding_dim, \
            "{}\'s embedding_dim is not correctly set to match its pretrained_emb shape".format(feature_name)
        embeddings = torch.from_numpy(embeddings).float()
        embedding_matrix.weight = torch.nn.Parameter(embeddings)
        if freeze:
            embedding_matrix.weight.requires_grad = False
        return embedding_matrix

    def dict2tensor(self, embedding_dict, feature_list=[], feature_source=[], feature_type=[], flatten_emb=False):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_list = []
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature_list and feature not in feature_list:
                continue
            # if feature_spec["type"] == "numeric":
            #     continue
            if feature in embedding_dict:
                feature_emb_list.append(embedding_dict[feature])
        # --- update for LiteFCS ---
        if self.cross_num:
            for feature_cross in self.feature_cross_choice:
                feature_emb_list.append(embedding_dict[feature_cross])
        # --- update for LiteFCS ---
        if flatten_emb:
            feature_emb = torch.cat(feature_emb_list, dim=-1)
        else:
            feature_emb = torch.stack(feature_emb_list, dim=1)
        # print(feature_emb)
        # exit()
        return feature_emb

    def forward(self, inputs, feature_source=[], feature_type=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature in self.embedding_layers:
                if feature_spec["type"] == "numeric":
                    # continue
                    inp = inputs[feature].float().view(-1, 1)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.feature_encoders:
                    embeddings = self.feature_encoders[feature](embeddings)
                feature_emb_dict[feature] = embeddings
        # --- update for LiteFCS ---
        if self.cross_num:
            for feature_cross in self.feature_cross_choice:
                feature_son = feature_cross.split(" & ")
                if len(feature_son) == 1:
                    continue
            
                for idx, cur_feature in enumerate(feature_son):
                    if idx == 0:
                        input_cross = inputs[cur_feature].long()
                    else:
                        input_cross = input_cross * self._feature_map.features[cur_feature]["vocab_size"] + \
                                      inputs[cur_feature].long()

                input_cross = input_cross.long()

                if hasattr(self, 'cross_vocabs'):
                    vocab_size = self.cross_vocabs[feature_cross]
                    input_cross = (input_cross % (vocab_size - 1)) + 1

                embeddings = self.embedding_layers[feature_cross](input_cross)
                feature_emb_dict[feature_cross] = embeddings

                if os.getenv('CUR_PERM_FEAT') is not None and feature_cross == os.getenv('CUR_PERM_FEAT'):
                    feature_emb_dict[feature_cross] = torch.mean(embeddings).unsqueeze(0).repeat(
                        embeddings.size(0), 1)
                    # feature_emb_dict[feature_cross] = torch.zeros_like(feature_emb_dict[feature_cross])
                    # logging.info(f'Permute features: {feature_cross} done.')
                else:
                    feature_emb_dict[feature_cross] = embeddings

        return feature_emb_dict

    def plot_embedding_each_field(self, feature_emb_dict):
        # t-SNE plot for each field
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import math
        from datetime import datetime
        
        # 计算子图布局
        n_features = len(feature_emb_dict)
        if n_features == 0:
            return
            
        # 计算行列数
        n_cols = min(3, n_features)  # 最多3列
        n_rows = math.ceil(n_features / n_cols)
        
        # 创建图形和子图
        plt.figure(figsize=(n_cols * 5, n_rows * 4))
        
        # 对每个特征进行t-SNE降维并绘制
        for i, (feature, embedding_layer) in enumerate(feature_emb_dict.items()):
            # 创建子图
            plt.subplot(n_rows, n_cols, i + 1)
            
            # 获取embedding数据
            if isinstance(embedding_layer, nn.Embedding):
                # 对于Embedding层，获取权重
                emb_data = embedding_layer.weight.detach().cpu().numpy()
            elif isinstance(embedding_layer, nn.Linear):
                # 对于Linear层，获取权重
                emb_data = embedding_layer.weight.detach().cpu().numpy().T
            elif isinstance(embedding_layer, torch.Tensor):
                # 如果直接是tensor
                emb_data = embedding_layer.detach().cpu().numpy()
            else:
                logging.warning(f"Unsupported embedding type for feature {feature}: {type(embedding_layer)}")
                continue
            
            # 如果数据太大，随机采样
            max_samples = 5000
            if emb_data.shape[0] > max_samples:
                indices = np.random.choice(emb_data.shape[0], max_samples, replace=False)
                emb_data = emb_data[indices]
            
            # 应用t-SNE降维到2D
            tsne = TSNE(n_components=2, random_state=42)
            emb_tsne = tsne.fit_transform(emb_data)
            
            # 绘制散点图
            plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], alpha=0.5)
            plt.title(f'Feature: {feature}')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
        
        plt.tight_layout()
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'embedding_tsne_plot_{time_str}.png')
        plt.close()
        
        logging.info(f"t-SNE plots saved to embedding_tsne_plot.png")
    
    def cluster_embedding(self):
        for feature, embedding_layer in self.embedding_layers.items():
            if isinstance(embedding_layer, nn.Embedding):
                if embedding_layer.weight.shape[0] <= 20:
                    continue
                emb_data = embedding_layer.weight.detach().cpu().numpy()
                
                scaler = StandardScaler()
                emb_data_scaled = scaler.fit_transform(emb_data)
                
                # 2. 获取聚类标签 (调用辅助函数)
                try:
                    labels = self._get_cluster_labels(emb_data_scaled, method='hdbscan')
                except Exception as e:
                    logging.error(f"Failed to cluster feature '{feature}': {e}")
                    continue

                # 3. 计算簇中心和更新权重 (这部分逻辑保持不变)
                unique_labels = {l for l in labels if l != -1}
                cluster_count = len(unique_labels)
                noise_count = np.sum(labels == -1)
                
                logging.info(f"Feature '{feature}': Found {cluster_count} clusters with {noise_count} noise points.")

                if cluster_count == 0:
                    logging.warning(f"Feature '{feature}': No clusters found. Skipping weight update.")
                    continue

                centers = {label: emb_data[labels == label].mean(axis=0) for label in unique_labels}
                
                new_weights = np.zeros_like(emb_data)
                for i, label in enumerate(labels):
                    if label != -1:
                        new_weights[i] = centers[label]
                    else:
                        new_weights[i] = emb_data[i] # 噪声点保留原样
                
                # 4. 更新PyTorch层权重
                device = embedding_layer.weight.device
                embedding_layer.weight = nn.Parameter(torch.from_numpy(new_weights).float().to(device))
                logging.info(f"Feature '{feature}' weights updated successfully.")


    def _get_cluster_labels(self, data, method, **kwargs):
        """
        根据指定的方法对数据进行聚类并返回标签。

        Args:
            data (np.ndarray): 待聚类的数据 (已经标准化).
            method (str): 聚类方法，可选 'dbscan', 'hdbscan', 'umap'.
            **kwargs: 传递给聚类算法的额外参数.

        Returns:
            np.ndarray: 每个数据点的聚类标签.
        """
        num_points, dimensions = data.shape
        logging.info(f"Clustering {num_points} points of {dimensions} dimensions using '{method}' method.")

        if method == 'dbscan':
            # --- 动态选择参数的DBSCAN ---
            min_samples = kwargs.get('min_samples', 2 * dimensions)
            if min_samples >= num_points: min_samples = num_points - 1
            if min_samples <= 1: return np.zeros(num_points, dtype=int)

            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(data)
            distances, _ = nbrs.kneighbors(data)
            k_distances = np.sort(distances[:, min_samples - 1])
            
            # 自动寻找拐点作为eps (一个简单的启发式方法)
            gradients = np.gradient(k_distances)
            elbow_index = np.argmax(gradients)
            chosen_eps = k_distances[elbow_index] if elbow_index > 0 else np.median(k_distances)
            
            logging.info(f"DBSCAN params: eps={chosen_eps:.4f}, min_samples={min_samples}")
            clusterer = DBSCAN(eps=chosen_eps, min_samples=min_samples, n_jobs=-1)
            return clusterer.fit_predict(data)

        elif method == 'hdbscan':
            # --- 直接使用HDBSCAN ---
            min_cluster_size = kwargs.get('min_cluster_size', 15)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        prediction_data=True, # 允许预测新点
                                        core_dist_n_jobs=-1)
            return clusterer.fit_predict(data)

        elif method == 'umap':
            # --- UMAP + HDBSCAN 组合方法 ---
            # UMAP 参数
            n_neighbors = kwargs.get('n_neighbors', 15)
            n_components = kwargs.get('n_components', 5) # 降维后的维度
            min_dist = kwargs.get('min_dist', 0.0)
            
            # HDBSCAN 参数
            min_cluster_size = kwargs.get('min_cluster_size', 15)

            logging.info(f"UMAP params: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}")
            logging.info(f"HDBSCAN params: min_cluster_size={min_cluster_size}")

            # 1. UMAP降维
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                random_state=42,
            )
            embedding = reducer.fit_transform(data)
            
            # 2. 在降维后的空间上运行HDBSCAN
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=-1)
            return clusterer.fit_predict(embedding)

        else:
            raise ValueError(f"Unknown clustering method: {method}")
