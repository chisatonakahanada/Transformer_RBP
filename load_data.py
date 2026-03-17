import os
import csv
import json
import torch
import yaml
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pandas.api.types import CategoricalDtype

class FeatureVectorDataset(Dataset):
    def __init__(self, df, embeddings_dict, rbp_matrices_dict, config):
        self.ids = df["ID"].tolist()  
        self.refseq_ids = df["Refseq_id"].tolist()  

        label_df = df.iloc[:, config["label_start_index"]:config["feature_start_index"]]
        label_numeric = label_df.apply(pd.to_numeric, errors='coerce')
        if label_numeric.isnull().values.any():
            print("Warning: NaN detected in label columns. Please check your label data.")
        self.labels = torch.tensor(label_numeric.values, dtype=torch.float32)

        rna_type_list = config["rna_type_list"]
        species_list = config["species_list"]

        # Category -> One-hot
        df["RNA_Type"] = df["RNA_Type"].astype(CategoricalDtype(categories=rna_type_list))
        df["Species"] = df["Species"].astype(CategoricalDtype(categories=species_list))

        self.species_onehot = torch.tensor(
            pd.get_dummies(df["Species"], prefix="Species").astype(float).values,
            dtype=torch.float32
        )
        self.rna_type_onehot = torch.tensor(
            pd.get_dummies(df["RNA_Type"], prefix="RNA_Type").astype(float).values,
            dtype=torch.float32
        )

        self.embeddings_dict = embeddings_dict
        self.rbp_matrices_dict = rbp_matrices_dict
        self.config = config

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]  # エンベディング用のキー（ID）
        refseq_id = self.refseq_ids[idx]  # RBP matrix 用のキー（Refseq_id）

        # 埋め込み (seq_len, emb_dim) - ID をキーとして取得
        embedding = self.embeddings_dict[id_]
        embedding_tensor = torch.from_numpy(embedding).float()

        # RBP 0/1 行列 (seq_len, num_rbps) - Refseq_id をキーとして取得
        rbp_tensor = None
        if refseq_id in self.rbp_matrices_dict and self.rbp_matrices_dict[refseq_id] is not None:
            rbp_mat = self.rbp_matrices_dict[refseq_id]
            rbp_tensor = torch.from_numpy(rbp_mat).float()
        else:
            pass

        rna_type_tensor = self.rna_type_onehot[idx]
        species_tensor = self.species_onehot[idx]
        label_tensor = self.labels[idx]

        return embedding_tensor, rna_type_tensor, species_tensor, rbp_tensor, label_tensor
    

def load_embeddings_npy(directory, ids):
    embeddings = {}
    missing_ids = []
    
    for id_ in tqdm(ids, desc="Loading embeddings", mininterval=10):
        path = os.path.join(directory, f"{id_}.npy")
        if os.path.exists(path):
            try:
                embeddings[id_] = np.load(path)
            except Exception:
                missing_ids.append(id_)
        else:
            missing_ids.append(id_)
    
    return embeddings, missing_ids


def load_rbp_matrices_csv(directory, ids, key_name="", num_rbps=None):
    """CSVファイルから RBP マトリックスと ID 情報をロード
    
    Args:
        directory: RBP行列ファイルが格納されたディレクトリ
        ids: ロード対象のIDのリスト（Refseq_idまたはID）
        key_name: ログ用のキー名（例: "eCLIP", "Reformer"）
        num_rbps: RBPの数。150の場合、重複列を除外する
    
    Returns:
        tuple: (matrices_dict, ids_dict)
            - matrices_dict: {id: rbp_matrix_array (seq_len, num_rbps) or None}
            - ids_dict: {id: {"refseq_id": str, "length": int}}
    """
    if not os.path.exists(directory):
        print(f"  Directory not found: {directory}")
        return {}, {}
    
    matrices = {}
    ids_info = {}
    loaded_count = 0
    
    desc = f"Loading RBP ({key_name})" if key_name else "Loading RBP matrices"
    for id_ in tqdm(ids, desc=desc, mininterval=10, disable=len(ids)==0):
        matrices[id_] = None
        path = os.path.join(directory, f"{id_}.csv")
        
        if not os.path.exists(path):
            continue
        
        try:
            df = pd.read_csv(path)
            if len(df) == 0:
                continue
            
            # num_rbps が 150 の場合、重複する列（前半が一致するもの）を除外
            if num_rbps == 150:
                columns = df.columns.tolist()
                rbp_groups = {}
                for col in columns:
                    rbp_name = col.split('_')[0]  
                    if rbp_name not in rbp_groups:
                        rbp_groups[rbp_name] = []
                    rbp_groups[rbp_name].append(col)
                
                selected_columns = []
                for rbp_name in sorted(rbp_groups.keys()):
                    sorted_cols = sorted(rbp_groups[rbp_name])
                    selected_columns.append(sorted_cols[0]) 
                
                df = df[selected_columns]
            
            rbp_matrix = df.values.astype(np.float32)
            if rbp_matrix.size == 0:
                continue
            
            seq_len, num_rbps_actual = rbp_matrix.shape
            ids_info[id_] = {"refseq_id": str(id_), "length": int(seq_len)}
            matrices[id_] = rbp_matrix
            loaded_count += 1
            
        except Exception:
            continue
    
    return matrices, ids_info

def collate_fn_with_none_rbp(batch):
    """None を含む RBP tensor を処理するカスタム collate 関数
    
    batch: [(embedding, rna_type, species, rbp_tensor or None, label), ...]
    
    Returns: (embedding, rna_type, species, rbp_tensor or None, label)
             rbp_tensor が全て None の場合は None、それ以外は stack したもの
    """
    embeddings = []
    rna_types = []
    species = []
    rbp_tensors = []
    labels = []
    
    for embedding, rna_type, sp, rbp, label in batch:
        embeddings.append(embedding)
        rna_types.append(rna_type)
        species.append(sp)
        rbp_tensors.append(rbp)
        labels.append(label)
    
    embeddings = torch.stack(embeddings)
    rna_types = torch.stack(rna_types)
    species = torch.stack(species)
    labels = torch.stack(labels)
    
    if all(r is None for r in rbp_tensors):
        rbp_stacked = None
    else:
        rbp_stacked = rbp_tensors
    
    return embeddings, rna_types, species, rbp_stacked, labels


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
