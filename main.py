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
from model import TransformerEncoder
from metrics import loss_and_metrics
from pandas.api.types import CategoricalDtype


class FeatureVectorDataset(Dataset):
    def __init__(self, df, embeddings_dict, rbp_matrices_dict, config):
        self.ids = df["ID"].tolist()  # Embeddings key : ID
        self.refseq_ids = df["Refseq_id"].tolist()  # RBP matrix key : Refseq_id

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


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_embeddings_npy(directory, ids):
    embeddings = {}
    for id_ in tqdm(ids, desc="Loading .npy embeddings", mininterval=10):
        path = os.path.join(directory, f"{id_}.npy")
        if os.path.exists(path):
            embeddings[id_] = np.load(path)
        else:
            raise FileNotFoundError(f"Embedding file not found: {path}")
    return embeddings

def load_rbp_matrices_csv(directory_eclip, directory_reformer, refseq_ids, id_to_refseq_map):
    """CSVファイルから RBP 行列と ID 情報をロード
    
    CSV 構造（新形式）:
    - 行 0: ヘッダー (RBPの名前) - AARS_K562,AATF_K562,ABCF1_K562,...
    - 行 1～配列長行まで: 各ポジションの 0/1 値 - 0,0,0,0,0,0,0,0,0,0,...
    
    Args:
        directory_eclip: ECLIP形式のRBP行列ファイルが格納されたディレクトリ
                        ファイル名: {refseq_id}.csv
        directory_reformer: Reformer形式のRBP行列ファイルが格納されたディレクトリ
                           ファイル名: RNA_{id}.csv
        refseq_ids: ロード対象のRefseq_idのリスト
        id_to_refseq_map: {id: refseq_id} のマッピング（Reformerディレクトリ用）
    
    Returns:
        tuple: (matrices_dict, ids_dict)
            - matrices_dict: {refseq_id: rbp_matrix_array (seq_len, num_rbps) or None}
            - ids_dict: {refseq_id: {"refseq_id": str, "length": int}}
    """
    matrices = {}
    ids_info = {}
    
    for refseq_id in tqdm(refseq_ids, desc="Loading RBP matrices from CSV", mininterval=10):
        matrices[refseq_id] = None
        rbp_matrix = None
        
        # Step 1: directory_eclip で {refseq_id}.csv を探す
        path_eclip = os.path.join(directory_eclip, f"{refseq_id}.csv")
        if os.path.exists(path_eclip):
            try:
                df = pd.read_csv(path_eclip)
                if len(df) > 0:
                    # ヘッダーはRBP名（行0）、データは行1から
                    rbp_matrix = df.iloc[1:, :].values.astype(np.float32)
                    if rbp_matrix.size > 0:
                        print(f"DEBUG: {refseq_id} - Loaded from eclip, shape={rbp_matrix.shape}")
            except Exception as e:
                print(f"Warning: Error loading RBP matrix from eclip {path_eclip}: {e}")
        
        # Step 2: eclip でロード失敗の場合、directory_reformer で {id}.csv を探す
        if rbp_matrix is None:
            # refseq_id に対応する id を探す
            corresponding_id = None
            for id_, rid in id_to_refseq_map.items():
                if rid == refseq_id:
                    corresponding_id = id_
                    break
            
            if corresponding_id is not None:
                possible_names = [f"{corresponding_id}.csv"]
                
                for filename in possible_names:
                    path_reformer = os.path.join(directory_reformer, filename)
                    if os.path.exists(path_reformer):
                        try:
                            df = pd.read_csv(path_reformer)
                            if len(df) > 0:
                                # 行0はヘッダー(RBP名)、データは行1から
                                rbp_matrix = df.iloc[1:, :].values.astype(np.float32)
                                if rbp_matrix.size > 0:
                                    print(f"DEBUG: {refseq_id} - Loaded from reformer using '{filename}', shape={rbp_matrix.shape}")
                                    break
                        except Exception as e:
                            print(f"Warning: Error loading RBP matrix from reformer {path_reformer}: {e}")
        
        # Step 3: いずれかから正常にロードできた場合、matrices に保存
        if rbp_matrix is not None and rbp_matrix.size > 0:
            seq_len = rbp_matrix.shape[0]
            num_rbps = rbp_matrix.shape[1]
            
            # ID 情報を保存
            ids_info[refseq_id] = {
                "refseq_id": str(refseq_id),
                "length": int(seq_len)
            }
            
            matrices[refseq_id] = rbp_matrix
        else:
            # どちらからもロード失敗した場合は None のまま（RNA配列からのみの予測）
            print(f"DEBUG: {refseq_id} - No RBP data available in either directory")
    
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
    
    # rbp_tensors: 全て None か、全て tensor か、混在か
    if all(r is None for r in rbp_tensors):
        # 全て None の場合
        rbp_stacked = None
    else:
        # 混在している場合、None は None のままリストに保持
        # （モデルの forward で None チェックが入っているため）
        rbp_stacked = rbp_tensors
    
    return embeddings, rna_types, species, rbp_stacked, labels


def get_dataloaders(df_train, df_valid, embedding_dir, rbp_matrices, config):
    # ID をキーとして embeddings をロード（元のまま）
    train_ids = df_train["ID"].tolist()
    valid_ids = df_valid["ID"].tolist()
    
    # train_ids = df_train["Refseq_id"].tolist()
    # valid_ids = df_valid["Refseq_id"].tolist()

    train_embeddings = load_embeddings_npy(embedding_dir, train_ids)
    valid_embeddings = load_embeddings_npy(embedding_dir, valid_ids)

    train_dataset = FeatureVectorDataset(df_train, train_embeddings, rbp_matrices, config)
    val_dataset = FeatureVectorDataset(df_valid, valid_embeddings, rbp_matrices, config)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn_with_none_rbp)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn_with_none_rbp)

    return train_loader, val_loader


def train(model, loader, optimizer, device, epoch, writer, config):
    model.train()
    total_loss, total_acc = 0, 0
    num_labels = config["num_labels"]

    all_preds, all_labels = [], []
    all_metrics = {
        'sample_avg_precision': 0.0,
        'ranking_loss': 0.0,
        'hamming_loss': 0.0,
        'coverage': 0.0,
        'one_error': 0.0,
        'sample_accuracy': 0.0,
        'per_label_acc': torch.zeros(num_labels, device=device),
        'precision': torch.zeros(num_labels, device=device),
        'recall': torch.zeros(num_labels, device=device),
        'specificity': torch.zeros(num_labels, device=device),
        'npv': torch.zeros(num_labels, device=device),
        'f1': torch.zeros(num_labels, device=device),
    }

    loop = tqdm(loader, desc=f"Train Epoch {epoch}")

    for batch in loop:
        x, rna_type_tensor, species_tensor, rbp_tensor, y = batch
        x = x.to(device)
        rna_type_tensor = rna_type_tensor.to(device)
        species_tensor = species_tensor.to(device)
        y = y.to(device)
        
        # rbp_tensor の処理: None / tensor / list のいずれか
        if rbp_tensor is not None:
            if isinstance(rbp_tensor, list):
                rbp_tensor = [r.to(device) if r is not None else None for r in rbp_tensor]
            else:
                rbp_tensor = rbp_tensor.to(device)

        optimizer.zero_grad()

        if isinstance(model, torch.nn.DataParallel):
            out = model.module.compute_loss_and_metrics(x, rna_type_tensor, species_tensor, rbp_tensor, y)
        else:
            out = model.compute_loss_and_metrics(x, rna_type_tensor, species_tensor, rbp_tensor, y)

        loss = out["loss"]
        acc = out["acc"]
        y_pred = out["y_pred"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

        for key in all_metrics:
            if isinstance(out[key], torch.Tensor):
                all_metrics[key] += out[key].detach()
            else:
                all_metrics[key] += out[key]

        all_preds.append(y_pred.detach().cpu())
        all_labels.append(y.detach().cpu())

        loop.set_postfix(loss=total_loss / len(all_preds), acc=total_acc / len(all_preds))

    num_batches = len(loader)
    avg_metrics = {k: (v / num_batches if isinstance(v, float) else v.cpu().numpy() / num_batches)
                   for k, v in all_metrics.items()}
    macro_f1 = avg_metrics["f1"].mean()

    # Micro F1
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

    if epoch == 0:
        header = ['epoch', 'loss', 'acc', 'macro_f1', 'micro_f1']
        header += ["sample_avg_precision", "ranking_loss", "hamming_loss", "coverage", "one_error", "sample_accuracy"]
        for key in ['per_label_acc', 'precision', 'recall', 'specificity', 'npv', 'f1']:
            header += [f"{key}_{i}" for i in range(num_labels)]
        writer.writerow(header)

    row = [epoch, total_loss / num_batches, total_acc / num_batches, macro_f1, micro_f1]
    row += [avg_metrics[k] for k in ["sample_avg_precision", "ranking_loss", "hamming_loss",
                                     "coverage", "one_error", "sample_accuracy"]]
    for key in ['per_label_acc', 'precision', 'recall', 'specificity', 'npv', 'f1']:
        row += avg_metrics[key].tolist()

    writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])


def validate(model, loader, device, epoch, writer, config):
    model.eval()
    total_loss, total_acc = 0, 0
    num_labels = config["num_labels"]

    all_preds, all_labels = [], []
    all_metrics = {
        'sample_avg_precision': 0.0,
        'ranking_loss': 0.0,
        'hamming_loss': 0.0,
        'coverage': 0.0,
        'one_error': 0.0,
        'sample_accuracy': 0.0,
        'per_label_acc': torch.zeros(num_labels, device=device),
        'precision': torch.zeros(num_labels, device=device),
        'recall': torch.zeros(num_labels, device=device),
        'specificity': torch.zeros(num_labels, device=device),
        'npv': torch.zeros(num_labels, device=device),
        'f1': torch.zeros(num_labels, device=device),
    }

    loop = tqdm(loader, desc=f"Val Epoch {epoch}")

    with torch.no_grad():
        for batch in loop:
            x, rna_type_tensor, species_tensor, rbp_tensor, y = batch
            x = x.to(device)
            rna_type_tensor = rna_type_tensor.to(device)
            species_tensor = species_tensor.to(device)
            y = y.to(device)
            
            # rbp_tensor の処理: None / tensor / list のいずれか
            if rbp_tensor is not None:
                if isinstance(rbp_tensor, list):
                    rbp_tensor = [r.to(device) if r is not None else None for r in rbp_tensor]
                else:
                    # Tensor の場合
                    rbp_tensor = rbp_tensor.to(device)

            if isinstance(model, torch.nn.DataParallel):
                out = model.module.compute_loss_and_metrics(x, rna_type_tensor, species_tensor, rbp_tensor, y)
            else:
                out = model.compute_loss_and_metrics(x, rna_type_tensor, species_tensor, rbp_tensor, y)

            loss = out["loss"]
            acc = out["acc"]
            y_pred = out["y_pred"]

            total_loss += loss.item()
            total_acc += acc.item()

            for key in all_metrics:
                if isinstance(out[key], torch.Tensor):
                    all_metrics[key] += out[key].detach()
                else:
                    all_metrics[key] += out[key]

            all_preds.append(y_pred.detach().cpu())
            all_labels.append(y.detach().cpu())

            loop.set_postfix(loss=total_loss / len(all_preds), acc=total_acc / len(all_preds))

    num_batches = len(loader)
    avg_metrics = {k: (v / num_batches if isinstance(v, float) else v.cpu().numpy() / num_batches)
                   for k, v in all_metrics.items()}
    macro_f1 = avg_metrics["f1"].mean()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

    if epoch == 0:
        header = ['epoch', 'loss', 'acc', 'macro_f1', 'micro_f1']
        header += ["sample_avg_precision", "ranking_loss", "hamming_loss", "coverage", "one_error", "sample_accuracy"]
        for key in ['per_label_acc', 'precision', 'recall', 'specificity', 'npv', 'f1']:
            header += [f"{key}_{i}" for i in range(num_labels)]
        writer.writerow(header)

    row = [epoch, total_loss / num_batches, total_acc / num_batches, macro_f1, micro_f1]
    row += [avg_metrics[k] for k in ["sample_avg_precision", "ranking_loss", "hamming_loss",
                                     "coverage", "one_error", "sample_accuracy"]]
    for key in ['per_label_acc', 'precision', 'recall', 'specificity', 'npv', 'f1']:
        row += avg_metrics[key].tolist()

    writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])


def main(log_dir):
    config = load_config("config.yaml")

    print("=== GPU環境確認 ===")
    print(f"CUDA_VISIBLE_DEVICES: '{os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}'")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    print("=== パラメーター ===")
    print(f"num_heads: {config['num_heads']}")
    print(f"num_layers: {config['num_layers']}")
    print(f"lr: {config['lr']}")
    print(f"input_data_train: {config['input_path_train_list']}")
    print(f"Batch size: {config['batch_size']}")

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    df_train = pd.read_csv(config["input_path_train_list"])
    df_valid = pd.read_csv(config["input_path_valid_list"])
    embeddings_dir = config["input_embeddings_dir"]

    label_cols = df_train.iloc[:, config["label_start_index"]:config["feature_start_index"]].columns

    df_train = df_train.dropna(subset=label_cols)
    df_valid = df_valid.dropna(subset=label_cols)

    # RNA_Type フィルタリング
    rna_type_filter = config.get("rna_type_filter", "ALL")
    if rna_type_filter != "ALL":
        if rna_type_filter not in config["rna_type_list"]:
            raise ValueError(
                f"Invalid rna_type_filter '{rna_type_filter}'. "
                f"Expected one of {config['rna_type_list']} or 'ALL'."
            )
        df_train = df_train[df_train["RNA_Type"] == rna_type_filter]
        df_valid = df_valid[df_valid["RNA_Type"] == rna_type_filter]
        print(f"Filtered by RNA_Type='{rna_type_filter}': train={len(df_train)}, valid={len(df_valid)}")

    # RBP 行列をロード（Refseq_id をキーとして使用）
    train_refseq_ids = df_train["Refseq_id"].tolist()
    valid_refseq_ids = df_valid["Refseq_id"].tolist()
    all_refseq_ids = list(set(train_refseq_ids + valid_refseq_ids))
    
    # ID → Refseq_id のマッピングを作成（Reformerディレクトリ用）
    id_to_refseq_map = {}
    for _, row in df_train.iterrows():
        id_to_refseq_map[row["ID"]] = row["Refseq_id"]
    for _, row in df_valid.iterrows():
        id_to_refseq_map[row["ID"]] = row["Refseq_id"]
    
    rbp_matrices, rbp_ids_info = load_rbp_matrices_csv(
        config["rbp_matrix_dir_eclip"],
        config["rbp_matrix_dir_reformer"],
        all_refseq_ids,
        id_to_refseq_map
    )

    train_loader, val_loader = get_dataloaders(df_train, df_valid, embeddings_dir, rbp_matrices, config)
    print("Got dataloaders.")
    print()
    
    train_found_csv = [rid for rid in train_refseq_ids if rid in rbp_ids_info]
    valid_found_csv = [rid for rid in valid_refseq_ids if rid in rbp_ids_info]
    train_with_data = [rid for rid in train_refseq_ids if rid in rbp_matrices and rbp_matrices[rid] is not None]
    valid_with_data = [rid for rid in valid_refseq_ids if rid in rbp_matrices and rbp_matrices[rid] is not None]
    print("=== データ詳細 ===")
    print(f"Train: {len(train_refseq_ids)} refseq_ids in list, {len(train_found_csv)} CSV files found, {len(train_with_data)} with matrix data")
    print(f"Valid: {len(valid_refseq_ids)} refseq_ids in list, {len(valid_found_csv)} CSV files found, {len(valid_with_data)} with matrix data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerEncoder(
        config=config,
        num_rna_type=len(config["rna_type_list"]),
        num_species=len(config["species_list"]),
        rbp_dim=config["num_rbps"],
    )
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    if torch.cuda.device_count() > 0 and config["batch_size"] % torch.cuda.device_count() != 0:
        print(f"Warning: Batch size ({config['batch_size']}) is not divisible by GPU count ({torch.cuda.device_count()})")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_log_path = os.path.join(log_dir, "train_log.csv")
    val_log_path = os.path.join(log_dir, "val_log.csv")

    for epoch in range(config["max_epochs"]):
        with open(train_log_path, 'a', newline='') as f_train, open(val_log_path, 'a', newline='') as f_val:
            train_writer = csv.writer(f_train, delimiter='\t')
            val_writer = csv.writer(f_val, delimiter='\t')

            train(model, train_loader, optimizer, device, epoch, train_writer, config)
            validate(model, val_loader, device, epoch, val_writer, config)

        if (epoch + 1) % config["model_save_interval"] == 0:
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt"))
            print(f"Model saved to model_epoch{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/default_run")
    args = parser.parse_args()
    main(args.log_dir)
