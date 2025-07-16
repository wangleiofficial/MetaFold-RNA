import os
import pickle
import numpy as np
from MetaFold.utils import metrics_fn, save2bpseq, creatmat
import torch
import torch.nn as nn
import torch.optim as optim
from MetaFold.model import MetaLearnerNet_v3
from MetaFold.data import RNADataset, RNADatasetNoLabels
from MetaFold.postprocess import postprocess
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

def evaluate(model, dataloader, device):
    model.eval()
    results = []
    result_dict = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", colour="blue"):
            predicts = batch["features"].to(device)
            labels = batch["labels"].to(device)
            rna_seq = batch["sequence"][0]
            rna_length = len(rna_seq)
            seq_feature = batch["seq_encoded"].to(device) 
            one_hot_seq_feature = F.one_hot(seq_feature, num_classes=4)

            # Create pair feature matrix
            pair_feature = creatmat(rna_seq)
            pair_feature = torch.tensor(pair_feature, dtype=torch.float).to(device)
            pair_feature = pair_feature.unsqueeze(0)  # add channel dimension
            pair_feature = pair_feature.unsqueeze(0)  # add batch dimension

            logits = model(predicts, seq_feature, pair_feature)
            
            postprocess_final_pred_contact = postprocess(logits, one_hot_seq_feature.float(), 0.01, 0.1, 100, 2.0, True, 2.2)
            pred = (postprocess_final_pred_contact > 0.5).float()
            result = metrics_fn(pred.cpu().squeeze(0), labels.cpu())
            results.append(result)
            # Store results in a dictionary for later analysis
            result_dict[batch["name"]] = {'pred_probs': postprocess_final_pred_contact.cpu().numpy(),
                                           'pred_contact': pred.cpu().numpy(),
                                           'rna_seq': rna_seq,}

    precision, recall, f1_score = map(list, zip(*results))
    avg_precision = np.average(precision)
    avg_recall = np.average(recall)
    avg_f1 = np.average(f1_score)

    return avg_precision, avg_recall, avg_f1, result_dict

# test args 
def parse_args():
    """Parse command line arguments for test."""
    import argparse
    parser = argparse.ArgumentParser(description="MetaFold-RNA Test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--casp_pdb_path', type=str, default="./dataset", help='Path to  the dataset')
    group.add_argument('--fasta-path', type=str, default="./dataset/test.fasta", help='Path to the FASTA file for testing')
    # model path                                
    parser.add_argument('--model-path', type=str, default="./model_checkpoint/model_pdb.pth", help='Path to the trained model')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device ID to use for testing')
    parser.add_argument('--output-path', type=str, default="./output", help='Path to the output directory')
    return parser.parse_args()

# =================================================================================
# Section 3: run MetaFold-RNA
# =================================================================================

if __name__ == '__main__':
    # --- 1. set logger ---
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 2. parse args ---
    args = parse_args()

    # --- 3. set device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    if args.fasta_path:
        fasta_path = args.fasta_path
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA path '{fasta_path}' does not exist. Please ensure the FASTA file is available.")
        logger.info(f"FASTA path: {fasta_path}")
        basename = os.path.basename(fasta_path)
        if not basename.endswith('.fasta'):
            raise ValueError(f"FASTA path '{fasta_path}' is invalid. Please provide a valid FASTA file.")
        # load fasta dataset
        test_dataset = RNADatasetNoLabels(fasta_path)
        test_loader = DataLoader(test_dataset, batch_size=1)
        test_loaders = {f"{basename.split('.')[0]}": test_loader}
    else:
        # load dataset from data_path
        dataset_path = args.data_path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist. Please ensure the dataset is available.")
        logger.info(f"Dataset path: {dataset_path}")

        casp15_test_path = os.path.join(dataset_path, "CASP15.pickle")
        casp16_test_path = os.path.join(dataset_path, "CASP16.pickle")
        pdb_test_path = os.path.join(dataset_path, "PDB.pickle")
        test_datasets_path = {
            "CASP15": casp15_test_path,
            "CASP16": casp16_test_path,
            "PDB": pdb_test_path,
        }
        test_loaders = {}
        for key, path in test_datasets_path.items():
            test_dataset = RNADataset(path)
            test_loaders[key] = DataLoader(test_dataset, batch_size=1)

    # --- 4. init model ---
    logger.info("\n--- Initializing Model ---")
    model = MetaLearnerNet_v3(num_blocks=3,
                              num_categories=4,
                              seq_dim=64,
                              pair_dim=32,
                              pair_feature_channels=1,
                              predicts_channels=4).to(device)

    model.load_state_dict(torch.load(args.model_path))

    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # --- 5. run metafold ---
    test_metrics = {}
    for data_name, test_loader in test_loaders.items():
        logger.info(f"Evaluating on {data_name} dataset...")
        avg_precision, avg_recall, avg_f1, result_dict = evaluate(model, test_loader,device)
        test_metrics[data_name] = {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1,
            }
        logger.info("-" * 50)
        for key, metrics in test_metrics.items():
            logger.info(f"  Test Metrics ({key}): Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-Score={metrics['f1_score']:.4f}")
        logger.info("-" * 50)

        # --- 6. save results ---
        output_path = args.output_path  
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, data_name)):
            os.makedirs(os.path.join(output_path, data_name))
        logger.info(f"Saving results to {os.path.join(output_path, data_name)}")
        # save results to a pickle file
        result_file = os.path.join(output_path, data_name, "results.pickle")
        with open(result_file, 'wb') as f:
            pickle.dump(result_dict, f)
        for key, result in result_dict.items():
            # bpseq save
            bpseq_file = os.path.join(output_path, data_name, f"{key}_pred.bpseq")
            save2bpseq(result['rna_seq'], result['pred_contact'], bpseq_file)

    logger.info("--- Evaluation Finished ---")





