import argparse
import os
import random
import pickle
import numpy as np
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn import MSELoss

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from deberta_ITHP import ITHP_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_result_filename(variant=None):
    timestamp = get_timestamp()
    if variant:
        return f"result_{variant}_{timestamp}.txt"
    return f"result_{timestamp}.txt"


def log_results(file_path, message):
    with open(file_path, 'a') as f:
        f.write(message + '\n')
        f.flush()


def calculate_binary_accuracy(preds, labels):
    binary_preds = (preds >= 0).astype(int)
    binary_labels = (labels >= 0).astype(int)
    binary_acc = accuracy_score(binary_labels, binary_preds)
    return binary_acc


def train(
        model,
        train_dataloader,
        validation_dataloader,
        test_data_loader,
        optimizer,
        scheduler,
        result_file,
):
    valid_losses = []
    test_accuracies = []
    mae_list = []
    corr_list = []
    f1_list = []
    ba_list = []  

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch_i, args.n_epochs)
        valid_loss = eval_epoch(model, validation_dataloader, epoch_i, args.n_epochs)

        if epoch_i != args.n_epochs - 1:
            train_message = f"TRAIN: epoch:{epoch_i + 1}, train_loss:{train_loss}, valid_loss:{valid_loss}"
            print(train_message)
            log_results(result_file, train_message)
        else:
            test_acc, test_mae, test_corr, test_f_score, test_ba = test_score_model(
                model, test_data_loader, epoch_i, args.n_epochs
            )
            test_message = (
                f"TEST: train_loss:{train_loss}, valid_loss:{valid_loss}, "
                f"test_acc:{test_acc}, mae:{test_mae}, corr:{test_corr}, "
                f"f1_score:{test_f_score}, binary_acc:{test_ba}"
            )
            print(test_message)
            log_results(result_file, test_message)

    return train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score, test_ba 


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--n_epochs", type=int, default=3)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=128)
parser.add_argument('--inter_dim', default=256, help='dimension of inter layers', type=int)
parser.add_argument("--drop_prob", help='drop probability for dropout -- encoder', default=0.3, type=float)
parser.add_argument('--p_lambda', default=0.3, help='coefficient -- lambda', type=float)
parser.add_argument('--p_beta', default=8, help='coefficient -- beta', type=float)
parser.add_argument('--p_gamma', default=32, help='coefficient -- gamma', type=float)
parser.add_argument('--beta_shift', default=1.0, help='coefficient -- shift', type=float)
parser.add_argument('--IB_coef', default=10, type=float)
parser.add_argument('--B0_dim', default=128, type=float)
parser.add_argument('--B1_dim', default=64, type=float)
parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
parser.add_argument("--save_model", action="store_true", default=False, help="Whether to save the trained model")
parser.add_argument("--model_save_dir", type=str, default="saved_models", help="Directory to save models")

# 🔥 9种消融实验变体配置
parser.add_argument("--run_full_ablation", action="store_true", default=False,
                    help="Run full ablation study for all 9 variants")
parser.add_argument("--ablation_variant", type=str, default="GICA",
                    choices=["T", "A", "V", "T+A", "T+V", "A+V", "T+A+V", "T+V+A", "GICA"],
                    help="Ablation variant to run")

# 保留原有的融合模式配置(用于GICA变体)
parser.add_argument("--fusion_mode", type=str, default="full",
                    choices=["b1_only", "b1_acoustic", "b1_visual", "full"],
                    help="Fusion mode for GICA variant")
parser.add_argument("--gating_mode", type=str, default="dual_gating",
                    choices=["no_gating", "single_gating", "dual_gating"],
                    help="Gating mechanism mode")

args = parser.parse_args()

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM, global_configs.TEXT_DIM)


class InputFeatures(object):
    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        prepare_input = prepare_deberta_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    tokenizer = DebertaV2Tokenizer.from_pretrained(model)
    return tokenizer


def get_appropriate_dataset(data):
    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    # 🔥 修改：直接加载单个pkl文件
    with open(f"./datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)
    
    # 假设pkl文件包含train, dev, test三个键
    train_data = data['train']
    dev_data = data['dev'] 
    test_data = data['test']

    train_dataset = get_appropriate_dataset(train_data)

    num_train_optimization_steps = (
            int(
                len(train_dataset) / args.train_batch_size /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0
    )

    dev_dataset = get_appropriate_dataset(dev_data)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True, num_workers=0
    )

    test_dataset = get_appropriate_dataset(test_data)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=0
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )

def set_random_seed(seed: int):
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int):
    from transformers import DebertaV2Config
    from types import SimpleNamespace

    multimodal_config = SimpleNamespace(
        beta_shift=args.beta_shift,
        dropout_prob=args.dropout_prob,
        inter_dim=args.inter_dim,
        drop_prob=args.drop_prob,
        p_beta=args.p_beta,
        p_gamma=args.p_gamma,
        p_lambda=args.p_lambda,
        max_seq_length=args.max_seq_length,
        B0_dim=int(args.B0_dim),
        B1_dim=int(args.B1_dim),
        ablation_variant=args.ablation_variant,  # 🔥 传递变体配置
        fusion_mode=args.fusion_mode,
        gating_mode=args.gating_mode,
    )

    config = DebertaV2Config.from_pretrained(args.model, num_labels=1)
    model = ITHP_DeBertaForSequenceClassification(config, multimodal_config)

    model.to(DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps * args.warmup_proportion,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch, max_epochs):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        
        visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

        logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1 = model(
            input_ids, visual_norm, acoustic_norm, input_mask, epoch, max_epochs
        )

        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))
        
        total_loss = loss + args.IB_coef * IB_loss

        if args.gradient_accumulation_step > 1:
            total_loss = total_loss / args.gradient_accumulation_step

        total_loss.backward()

        tr_loss += total_loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model, validation_dataloader, epoch, max_epochs):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(validation_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            
            visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1 = model(
                input_ids, visual_norm, acoustic_norm, input_mask, epoch, max_epochs
            )

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            total_loss = loss + args.IB_coef * IB_loss

            if args.gradient_accumulation_step > 1:
                total_loss = total_loss / args.gradient_accumulation_step

            dev_loss += total_loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model, test_data_loader, epoch, max_epochs):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            
            visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1 = model(
                input_ids, visual_norm, acoustic_norm, input_mask, epoch, max_epochs
            )

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model, test_data_loader, epoch, max_epochs, use_zero=False):
    preds, y_test = test_epoch(model, test_data_loader, epoch, max_epochs)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0])

    test_preds_a7 = np.clip(preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(y_test, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(y_test, a_min=-2.0, a_max=2.0)

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]
    mult_a7 = accuracy_score((test_truth_a7[non_zeros] > 0), (test_preds_a7[non_zeros] > 0))

    f_score = f1_score((test_truth_a7[non_zeros] > 0), (test_preds_a7[non_zeros] > 0), average="weighted")

    binary_truth = test_truth_a7[non_zeros] > 0
    binary_preds = test_preds_a7[non_zeros] > 0
    binary_acc = calculate_binary_accuracy(test_preds_a7[non_zeros], test_truth_a7[non_zeros])

    return mult_a7, mae, corr, f_score, binary_acc


def run_full_ablation_study():
    """运行完整的9种变体消融实验"""
    
    print("\n" + "="*100)
    print("🔥 开始完整模态消融实验 (Comprehensive Modality & Architecture Ablation)")
    print("="*100 + "\n")
    
    variants = ['T', 'A', 'V', 'T+A', 'T+V', 'A+V', 'T+A+V', 'T+V+A', 'GICA']
    
    variant_descriptions = {
        'T': '仅文本 (Baseline: DeBERTa → Classifier)',
        'A': '仅声学 (Acoustic → MLP → Classifier)',
        'V': '仅视觉 (Visual → MLP → Classifier)',
        'T+A': '双模态Layer 1 (Text+Acoustic → B_G → Classifier)',
        'T+V': '双模态Layer 1 (Text+Visual → B_F → Classifier)',
        'A+V': '双模态Layer 1 (Acoustic+Visual → B_G → Classifier)',
        'T+A+V': '三模态flat (Concat(T,A,V) → Fusion MLP → Classifier)',
        'T+V+A': '层次变体 (Layer1: T+V→B_G, Layer2: B_G+A→B_1)',
        'GICA': '完整层次模型 (Layer1: T+A→B_G, Layer2: B_G+V→B_1)'
    }
    
    all_results = {}
    
    for variant in variants:
        print("\n" + "="*100)
        print(f"🚀 运行变体 {variant}: {variant_descriptions[variant]}")
        print("="*100 + "\n")
        
        args.ablation_variant = variant
        
        result_file = get_result_filename(variant=variant)
        
        config_message = (
            f"Full Ablation Study - Variant: {variant}\n"
            f"Description: {variant_descriptions[variant]}\n"
            f"Dataset: {args.dataset}\n"
            f"Seed: {args.seed}\n"
            f"Epochs: {args.n_epochs}\n"
            f"Learning Rate: {args.learning_rate}\n"
            f"Batch Size: {args.train_batch_size}\n"
        )
        log_results(result_file, config_message)
        print(config_message)
        
        set_random_seed(args.seed)
        
        (
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            num_train_optimization_steps,
        ) = set_up_data_loader()
        
        model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)
        
        train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score, test_ba = train(
            model,
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            optimizer,
            scheduler,
            result_file,
        )
        
        all_results[variant] = {
            'description': variant_descriptions[variant],
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'test_acc': test_acc,
            'test_mae': test_mae,
            'test_corr': test_corr,
            'test_f_score': test_f_score,
            'test_ba': test_ba
        }
        
        print(f"\n✅ 变体 {variant} 完成!")
    
    print("\n" + "="*100)
    print("📊 完整消融实验汇总结果 (Full Ablation Study Summary)")
    print("="*100 + "\n")
    
    summary_file = get_result_filename(variant="full_ablation_summary")
    
    summary_header = (
        f"{'Variant':<10} {'Description':<55} {'Acc':<8} {'MAE':<8} {'Corr':<8} {'F1':<8} {'BA':<8}\n"
        + "="*120 + "\n"
    )
    print(summary_header)
    log_results(summary_file, summary_header)
    
    for variant in variants:
        results = all_results[variant]
        summary_line = (
            f"{variant:<10} {results['description']:<55} "
            f"{results['test_acc']:<8.4f} {results['test_mae']:<8.4f} "
            f"{results['test_corr']:<8.4f} {results['test_f_score']:<8.4f} "
            f"{results['test_ba']:<8.4f}"
        )
        print(summary_line)
        log_results(summary_file, summary_line)
    
    print("\n" + "="*100)
    print("🎉 完整消融实验全部完成!")
    print("="*100 + "\n")
    
    return all_results


def main():
    print(f"Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    
    if args.run_full_ablation:
        run_full_ablation_study()
    else:
        result_file = get_result_filename(variant=args.ablation_variant)
        
        config_message = (
            f"Single Variant Training\n"
            f"Variant: {args.ablation_variant}\n"
            f"Dataset: {args.dataset}\n"
            f"Seed: {args.seed}\n"
        )
        print(config_message)
        log_results(result_file, config_message)
        
        set_random_seed(args.seed)
        (
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            num_train_optimization_steps,
        ) = set_up_data_loader()

        model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)

        train(
            model,
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            optimizer,
            scheduler,
            result_file,
        )

    print(f"Experiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
