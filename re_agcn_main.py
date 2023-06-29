from __future__ import absolute_import, division, print_function
from model.bert import BertModel
import argparse
import logging
import os
import random
import time
import json
import datetime
import torch.distributed
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset,Dataset)
from tqdm import tqdm, trange
from torch.cuda import amp
from re_agcn_model import ReAgcn
from model import BertConfig
from model import BertTokenizer
from model import BertAdam
import optuna
from model import LinearWarmUpScheduler
from model import VOCAB_NAME
from data_utils import (
    RE_Processor
)
# from apex import amp
from utils import is_main_process,save_zen_model
from metrics import (
    compute_metrics,
    compute_micro_f1,
    semeval_official_eval
)
from optuna.visualization import plot_optimization_history
torch.autograd.set_detect_anomaly(True)
scaler = torch.cuda.amp.GradScaler()
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

last_memory=0
def get_memory():
    global last_memory
    last_memory=torch.cuda.memory_allocated()/1024/1024
    return last_memory
def get_memory_diff():
    last=last_memory
    total=get_memory()
    return total-last,total

def get_model_param(model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape))
        if p.requires_grad:
            n_trainable_params += n_params.item()
        else:
            n_nontrainable_params += n_params.item()
    logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
    logger.info('> training arguments:')
    return {
        "n_trainable_params": n_trainable_params,
        "n_nontrainable_params": n_nontrainable_params,
        "n_params": n_nontrainable_params + n_trainable_params
    }

def train(args, model, tokenizer, processor, device, n_gpu, results={}):
    results["best_checkpoint"] = 0
    results["best_acc_score"] = 0
    results["best_f1_score"] = 0
    results["best_dev_f1_score"] = 0
    results["best_mrr_score"] = 0
    results["best_checkpoint_path"] = ""
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,

                             t_total=num_train_optimization_steps)
    print("lr: {} warm: {} total_step: {}".format(args.learning_rate, args.warmup_proportion, num_train_optimization_steps))


    # if n_gpu > 1:
    #
    #     model = torch.nn.DataParallel(model,device_ids=[0,1])


    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    train_data = processor.build_dataset(train_examples, tokenizer, args.max_seq_length,args.max_tree_length, "train", args)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,shuffle=(train_sampler is None), batch_size=args.train_batch_size)

    for epoch_num in trange(int(args.num_train_epochs), desc="Epoch"):

        optimizer.zero_grad()
        model.zero_grad()
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_iter = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_iter):
            if args.max_steps > 0 and global_step > args.max_steps:
                break
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, valid_ids, segment_ids, label_ids, e1_mask, e2_mask, dep_type_matrix,tree,node_order,edge_order,tree_type,finlist,treerev,node_orderrev,edge_orderrev,dep_type,dep_adj = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, e1_mask=e1_mask, e2_mask=e2_mask,
                         valid_ids=valid_ids, dep_adj_matrix=dep_type_matrix, dep_type_matrix=dep_type_matrix,
                         tree=tree, node_order=node_order, edge_order=edge_order,finlist=finlist,treerev=treerev,
                         noderev=node_orderrev,edgerev=edge_orderrev,dep_type=dep_type,dep_adj=dep_adj)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:

                with torch.autograd.detect_anomaly():
                   loss.backward()

                # scaler.scale(loss).backward()

            # print(model.bert.embeddings.word_embeddings.weight.grad)
            # print(model.classifier.weight.grad)

            if is_main_process():
                train_iter.update(1)
                perplexity = torch.exp(torch.tensor(loss))
                train_iter.set_postfix_str(f"Step: {global_step} Loss: {loss:.5f} ppl: {perplexity:.5f}")

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                global_step += 1

        if args.local_rank == -1 or torch.distributed.get_rank() == 0 or args.world_size <= 1:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "epoch-{}".format(epoch_num))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_zen_model(output_dir, model, processor, tokenizer)

            #eval dev
            # result = evaluate(args, model, tokenizer, processor, device, mode="dev")
            # logger.info(result)


    loss = tr_loss / nb_tr_steps if args.do_train else None
    return loss, global_step

def evaluate(args, model, tokenizer, processor, device, mode="test", output_dir='./'):
    label_map = processor.labels_dict
    id2label_map = {i : label for label, i in processor.labels_dict.items()}
    id2label_map[6] = 'Other'

    if mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    eval_data = processor.build_dataset(examples, tokenizer, args.max_seq_length,args.max_tree_length, mode, args)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0
    pred_scores = None
    out_label_ids = None
    eval_start_time = time.time()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, valid_ids, segment_ids, label_ids, e1_mask, e2_mask, dep_type_matrix,tree,node_order,edge_order,tree_type,finlist,treerev,node_orderrev,edge_orderrev,dep_type,dep_adj = batch



        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, e1_mask=e1_mask, e2_mask=e2_mask,
                           dep_adj_matrix=dep_type_matrix, dep_type_matrix=dep_type_matrix,
                           valid_ids=valid_ids, tree=tree, node_order=node_order, edge_order=edge_order,
                           finlist=finlist,treerev=treerev,noderev=node_orderrev,edgerev=edge_orderrev,dep_type=dep_type,dep_adj=dep_adj)

        nb_eval_steps += 1
        if pred_scores is None:
            pred_scores = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            pred_scores = np.append(pred_scores, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    preds = np.argmax(pred_scores, axis=1)

    eval_run_time = time.time() - eval_start_time

    if args.task_name == 'semeval':

        result = semeval_official_eval(id2label_map, preds, out_label_ids, output_dir)
    else:
        result = {
            "f1":compute_micro_f1(preds, out_label_ids, label_map, ignore_label='other', output_dir=output_dir)
        }

    result["eval_run_time"] = eval_run_time
    result["inference_time"] = eval_run_time / len(examples)

    logging.info(result)

    return result

def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="./",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Model path")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="Model name")
    parser.add_argument("--max_seq_length",
                        default=200,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_tree_length",
                        default=200,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to predict.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=10,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=-1.0,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.06,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--vocab_file',
                        type=str, default=None,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="world size")
    parser.add_argument('--init_method',
                        type=str,
                        default='tcp://127.0.0.1:23456')
    parser.add_argument('--dep_type', type=str, default='local_global_graph',choices=["full_graph", "local_graph","global_graph","local_global_graph"])
    parser.add_argument('--num_gcn_layers', type=int, default=2)
    parser.add_argument('--gpu_id', type=str, default='0,1')
    # parser.add_argument('--local_rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    args = parser.parse_args()

    args.task_name = args.task_name.lower()

    return args
args = get_args()
def objective(trial):
    lr=trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    batchsize=trial.suggest_int("batchsize",4,256)
    warmingup=trial.suggest_float("warmingup",0.01,0.99)
    epoch=40
    print(lr)
    print(batchsize)
    print(warmingup)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "cuda" if torch.cuda.is_available() and not args.no_cuda else
    args.n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    batchsize = batchsize // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    processor = RE_Processor(dep_type=args.dep_type)
    processor.prepare_type_dict(args.data_dir)
    processor.prepare_labels_dict(args.data_dir)
    processor.prepare_tree_type_dict(args.data_dir)
    label_list = processor.labels_dict.keys()
    dep_type_list = processor.types_dict.keys()
    tree_labels = processor.tree_node_type.keys()
    num_labels = len(label_list)
    type_num = len(dep_type_list)
    tree_labels_len = len(tree_labels)

    if args.vocab_file is None:
        args.vocab_file = os.path.join(args.model_path, VOCAB_NAME)

    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=args.max_seq_length)
    tokenizer.add_never_split_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])

    config = BertConfig.from_json_file(os.path.join(args.model_path, "config.json"))

    config.__dict__["num_gcn_layers"] = args.num_gcn_layers
    config.__dict__["num_labels"] = num_labels
    config.__dict__["type_num"] = type_num
    config.__dict__["dep_type"] = args.dep_type
    config.__dict__["tree_labels_len"] = tree_labels_len
    model = ReAgcn(config)
    model.bert = BertModel.from_pretrained(args.model_path)
    model.to(device)

    n_gpu = args.n_gpu
    results = {}
    results["best_checkpoint"] = 0
    results["best_acc_score"] = 0
    results["best_f1_score"] = 0
    results["best_dev_f1_score"] = 0
    results["best_mrr_score"] = 0
    results["best_checkpoint_path"] = ""
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / batchsize / args.gradient_accumulation_steps) * epoch
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=lr,
                         warmup=warmingup,
                         t_total=num_train_optimization_steps)
    print("lr: {} warm: {} total_step: {}".format(args.learning_rate, args.warmup_proportion,
                                                  num_train_optimization_steps))

    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model, device_ids=[0, 1])

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    train_data = processor.build_dataset(train_examples, tokenizer, args.max_seq_length, args.max_tree_length, "train",
                                         args)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batchsize)

    for epoch_num in trange(int(epoch), desc="Epoch"):
        optimizer.zero_grad()
        model.zero_grad()
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_iter = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_iter):
            if args.max_steps > 0 and global_step > args.max_steps:
                break
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, valid_ids, segment_ids, label_ids, e1_mask, e2_mask, dep_type_matrix, tree, node_order, edge_order, tree_type, finlist, treerev, node_orderrev, edge_orderrev, dep_type, dep_adj = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, e1_mask=e1_mask, e2_mask=e2_mask,
                         valid_ids=valid_ids, dep_adj_matrix=dep_type_matrix, dep_type_matrix=dep_type_matrix,
                         tree=tree, node_order=node_order, edge_order=edge_order, finlist=finlist, treerev=treerev,
                         noderev=node_orderrev, edgerev=edge_orderrev, dep_type=dep_type, dep_adj=dep_adj)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:

                with torch.autograd.detect_anomaly():
                    loss.backward()

                # scaler.scale(loss).backward()

            # print(model.bert.embeddings.word_embeddings.weight.grad)
            # print(model.classifier.weight.grad)

            if is_main_process():
                train_iter.update(1)
                perplexity = torch.exp(torch.tensor(loss))
                train_iter.set_postfix_str(f"Step: {global_step} Loss: {loss:.5f} ppl: {perplexity:.5f}")

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                global_step += 1

        if args.local_rank == -1 or torch.distributed.get_rank() == 0 or args.world_size <= 1:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "epoch-{}".format(epoch_num))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_zen_model(output_dir, model, processor, tokenizer)

            # eval dev
        result = evaluate(args, model, tokenizer, processor, device, mode="test")
        a = result["f1"]
        print(a)
        trial.report(a, epoch_num)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return a
def train_func(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.distributed.init_process_group(backend="gloo")
    # args.local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(args.local_rank)
    # gpus = [0, 1, 2, 3]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))

    # args.device = device = torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)
    # device = torch.device(args.local_rank)
    # device = torch.device("cuda", args.local_rank)
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    processor = RE_Processor(dep_type=args.dep_type)
    processor.prepare_type_dict(args.data_dir)
    processor.prepare_labels_dict(args.data_dir)
    processor.prepare_tree_type_dict(args.data_dir)
    label_list = processor.labels_dict.keys()
    dep_type_list = processor.types_dict.keys()
    tree_labels=processor.tree_node_type.keys()
    num_labels = len(label_list)
    type_num = len(dep_type_list)
    tree_labels_len = len(tree_labels)

    if args.vocab_file is None:
        args.vocab_file = os.path.join(args.model_path, VOCAB_NAME)
    print("LOAD tokenizer from", args.vocab_file)
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=args.max_seq_length)
    tokenizer.add_never_split_tokens(["<e1>","</e1>","<e2>","</e2>"])
    print("LOAD CHECKPOINT from", args.model_path)
    config = BertConfig.from_json_file(os.path.join(args.model_path, "config.json"))
    config.__dict__["num_gcn_layers"] = args.num_gcn_layers
    config.__dict__["num_labels"] = num_labels
    config.__dict__["learning_rate"] = args.learning_rate
    config.__dict__["batch_size"]=args.train_batch_size
    config.__dict__["type_num"] = type_num
    config.__dict__["dep_type"] = args.dep_type
    config.__dict__["tree_labels_len"]=tree_labels_len
    model = ReAgcn(config)
    model.bert=BertModel.from_pretrained(args.model_path)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[args.local_rank])
    #
    #
    # model._set_static_graph()

    train(args, model, tokenizer, processor, device, args.n_gpu)


def test_func(args):
    args.device = device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    if args.vocab_file is None:
        args.vocab_file = os.path.join(args.model_path, VOCAB_NAME)
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=args.max_seq_length)
    tokenizer.add_never_split_tokens(["<e1>","</e1>","<e2>","</e2>"])
    config = BertConfig.from_json_file(os.path.join(args.model_path, "config.json"))

    model=ReAgcn(config)
    model = model.from_pretrained(args.model_path)

    dict_bin = torch.load(os.path.join(args.model_path, "dict.bin"))
    # del dict_bin['labels_dict']['other']
    # dict_bin['labels_dict']['Other']=6
    # print(dict_bin['labels_dict'])
    processor = RE_Processor(dep_type=config.dep_type, types_dict=dict_bin["types_dict"], labels_dict=dict_bin["labels_dict"],tree_node_type=dict_bin["tree_type_dict"])
    model.cuda('cuda:0')
    result = evaluate(args, model, tokenizer, processor, device, mode="test")
    logger.info(result)

def predict_func(args):
    pass

def main():
    args = get_args()
    if args.do_train:
        train_func(args)
    elif args.do_test:
        test_func(args)
    elif args.do_predict:
        predict_func(args)

if __name__ == "__main__":
    # main()
    study = optuna.create_study(direction="maximize",pruner=optuna.pruners.ThresholdPruner(75))
    study.optimize(objective, n_trials=200)
    plot_optimization_history(study).show()
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
