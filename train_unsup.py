# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka


import argparse
import logging
import os
from pathlib import Path
from transformers import BertConfig, BertModel
# from datasets import load_dataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from SimCSE import SimCSE
from CSECollator import CSECollator
from dataReader import DataReader
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str,default='./data/weikong/train_weikong_dataset.xlsx', help="train text file")
    parser.add_argument("--pretrained", type=str, default="./pretrain_models/chinese-bert-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="./output", help="model output path")
    parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=64, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--display_interval", type=int, default=100, help="display interval")
    parser.add_argument("--save_interval", type=int, default=860, help="save interval")
    parser.add_argument("--pool_type", type=str, default="cls", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    args = parser.parse_args()
    return args

    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--train_file", type=str, help="train text file")
    # args = parser.parse_args()
    # return args



def load_data(args, tokenizer):
    # data_files = {"train": args.train_file}
    # ds = load_dataset("text", data_files=data_files)
    #
    # ds_tokenized = ds.map(lambda example: tokenizer(example["text"]), num_proc=args.num_proc)
    # collator = CSECollator(tokenizer, max_len=args.max_length)
    # dl = DataLoader(ds_tokenized["train"],
    #                 batch_size=args.batch_size,
    #                 collate_fn=collator.collate)

    data_files = args.train_file
    dataset = DataReader(tokenizer,data_files,100)
    collator = CSECollator(tokenizer, max_len=args.max_length)
    dl = DataLoader(dataset=dataset,collate_fn=collator.collate,batch_size=args.batch_size,shuffle=False)
    return dl



def compute_infoceLoss(y_pred, tao=0.05, device="cuda"):
    """

    :param y_pred: 模型输出，维度[B,H]
    :param tao: 温度系数
    :param device:
    :return:
    """
    idxs = torch.arange(0, y_pred.shape[0], device=device)

    y_true = idxs + 1 - idxs % 2 * 2

    t1 = time.time()
    #[B,1,H]
    a = y_pred.unsqueeze(1)
    # [1,B,H]
    b = y_pred.unsqueeze(0)
    #[B,B]
    similarities = F.cosine_similarity( a, b, dim=2)
    t2 = time.time()
    print('time is %.4f' % (t2 - t1)) #cpu情况下 B=64 time is 0.2021

    t1 = time.time()
    #自己实现的cos——similarity计算，貌似比torch.cosine_similarity()要快
    #[B,H]
    a_new = y_pred
    #[H,B]
    b_new = y_pred.T
    #[B,B]
    d = torch.matmul(a_new,b_new)
    # [B,B]
    length = torch.mm(torch.norm(a_new,dim=1).unsqueeze(1),torch.norm(b_new,dim=0).unsqueeze(0))
    cos = d/length
    t2 = time.time()
    print('time is %.4f'%(t2-t1)) #cpu情况下 B=64 time is 0.0348

    #单位对角矩阵——对角线上为1e12很大的值
    c = torch.eye(y_pred.shape[0], device=device) * 1e12

    # 单位对角矩阵——对角线上为1-1e12很小的值
    similarities = similarities - c

    similarities = similarities / tao

    loss = F.cross_entropy(similarities, y_true)

    return torch.mean(loss)


def train(args):
    # args.device  = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args.device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained, mirror="tuna")
    dl = load_data(args, tokenizer)

    conf = BertConfig.from_pretrained(args.pretrained)
    model = SimCSE(conf,args.pretrained, args.pool_type, args.dropout_rate).to(args.device)



    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model_out = Path(args.model_out)
    if not model_out.exists():
        os.mkdir(model_out)

    model.train()
    batch_idx = 0
    for epoch_idx in range(args.epochs):
        for data in tqdm(dl,ncols=50):
            batch_idx += 1

            pred = model(input_ids=data["input_ids"].to(args.device),
                         attention_mask=data["attention_mask"].to(args.device),
                         token_type_ids=data["token_type_ids"].to(args.device))

            loss = compute_infoceLoss(pred, args.tao, args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()

            if batch_idx % args.display_interval == 0:
                logging.info(f"batch_idx: {batch_idx}, loss: {loss:>10f}")

        save_path = model_out / "0824_bert_epoch_{0}-batch_{1}-loss_{2:.6f}".format(epoch_idx, batch_idx, loss)
        model.save_pretrained(
            save_path)
        tokenizer.save_vocabulary(
            save_path)
        conf.architectures =  ["BertForMaskedLM"]
        conf.save_pretrained(save_path)

    conf = BertConfig.from_pretrained(save_path)
    simcse = SimCSE(conf,save_path)




def main():
    args = parse_args()
    print('args',args)
    train(args)


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
