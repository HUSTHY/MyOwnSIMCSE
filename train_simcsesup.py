import argparse
import logging
import os
from pathlib import Path
from transformers import BertConfig

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from SimCSESup import SimCSESup
from dataReader_sup import DataReaderSup
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str,default='./data/shanghai_sup/train_2021-0907.xlsx', help="train text file")
    parser.add_argument("--dev_file", type=str, default='./data/shanghai_sup/dev_2021-0907.xlsx',
                        help="train text file")
    parser.add_argument("--pretrained", type=str, default="./pretrain_models/chinese-bert-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="./output", help="model output path")
    parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=64, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=30, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--display_interval", type=int, default=500, help="display interval")
    parser.add_argument("--save_interval", type=int, default=860, help="save interval")
    parser.add_argument("--pool_type", type=str, default="cls", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    args = parser.parse_args()
    return args




def load_data(args, tokenizer):
    train_dataset = DataReaderSup(tokenizer,args.train_file,100)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=False)

    dev_dataset = DataReaderSup(tokenizer, args.dev_file, 100)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_dataloader,dev_dataloader


def train(args):
    args.device  = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    train_dataloader,dev_dataloader = load_data(args, tokenizer)


    conf = BertConfig.from_pretrained(args.pretrained)
    # model = SimCSE(conf,args.pretrained, args.pool_type, args.dropout_rate).to(args.device)
    model = SimCSESup.from_pretrained(pretrained_model_name_or_path=args.pretrained, config=conf).to(args.device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model_out = Path(args.model_out)
    if not model_out.exists():
        os.mkdir(model_out)

    model.train()
    batch_idx = 0
    max_acc = 0
    for epoch_idx in range(args.epochs):
        for batch in tqdm(train_dataloader,ncols=50):
            batch_idx += 1

            batch = [t.to(args.device) for t in batch]
            #比较重要的是——把text_a和text_b合并在一起，在使用infoNCELoss的时候方便计算相似度
            input_ids = torch.cat([batch[0],batch[3]],dim=0)
            attention_mask = torch.cat([batch[1],batch[4]],dim=0)
            token_type_ids = torch.cat([batch[2],batch[5]],dim=0)

            pred = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            labels = torch.arange(pred.size(0)).to(args.device)
            loss = F.cross_entropy(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()

            if batch_idx % args.display_interval == 0:
                logging.info(f"batch_idx: {batch_idx}, loss: {loss:>10f}")


        acc = evaluation(model,dev_dataloader,args)
        if acc>max_acc:
            max_acc = acc
            save_path = os.path.join(model_out, "supvervised")
            model.save_pretrained(
                save_path)
            tokenizer.save_vocabulary(
                save_path)
        logging.info(f"acc: {acc:>10f}, max_acc: {max_acc:>10f}")


def evaluation(model,dev_dataloader,args):
    total = 0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dev_dataloader,ncols=50):
            batch = [t.to(args.device) for t in batch]
            input_ids = torch.cat([batch[0],batch[3]],dim=0)
            attention_mask = torch.cat([batch[1],batch[4]],dim=0)
            token_type_ids = torch.cat([batch[2],batch[5]],dim=0)

            pred = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            labels = torch.arange(pred.size(0)).to(args.device)
            pred = torch.argmax(pred,dim=1)
            correct = (labels==pred).sum()
            total_correct += correct
            total += pred.size(0)

    acc = total_correct/total
    return acc


def main():
    args = parse_args()
    print('args',args)
    train(args)


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
