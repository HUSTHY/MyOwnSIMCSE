import pandas as pd
import numpy as np
from  dataReader import DataReader
from transformers import BertTokenizer
from transformers import BertConfig,BertModel
import torch
from torch.utils.data import DataLoader
from SimCSE import SimCSE
from tqdm import tqdm

import torch.nn.functional as F

def alltexts2vecs(simcse,normal_model,tokenizer,device):
    simcse_embeddings = []
    normal_embeddings = []

    normal_model.eval()
    normal_model.to(device)
    simcse.eval()
    simcse.to(device)

    data_files = 'data/weikong/train_weikong_dataset.xlsx'
    dataset = DataReader(tokenizer, data_files, 100)
    dl = DataLoader(dataset=dataset, batch_size=32 , shuffle=False)
    with torch.no_grad():
        for data in tqdm(dl, ncols=50):
            output = simcse(input_ids=data["input_ids"].to(device),
                         attention_mask=data["attention_mask"].to(device),
                         token_type_ids=data["token_type_ids"].to(device))
            pred = output
            vec = pred.cpu().detach().numpy()
            simcse_embeddings.extend(vec)


        for data in tqdm(dl, ncols=50):
            output = normal_model(input_ids=data["input_ids"].to(device),
                         attention_mask=data["attention_mask"].to(device),
                         token_type_ids=data["token_type_ids"].to(device),return_dict=True, output_hidden_states=True)
            pred = output.hidden_states[-1].mean(dim=1)
            vec = pred.cpu().detach().numpy()
            normal_embeddings.extend(vec)

    simcse_embeddings = np.array(simcse_embeddings)
    normal_embeddings = np.array(normal_embeddings)

    simcse_embeddings = torch.from_numpy(simcse_embeddings).to(device)
    normal_embeddings = torch.from_numpy(normal_embeddings).to(device)

    return simcse_embeddings,normal_embeddings


def compute_cossim_topk(query_emebdding,inside_embeddings):
    d = torch.mul(query_emebdding, inside_embeddings)  # 计算对应元素相乘
    a_len = torch.norm(query_emebdding, dim=1)  # 2范数，也就是模长
    b_len = torch.norm(inside_embeddings, dim=1)
    cos = torch.sum(d, dim=1) / (a_len * b_len)  # 得到相似度
    topk = torch.topk(cos,100)
    return topk



def query2vec(text,bert,tokenizer,device,max_len):
    text = text[0:max_len-2]
    inputs = tokenizer(text)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    paddings = [0] * (max_len - len(input_ids))
    input_ids += paddings
    attention_mask += paddings

    input_ids = torch.as_tensor(input_ids, dtype=torch.long).to(device)
    attention_mask = torch.as_tensor(attention_mask, dtype=torch.long).to(device)


    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    token_type_ids = torch.zeros_like(input_ids).to(device)
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids}
    bert.to(device)
    bert.eval()

    with torch.no_grad():

        if bert.__module__ =='SimCSE':
            output = bert(**inputs)
            pred = output
        else:
            output = bert(**inputs,return_dict=True, output_hidden_states=True)
            pred = output.hidden_states[-1].mean(dim=1)

    return pred



def query_searchs():
    query_df = pd.read_excel('data/weikong/test_data0824.xlsx')
    print(len(query_df))
    query_df.drop_duplicates(inplace=True)
    print(len(query_df))

    df = pd.read_excel('data/weikong/train_weikong_dataset.xlsx')

    texts = df['question'].values.tolist()
    querys = query_df['question'].values.tolist()

    normal_recommends = []
    normal_querys = []
    normal_simi = []


    simcse_querys = []
    simcse_recommends = []
    simcse_simi = []



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # simcse_path = './output/0824_roberta_epoch_0-batch_864-loss_0.035382'
    simcse_path = './output/0824_bert_epoch_0-batch_864-loss_0.014020'
    tokenizer = BertTokenizer.from_pretrained(simcse_path, mirror="tuna")
    conf = BertConfig.from_pretrained(simcse_path)
    simcse = SimCSE(conf, simcse_path)


    # normal_model_path = './pretrain_models/chinese-roberta-wwm-ext'
    normal_model_path = './pretrain_models/chinese-bert-wwm-ext'

    bert = BertModel.from_pretrained(normal_model_path)

    simcse_embeddings, normal_embeddings = alltexts2vecs(simcse=simcse,normal_model=bert,tokenizer=tokenizer,device=device)

    max_len = 64
    for query in tqdm(querys,desc='simcse recommend',ncols=50):
        q_vec  = query2vec(text=query,bert=simcse,tokenizer=tokenizer,device=device,max_len=max_len)
        simcse_topk = compute_cossim_topk(q_vec,simcse_embeddings)
        indexs = simcse_topk.indices.data.tolist()
        sims = simcse_topk.values.data.tolist()
        for index, sim in zip(indexs, sims):
            simcse_querys.append(query)
            simcse_recommends.append(texts[index])
            simcse_simi.append(sim)

        simcse_querys.append('')
        simcse_recommends.append('')
        simcse_simi.append('')

    for query in tqdm(querys,desc='bert recommend',ncols=50):
        q_vec  = query2vec(text=query,bert=bert,tokenizer=tokenizer,device=device,max_len=max_len)
        normal_topk = compute_cossim_topk(q_vec,normal_embeddings)

        indexs = normal_topk.indices.data.tolist()
        sims = normal_topk.values.data.tolist()
        for index, sim in zip(indexs, sims):
            normal_querys.append(query)
            normal_recommends.append(texts[index])
            normal_simi.append(sim)

        normal_querys.append('')
        normal_recommends.append('')
        normal_simi.append('')


    normal_df = pd.DataFrame()
    normal_df['query'] = normal_querys
    normal_df['recommend'] = normal_recommends
    normal_df['simi'] = normal_simi
    # normal_writer = pd.ExcelWriter('output/weikong/normal_result_roberta.xlsx')
    normal_writer = pd.ExcelWriter('output/weikong/normal_result_bert.xlsx')
    normal_df.to_excel(normal_writer,index=False)
    normal_writer.save()

    simcse_df = pd.DataFrame()
    simcse_df['query'] = simcse_querys
    simcse_df['recommend'] = simcse_recommends
    simcse_df['simi'] = simcse_simi
    # simcse_writer = pd.ExcelWriter('output/weikong/simcse_result_roberta.xlsx')
    simcse_writer = pd.ExcelWriter('output/weikong/simcse_result_bert.xlsx')
    simcse_df.to_excel(simcse_writer, index=False)
    simcse_writer.save()





def compute_similarity(texta,textb):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # simcse_path = './output/0824_roberta_epoch_0-batch_864-loss_0.035382'
    simcse_path = './output/0824_bert_epoch_0-batch_864-loss_0.014020'
    tokenizer = BertTokenizer.from_pretrained(simcse_path, mirror="tuna")
    conf = BertConfig.from_pretrained(simcse_path)
    simcse = SimCSE(conf, simcse_path)
    max_len = 64
    embedding_a = query2vec(texta, simcse, tokenizer, device, max_len)
    embedding_b = query2vec(textb, simcse, tokenizer, device, max_len)

    d = torch.mul(embedding_a, embedding_b)  # 计算对应元素相乘
    a_len = torch.norm(embedding_a, dim=1)  # 2范数，也就是模长
    b_len = torch.norm(embedding_b, dim=1)
    cos = torch.sum(d, dim=1) / (a_len * b_len)  # 得到相似度
    print('%s ------%s  simcse similarity %s'%(texta,textb,cos))

    print('*'*100)

    normal_model_path = './pretrain_models/chinese-bert-wwm-ext'
    bert = BertModel.from_pretrained(normal_model_path)
    embedding_a = query2vec(texta, bert, tokenizer, device, max_len)
    embedding_b = query2vec(textb, bert, tokenizer, device, max_len)

    d = torch.mul(embedding_a, embedding_b)  # 计算对应元素相乘
    a_len = torch.norm(embedding_a, dim=1)  # 2范数，也就是模长
    b_len = torch.norm(embedding_b, dim=1)
    cos = torch.sum(d, dim=1) / (a_len * b_len)  # 得到相似度
    print('%s ------%s  bert similarity %s'%(texta,textb,cos))

if __name__ == '__main__':
    # query_searchs()

    texta = '支付好了'
    textb = '在哪支付'
    compute_similarity(texta, textb)



















