import pandas as pd
from transformers import BertTokenizer,BertConfig
from SimCSESup import SimCSESup
from SimCSE import SimCSE
import torch
from tqdm import  tqdm
import os
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from dataReaderNew import DataReaderNew



def embedding(dataloader,model,device):
    vectors = []
    for batch in tqdm(dataloader,desc='embedding'):
        batch = [t.to(device) for t in batch]
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
        embedding = model.encoding(inputs)
        vectors.append(embedding)

    vectors = torch.cat(vectors,dim=0)

    return vectors

def cos_sim(a,b):
    d = torch.mul(a, b)  # 计算对应元素相乘
    a_len = torch.norm(a, dim=1)  # 2范数，也就是模长
    b_len = torch.norm(b, dim=1)
    cos = torch.sum(d, dim=1) / (a_len * b_len)  # 得到相似度
    return cos

def  similarity_valdation():
    pretrained = './output/supvervised'
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    config = BertConfig.from_pretrained(pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    simcsesup = SimCSESup.from_pretrained(pretrained_model_name_or_path=pretrained, config=config).to(device)

    pretrained = './output/0824_bert_epoch_0-batch_864-loss_0.014020'
    config = BertConfig.from_pretrained(pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    simcseunsup = SimCSE.from_pretrained(pretrained_model_name_or_path=pretrained, config=config).to(device)


    df = pd.read_excel('./data/classification_val_dataset_2w_similarity_prelabels.xlsx')
    texts_a = df['text_a'].values.tolist()
    texts_b = df['text_b'].values.tolist()

    a_dataReader = DataReaderNew(tokenizer=tokenizer,datas=texts_a,max_len=64)
    a_dataLoader = DataLoader(dataset=a_dataReader,shuffle=False,batch_size=64)
    a_embeddings = embedding(model=simcsesup,dataloader=a_dataLoader,device=device)

    b_dataReader = DataReaderNew(tokenizer=tokenizer, datas=texts_b, max_len=64)
    b_dataLoader = DataLoader(dataset=b_dataReader, shuffle=False, batch_size=64)
    b_embeddings = embedding(model=simcsesup, dataloader=b_dataLoader, device=device)

    sup_simlaritys = []
    for a, b in tqdm(zip(a_embeddings, b_embeddings), desc='compute similarity'):
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        sim = cos_sim(a,b)
        sim = sim.detach().cpu().tolist()[0]
        sup_simlaritys.append(sim)



    a_dataReader = DataReaderNew(tokenizer=tokenizer, datas=texts_a, max_len=64)
    a_dataLoader = DataLoader(dataset=a_dataReader, shuffle=False, batch_size=64)
    a_embeddings = embedding(model=simcseunsup, dataloader=a_dataLoader, device=device)

    b_dataReader = DataReaderNew(tokenizer=tokenizer, datas=texts_b, max_len=64)
    b_dataLoader = DataLoader(dataset=b_dataReader, shuffle=False, batch_size=64)
    b_embeddings = embedding(model=simcseunsup, dataloader=b_dataLoader, device=device)

    unsup_simlaritys = []
    for a, b in tqdm(zip(a_embeddings, b_embeddings), desc='compute similarity'):
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        sim = cos_sim(a, b)
        sim = sim.detach().cpu().tolist()[0]
        unsup_simlaritys.append(sim)



    df['simcsesup_similarity'] = sup_simlaritys
    df['simcseunsup_similarity'] = unsup_simlaritys



    writer = pd.ExcelWriter('./output/classification_val_dataset_2W_0831_similarity_simcse_result.xlsx')
    df.to_excel(writer, index=False)
    writer.save()









if __name__ == '__main__':
    similarity_valdation()
    # comutpe_sim()