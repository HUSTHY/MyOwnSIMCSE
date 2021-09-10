
from tqdm import tqdm
import torch
import pandas as pd

class DataReaderNew(object):
    def __init__(self,tokenizer,datas,max_len):
        self.tokenizer = tokenizer
        self.datas = datas
        self.max_len = max_len
        self.dataList = self.datas_to_torachTensor()
        self.allLength = len(self.dataList)

    def convert_text2ids(self,text):
        text = text[0:self.max_len-2]
        inputs = self.tokenizer(text)

        input_ids = inputs['input_ids']
        # lenght = len(input_ids)
        attention_mask = inputs['attention_mask']
        paddings = [0] * (self.max_len - len(input_ids))
        input_ids += paddings
        attention_mask += paddings

        token_type_id = [0] * self.max_len

        return input_ids, attention_mask, token_type_id


    def datas_to_torachTensor(self):
        res = []
        for line in tqdm(self.datas,desc='tokenization',ncols=50):
            temp = []
            input_ids, attention_mask, token_type_id = self.convert_text2ids(text=line)
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            token_type_id = torch.as_tensor(token_type_id, dtype=torch.long)
            temp.append(input_ids)
            temp.append(attention_mask)
            temp.append(token_type_id)
            res.append(temp)
        return res

    def __getitem__(self, item):
        input_ids = self.dataList[item][0]
        attention_mask = self.dataList[item][1]
        token_type_id = self.dataList[item][2]
        # return {'input_ids':input_ids,'attention_mask':attention_mask,'token_type_ids':token_type_id}
        return input_ids,attention_mask,token_type_id


    def __len__(self):
        return self.allLength