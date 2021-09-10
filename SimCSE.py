# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka

import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers import BertPreTrainedModel
import torch


# class SimCSE(BertPreTrainedModel):
#     def __init__(self,config,pretrained="hfl/chinese-bert-wwm-ext", pool_type="cls", dropout_prob=0.3):
#         super(SimCSE,self).__init__(config)
#         config.attention_probs_dropout_prob = dropout_prob
#         config.hidden_dropout_prob = dropout_prob
#         self.bert = BertModel.from_pretrained(pretrained, config=config)
#         assert pool_type in ["cls", "pooler"], "invalid pool_type: %s" % pool_type
#         self.pool_type = pool_type
#
#     def forward(self, input_ids, attention_mask, token_type_ids):
#         output = self.bert(input_ids,
#                               attention_mask=attention_mask,
#                               token_type_ids=token_type_ids)
#         if self.pool_type == "cls":
#             output = output.last_hidden_state[:, 0]
#         elif self.pool_type == "pooler":
#             output = output.pooler_output
#         return output
#
#
#     def encoding(self,inputs):
#         self.bert.eval()
#         with torch.no_grad():
#             output = self.bert(**inputs, return_dict=True, output_hidden_states=True)
#             embedding = output.hidden_states[-1]
#             embedding = self.pooling(embedding, inputs)
#         return embedding

class SimCSE(BertPreTrainedModel):
    def __init__(self,config, pool_type="cls", dropout_prob=0.3):
        super(SimCSE,self).__init__(config)
        config.attention_probs_dropout_prob = dropout_prob
        config.hidden_dropout_prob = dropout_prob
        self.bert = BertModel(config=config)
        assert pool_type in ["cls", "pooler"], "invalid pool_type: %s" % pool_type
        self.pool_type = pool_type

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        if self.pool_type == "cls":
            output = output.last_hidden_state[:, 0]
        elif self.pool_type == "pooler":
            output = output.pooler_output
        return output

    def pooling(self, token_embeddings, inputs):
        """
        mask平均池化
        :param token_embeddings: [B,S]
        :param input: [B,S,H]
        :return: output_vector [B,H]
        """
        output_vectors = []
        # attention_mask
        attention_mask = inputs['attention_mask']
        # [B,L]------>[B,L,1]------>[B,L,768],矩阵的值是0或者1
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # 这里做矩阵点积，就是对元素相乘(序列中padding字符，通过乘以0给去掉了)[B,L,768]
        t = token_embeddings * input_mask_expanded
        # [B,768]
        sum_embeddings = torch.sum(t, 1)

        # [B,768],最大值为seq_len
        sum_mask = input_mask_expanded.sum(1)
        # 限定每个元素的最小值是1e-9，保证分母不为0
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # 得到最后的具体embedding的每一个维度的值——元素相除
        output_vectors.append(sum_embeddings / sum_mask)

        # 列拼接
        output_vector = torch.cat(output_vectors, 1)

        return output_vector

    def encoding(self, inputs):
        self.bert.eval()
        with torch.no_grad():
            output = self.bert(**inputs, return_dict=True, output_hidden_states=True)
            embedding = output.hidden_states[-1]
            embedding = self.pooling(embedding, inputs)
        return embedding