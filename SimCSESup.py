from transformers import  BertModel
from transformers import BertPreTrainedModel
import torch
class SimCSESup(BertPreTrainedModel):
    def __init__(self,config, pool_type="cls", dropout_prob=0.3,tao=0.05):
        super(SimCSESup,self).__init__(config)
        config.attention_probs_dropout_prob = dropout_prob
        config.hidden_dropout_prob = dropout_prob
        self.tao = tao
        self.bert = BertModel(config)
        assert pool_type in ["cls", "pooler"], "invalid pool_type: %s" % pool_type
        self.pool_type = pool_type

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        #[2B,S,H]
        output = output.last_hidden_state
        #[2B,H]
        output = self.pooling(output,attention_mask)

        b_s = int(output.size(0)/2)

        #batch内前面一半是text_a
        z1 = output[0:b_s, :]
        #后面一半是text_b;text_a和text_b互为正样本对
        z2 = output[b_s:,:]

        #[B,B]
        cos_z1_z2 = self.cossimilarity(z1,z2)
        # [B,B]
        cos_z1_z1 = self.cossimilarity(z1,z1)
        #对角矩阵，对角线为1e12
        c = torch.eye(cos_z1_z1.shape[0], device=cos_z1_z1.device) * 1e12
        cos_z1_z1 = cos_z1_z1-c

        #[B,2B]
        cos = torch.cat([cos_z1_z2,cos_z1_z1],dim=1)/self.tao
        return cos


    def cossimilarity(self,v1,v2):
        """

        :param v1: [B,H]
        :param v2: [B,H]
        :return:
        """
        v2 = v2.T
        d = torch.matmul(v1,v2)
        length = torch.mm(torch.norm(v1,dim=1).unsqueeze(1),torch.norm(v2,dim=0).unsqueeze(0))
        cos = d/length
        return cos


    def pooling(self,token_embeddings,inputs):
        """
        mask平均池化
        :param token_embeddings: [B,S]
        :param input: [B,S,H]
        :return: output_vector [B,H]
        """
        output_vectors = []
        #attention_mask
        attention_mask = inputs['attention_mask']
        #[B,L]------>[B,L,1]------>[B,L,768],矩阵的值是0或者1
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #这里做矩阵点积，就是对元素相乘(序列中padding字符，通过乘以0给去掉了)[B,L,768]
        t = token_embeddings * input_mask_expanded
        #[B,768]
        sum_embeddings = torch.sum(t, 1)

        # [B,768],最大值为seq_len
        sum_mask = input_mask_expanded.sum(1)
        #限定每个元素的最小值是1e-9，保证分母不为0
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        #得到最后的具体embedding的每一个维度的值——元素相除
        output_vectors.append(sum_embeddings / sum_mask)

        #列拼接
        output_vector = torch.cat(output_vectors, 1)

        return  output_vector

    def encoding(self,inputs):
        self.bert.eval()
        with torch.no_grad():
            output = self.bert(**inputs, return_dict=True, output_hidden_states=True)
            embedding = output.hidden_states[-1]
            embedding = self.pooling(embedding, inputs)
        return embedding