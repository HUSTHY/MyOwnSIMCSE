import torch
import torch.nn.functional as f
if __name__ == '__main__':
    # y_pred = torch.randn(4,768)
    #
    # a_new = y_pred
    # b_new = y_pred.T
    #
    # d = torch.matmul(a_new, b_new)
    # # [B,B]
    # length = torch.mm(torch.norm(a_new, dim=1).unsqueeze(1), torch.norm(b_new, dim=0).unsqueeze(0))
    # cos_pos = d / length
    # cos_neg = torch.zeros_like(cos_pos)
    #
    # print(cos_pos.shape)
    # print(cos_neg.shape)
    # cos = torch.cat([cos_pos,cos_neg],dim=1)
    # print(cos.shape)
    # labels = torch.arange(cos.size(0))
    # loss = f.cross_entropy(cos,labels)
    # print(loss)

    a = torch.randn(6,5)
    print(a)
    b = a.view(3,2,5)
    d = b[:,0,:]
    e = b[:,1,:]
    print(b)
    print(d)
    print(e)

    # a = torch.randn(6, 5)
    # print(a)
    # b = a.to('cuda:0')
    # print(b.device)