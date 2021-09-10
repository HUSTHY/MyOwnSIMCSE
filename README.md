![](https://img.shields.io/badge/Python-3.7.5-blue)
![](https://img.shields.io/badge/torch-1.8.0-green)
![](https://img.shields.io/badge/transformers-4.5.1-green)
![](https://img.shields.io/badge/tqdm-4.49.0-green)

<h3 align="center">
<p>A PyTorch implementation of unsupervised SimCSE and supervised </p>
</h3>

[SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)

---

### 1. 用法

#### 无监督训练 
```bash
python train_unsup.py 
```

#### 有监督训练 
```bash
python train_simcsesup.py 
```

#### 相似文验证
```bash
python similarity_valdation.py
```

####最新结果对比
```
best_sbert_threshold: 0.7600 -------best_sbert_acc:0.9970
best_simcsesup_threshold: 0.5900 -------best_simcsesup_acc:0.9889
best_simcseunsup_threshold: 0.4200 -------best_simcseunsup_acc:0.8865
```



### 2. 参考
- [SimCSE](https://github.com/princeton-nlp/SimCSE)
- [SimCSE-Chinese](https://github.com/zhengyanzhao1997/NLP-model/tree/main/model/model/Torch_model/SimCSE-Chinese)
- [SIMCSE_unsup](https://github.com/KwangKa/SIMCSE_unsup)
