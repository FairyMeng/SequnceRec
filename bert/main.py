from pytorch_transformers import BertModel, BertConfig, BertTokenizer
import torch
from torch import nn
import numpy as np
import random

# ——————构造模型——————
class TextNet(nn.Module):
    def __init__(self, code_length):  # code_length为fc映射到的维度大小
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('F:/hhb_code/bert/config.json')    #读取设置的参数
        self.textExtractor = BertModel.from_pretrained('F:/hhb_code/bert/pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    #固定参数
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(10)

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features


textNet = TextNet(code_length=64)

# ——————输入处理——————
tokenizer = BertTokenizer.from_pretrained('F:/hhb_code/bert/vocab.txt')

# texts = ["[CLS] This was a good story, [SEP]",
#          "[CLS] Jim Henson was a puppeteer [SEP]"]
texts = ["[CLS]This game is a family favorite. You use train pieces to make trains to destinations in the United States (ex. Houston to Miami). This game balances strategy and luck nicely which I really enjoy. It can be a pretty long game depending on how many people you play with, but it's great for Family Fun Nights. There is also an app version that is really fun to play.[SEP]"]
tokens, segments, input_masks = [], [], []
for text in texts:#循环次数为句子的数量，[sep]分割
    tokenized_text = tokenizer.tokenize(text)  # 用tokenizer对句子分词,tokenized_text是每个单词
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表，indexed_tokens是每个词在词典vobcab中的序号
    tokens.append(indexed_tokens)
    segments.append([0] * len(indexed_tokens))
    input_masks.append([1] * len(indexed_tokens))


max_len = max([len(single) for single in tokens])  # 最大的句子长度

for j in range(len(tokens)):
    padding = [0] * (max_len - len(tokens[j]))
    tokens[j] += padding
    segments[j] += padding
    input_masks[j] += padding
# segments列表全0，因为只有一个句子1，没有句子2
# input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
# 相当于告诉BertModel不要利用后面0的部分

# 转换成PyTorch tensors
tokens_tensor = torch.tensor(tokens)
segments_tensors = torch.tensor(segments)
input_masks_tensors = torch.tensor(input_masks)

# ——————提取文本特征——————
text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)  # text_hashCodes是一个32-dim文本特征
print(text_hashCodes)
