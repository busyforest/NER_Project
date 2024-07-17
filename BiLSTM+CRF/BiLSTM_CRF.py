from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim

torch.manual_seed(1)

def read_train_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentence = []
        tag = []
        sentences = []
        tags = []
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) == 2:
                word, label = tokens
                sentence.append(word)
                tag.append(label)
            else:
                sentences.append(sentence)
                tags.append(tag)
                sentence = []
                tag = []
    return sentences, tags


def read_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentences = []
        sentence = []
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) == 2:
                word = tokens[0]
                sentence.append(word)
            else:
                sentences.append(sentence)
                sentence = []
    return sentences

def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, to_ix["O"]) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def argmax(vec):
    # 得到最大的值的索引
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]  # max_score的维度为1
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # 维度为1*5
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        # 基本参数
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_set_size = len(tag_to_ix)

        # 非手写层
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden_to_tag = nn.Linear(hidden_dim, self.tag_set_size)

        # 手写CRF内容
        self.transitions = nn.Parameter(torch.rand(self.tag_set_size, self.tag_set_size))
        # 任何标签转移到序列开始标签的概率都是0
        # 序列终止标签转移到任何标签的概率都是0
        self.transitions.data[tag_to_ix[START_TAG], :] = -2147483647
        self.transitions.data[:, tag_to_ix[END_TAG]] = -2147483647
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    # 计算所有路径分数之和，即归一化因子
    def compute_total_score(self, feats):
        init_alphas = torch.full((1, self.tag_set_size), -2147483647)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tag_set_size):
                emission_score = feat[next_tag].view(1, -1).expand(1, self.tag_set_size)
                transition_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + emission_score + transition_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[END_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    # 根据输入的 id 序列得到对应各个标签的得分/emission score/feats
    def get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()  # (h_0, c_0)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden_to_tag(lstm_out)  # len(s)*5
        return lstm_feats

    def compute_sentence_score(self, feats, tags):
        score = torch.zeros(1)
        # 将START_TAG的标签拼接到tag序列最前面
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[END_TAG], tags[-1]]
        return score

    def viterbi_decode(self, feats):
        # 预测序列的得分，维特比解码，输出得分与路径值
        backpointers = []  # 用于存储回溯指针，每个位置保存到达当前tag时前一个tag的索引

        # 初始化前向变量init_vvars，大小为(1, tagset_size)，初始时除了起始标签的概率为0，其余均为-10000
        init_vvars = torch.full((1, self.tag_set_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars  # forward_var保存的是到达当前节点的最优路径得分
        for feat in feats:
            bptrs_t = []  # 当前时间步的回溯指针列表
            viterbivars_t = []  # 当前时间步的路径得分列表

            for next_tag in range(self.tag_set_size):
                # 计算到达next_tag的所有路径的分数
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)  # 找到分数最高的前一个tag的索引
                bptrs_t.append(best_tag_id)  # 保存这个索引
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))  # 保存最高分值

            # 更新forward_var，包含当前时间步每个tag的最高分路径得分
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)  # 保存当前时间步的回溯指针

        # 计算其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[END_TAG]]
        best_tag_id = argmax(terminal_var)  # 找到到达STOP_TAG的最高分的路径
        path_score = terminal_var[0][best_tag_id]  # 记录路径的最高得分

        best_path = [best_tag_id]  # 初始化最优路径，从STOP_TAG开始
        # 通过回溯指针回溯最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # 无需返回最开始的START_TAG位置，移除
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # 确认路径的起点为START_TAG
        best_path.reverse()  # 把从后向前的路径正过来，得到从START到STOP的路径
        return path_score, best_path  # 返回路径得分和最优路径

    def neg_log_likelihood(self, sentence, tags):  # 损失函数
        feats = self.get_lstm_features(sentence)  # len(s)*5
        forward_score = self.compute_total_score(feats)  # 规范化因子/配分函数
        gold_score = self.compute_sentence_score(feats, tags)  # 正确路径得分
        return forward_score - gold_score  # Loss（已取反）

    def forward(self, sentence):
        lstm_feats = self.get_lstm_features(sentence)
        score, tag_seq = self.viterbi_decode(lstm_feats)
        return score, tag_seq


START_TAG = "START"
END_TAG = "END"
embedding_dim = 32
hidden_dim = 256
train_path = "NER/Chinese/train.txt"
test_path = "NER/Chinese/validation.txt"
sentences, tags = read_train_data(train_path)
unique_words = OrderedDict()
for sentence in sentences:
    for word in sentence:
        if word not in unique_words:
            unique_words[word] = len(unique_words)
word_to_ix = {}
for ix, word in enumerate(unique_words):
    word_to_ix[word] = ix

unique_tags = OrderedDict()
for tag in tags:
    for label in tag:
        if label not in unique_tags:
            unique_tags[label] = len(unique_tags)
tag_to_ix = {}
for ix, label in enumerate(unique_tags):
    tag_to_ix[label] = ix
ix_to_tag = {ix: label for label, ix in tag_to_ix.items()}
tag_to_ix[START_TAG] = len(tag_to_ix)
tag_to_ix[END_TAG] = len(tag_to_ix)
# model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim)
# # model.load_state_dict(torch.load('NER/Chinese/model.pth'))
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
# for epoch in range(5):
#     total_loss = 0  # 初始化总loss
#     count = 0  # 初始化计数器
#     for sentence, tag in zip(sentences, tags):
#         model.zero_grad()
#         # 输入
#         sentence_in = prepare_sequence(sentence, word_to_ix)
#         targets = torch.tensor([tag_to_ix[t] for t in tag], dtype=torch.long)
#         # 获取loss
#         loss = model.neg_log_likelihood(sentence_in, targets)
#         print("loss: ", loss[0])
#         total_loss += loss.item()  # 累加loss
#         count += 1  # 更新计数器
#         # 反向传播
#         loss.backward()
#         optimizer.step()
#     # 计算并打印平均loss
#     avg_loss = total_loss / count
#     print("Epoch: %d, Average Loss: %.4f" % (epoch + 1, avg_loss))
#     torch.save(model.state_dict(), 'NER/Chinese/model.pth')
# 假设 BiLSTM_CRF 是我们定义的模型类
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim)
model.load_state_dict(torch.load('NER/Chinese/model.pth'))
sentences_pred = read_test_data(test_path)
for sentence_pred in sentences_pred:
    sentence_pred_in = prepare_sequence(sentence_pred, word_to_ix)
    _,best_path = model(sentence_pred_in)
    for i in range(len(best_path)):
        print(sentence_pred[i], ix_to_tag[best_path[i]])
    print()
with open("NER/Chinese/predictions.txt", 'w', encoding='utf-8') as f:
        for sentence_pred in sentences_pred:
            sentence_pred_in = prepare_sequence(sentence_pred, word_to_ix)
            _, best_path = model(sentence_pred_in)
            for i in range(len(best_path)):
                f.write(f"{sentence_pred[i]} {ix_to_tag[best_path[i]]}\n")
            f.write("\n")  # 每个句子后面加一个空行
