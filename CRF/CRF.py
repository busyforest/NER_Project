import sklearn_crfsuite
import joblib
import logging


def load_data(file_path):
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                features = parts[3:-1]  # 特征从第4列开始到倒数第2列
                label = parts[-1]
                feature_dict = {f.split(':')[0]: f.split(':')[1] for f in features}
                sentence.append((word, feature_dict, label))
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences


def prepare_data(sentences):
    X = []
    y = []
    for sentence in sentences:
        X.append([features for _, features, _ in sentence])
        y.append([label for _, _, label in sentence])
    return X, y


# 配置日志
logging.basicConfig(level=logging.INFO)

# 加载训练数据
train_sentences = load_data('NER/Chinese/train_with_template.txt')
X_train, y_train = prepare_data(train_sentences)

# 训练CRF模型
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.0005,
    c2=0.0005,
    max_iterations=60,
    all_possible_transitions=False,
    verbose=True
)

crf.fit(X_train, y_train)

# 保存模型
joblib.dump(crf, 'crf_model.joblib')

# 加载测试数据
test_sentences = load_data('NER/Chinese/test_with_template.txt')
X_test, y_test = prepare_data(test_sentences)

# 进行预测
y_pred = crf.predict(X_test)
with open('NER/Chinese/output.txt', 'w', encoding='utf-8') as f:
    for sentence, pred_labels in zip(test_sentences, y_pred):
        for (word, _, true_label), pred_label in zip(sentence, pred_labels):
            f.write(f"{word} {pred_label}\n")
        f.write("\n")

