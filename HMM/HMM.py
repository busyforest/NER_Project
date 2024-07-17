from collections import defaultdict

initial_matrix = defaultdict(float)
initial_total = 0
transition_matrix = defaultdict(lambda: defaultdict(float))
transition_total = defaultdict(float)
emission_matrix = defaultdict(lambda: defaultdict(float))
emission_total = defaultdict(float)
zero_num = 0
total = 0


def hmm_learn(file_path):
    words = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) == 2:
                word, label = tokens
                words.append(word)
                labels.append(label)
            else:
                get_initial_data(labels)
                get_transition_data(labels)
                get_emission_data(words, labels)
                words.clear()
                labels.clear()
        compute_matrix()


def get_initial_data(labels):
    global initial_total
    initial_matrix[labels[0]] += 1
    initial_total += 1


def get_transition_data(labels):
    for i in range(len(labels) - 1):
        transition_matrix[labels[i]][labels[i + 1]] += 1
        transition_total[labels[i]] += 1


def get_emission_data(words, labels):
    for i in range(len(words)):
        emission_matrix[labels[i]][words[i]] += 1
        emission_total[labels[i]] += 1


def compute_matrix():
    for i in transition_matrix:
        initial_matrix[i] = (initial_matrix[i] + 1) / initial_total
    for i in transition_matrix:
        for j in transition_matrix:
            transition_matrix[i][j] = (transition_matrix[i][j] + 1) / transition_total[i]
    for i in emission_matrix:
        for j in emission_matrix[i]:
            emission_matrix[i][j] = (emission_matrix[i][j] + 1) / emission_total[i]


def hmm_test(file_path):
    global zero_num
    global total
    words = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) == 2:
                word, label = tokens
                words.append(word)
                labels.append(label)
            else:
                viterbi(words)
                total += 1
                words.clear()
                labels.clear()


def viterbi(words):
    global zero_num
    # 初始化
    prob = defaultdict(lambda: defaultdict(float))
    path = defaultdict(lambda: defaultdict(float))
    T = len(words)
    for i in initial_matrix:
        if emission_matrix[i][words[0]] == 0:
            temp_count = 0
            for j in emission_matrix[i]:
                if emission_matrix[i][j] == 0:
                    temp_count += 1
            avg_emission_prob = 1 / temp_count
            prob[0][i] = initial_matrix[i] * avg_emission_prob * 1e-10
        else:
            prob[0][i] = initial_matrix[i] * emission_matrix[i][words[0]]
        path[0][i] = 0
        if prob[0][i] == 0:
            prob[0][i] = 1e-10
    # 递推
    for t in range(1, T):
        for i in transition_matrix:
            temp = 0
            index = list(initial_matrix.keys())[0]
            for j in transition_matrix:
                if prob[t - 1][j] * transition_matrix[j][i] > temp:
                    temp = prob[t - 1][j] * transition_matrix[j][i]
                    index = j
            if emission_matrix[i][words[t]] == 0:
                temp_count = 0
                for j in emission_matrix[i]:
                    if emission_matrix[i][j] == 0:
                        temp_count += 1
                avg_emission_prob = 1 / temp_count
                prob[t][i] = temp * avg_emission_prob * 1e-10
                # print(words[t], i, temp * avg_emission_prob * 1e-20)
            else:
                prob[t][i] = temp * emission_matrix[i][words[t]]
            path[t][i] = index
            if prob[t][i] == 0:
                print(words[t], i)
                prob[t][i] = 1e-300

    # 终止
    final_prob = 0
    final_path = defaultdict(float)
    for i in transition_matrix:
        if prob[T - 1][i] > final_prob:
            final_prob = prob[T - 1][i]
            final_path[T - 1] = i
    # print(final_prob)
    # if final_prob == 1e-300:
    #     zero_num += 1
    #     for t in range(T - 2, -1, -1):
    #         final_path[t] = "O"
    # else:
    for t in range(T - 2, -1, -1):
        final_path[t] = path[t + 1][final_path[t + 1]]
    # for t in words:
    #     temp = 0
    #     for i in emission_matrix:
    #         if emission_matrix[i][t] != 0:
    #             temp += 1
    #     if temp == 0:
    #         print(t)
    for t in range(T):
        string = words[t] + " " + final_path[t]
        print(string)
    print(final_prob)
    with open("NER/Chinese/result.txt", "a", encoding="UTF-8") as file:
        for t in range(T):
            print(words[t], final_path[t])
            string = words[t] + " " + final_path[t] + "\n"
            file.write(string)
        file.write("\n")


train_path = "NER/Chinese/train.txt"
test_path = "NER/Chinese/validation.txt"
hmm_learn(train_path)
hmm_test(test_path)
# print(zero_num/total)
