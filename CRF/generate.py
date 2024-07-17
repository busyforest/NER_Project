def generate_features(input_file, output_file, template_file):
    with open(template_file, 'r', encoding='utf-8') as template:
        templates = [line.strip() for line in template if not line.startswith('#')]

    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = f.read().strip().split('\n\n')

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            lines = sentence.split('\n')
            tokens = [line.split()[0] for line in lines]
            labels = [line.split()[1] for line in lines]
            for i in range(len(tokens)):
                features = []
                for template in templates:
                    feature = template
                    for j in range(-2, 3):
                        if 0 <= i + j < len(tokens):
                            feature = feature.replace(f'%x[{j},0]', tokens[i + j])
                        else:
                            feature = feature.replace(f'%x[{j},0]', 'BOS' if i + j < 0 else 'EOS')
                    features.append(feature)
                f.write(f'{tokens[i]} {" ".join(features)} {labels[i]}\n')
            f.write('\n')


# 调用函数生成特征文件
generate_features('NER/English/train.txt', 'NER/English/train_with_template.txt', 'NER/template_for_crf.utf8')
