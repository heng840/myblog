# Tokenizer
分词结果：比如：
```
['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`','a', '`', 'Ġand', 'Ġ`', 'b', '`."""', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```
This tokenizer has a few special symbols, like Ġ and Ċ, which denote spaces and newlines, respectively.

### fast tokenizer
这是AutoTokenizer默认的设置
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))
```
这样得到一个BatchEncoding 。有如下属性：encoding.tokens()、encoding.word_ids()，word_to_chars等方法，可以对应会原来文本的起始和终止位置

### pipeline方法和它的原始的实现过程
#### pipeline方法
```jupyterpython
from transformers import pipeline

token_classifier = pipeline("token-classification")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

# print:
[{'entity': 'I-PER', 'score': 0.9993828, 'index': 4, 'word': 'S', 'start': 11, 'end': 12},
 {'entity': 'I-PER', 'score': 0.99815476, 'index': 5, 'word': '##yl', 'start': 12, 'end': 14},
 {'entity': 'I-PER', 'score': 0.99590725, 'index': 6, 'word': '##va', 'start': 14, 'end': 16},
 {'entity': 'I-PER', 'score': 0.9992327, 'index': 7, 'word': '##in', 'start': 16, 'end': 18},
 {'entity': 'I-ORG', 'score': 0.97389334, 'index': 12, 'word': 'Hu', 'start': 33, 'end': 35},
 {'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging', 'start': 35, 'end': 40},
 {'entity': 'I-ORG', 'score': 0.98879766, 'index': 14, 'word': 'Face', 'start': 41, 'end': 45},
 {'entity': 'I-LOC', 'score': 0.99321055, 'index': 16, 'word': 'Brooklyn', 'start': 49, 'end': 57}]

# 加入aggregation_strategy方法，
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
# print
[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18},
 {'entity_group': 'ORG', 'score': 0.97960204, 'word': 'Hugging Face', 'start': 33, 'end': 45},
 {'entity_group': 'LOC', 'score': 0.99321055, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

那么如何从最开始的方法，手动的处理？
#### From inputs to predictions
1. 需要调用： model， tokenizer。
2. 需要自定义predict函数
```python
import torch

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)
```
3. 需要用上id2label，默认：
```
{0: 'O',
 1: 'B-MISC',
 2: 'I-MISC',
 3: 'B-PER',
 4: 'I-PER',
 5: 'B-ORG',
 6: 'I-ORG',
 7: 'B-LOC',
 8: 'I-LOC'}
```
未aggregate前，会希望类似：Hu是B-开始的，而##gging是I-。
4. 需要使用return_offsets_mapping，获取offset
5. 需要自定义group函数聚合tokens，用上offset,获取start和end

## QA问题
使用模型，输入是：context+question，输出：在context里检索，得到start和end，从而得到answer

### use the model 
-> 获取start和end，主要是数学上处理很复杂

### handle long contexts
```python
inputs = tokenizer(
    sentence, 
    truncation=True, 
    return_overflowing_tokens=True,  # 核心在这里，将sentence进行chunk处理
    max_length=6, 
    stride=2
)

# 实际使用时：
inputs = tokenizer(
    question,
    long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

```
这样得到：
```python
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)
->
torch.Size([2, 384]) torch.Size([2, 384])
这意味着两个logits，长度为384（max_length),接下来分批处理即可
```

## Normalization and pre-tokenization
### Normalization
### pre-tokenization
预标记化，不同分词器的方法不同。
### SentencePiece
1. It considers the text as a sequence of Unicode characters, and replaces spaces with a special character, ▁.
2. reversible 

## Byte-Pair Encoding tokenization
GPT family


