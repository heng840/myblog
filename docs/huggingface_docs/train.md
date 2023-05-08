
#### Transformers are language models
 预训练+微调

All the Transformer models mentioned above (GPT, BERT, BART, T5, etc.) have been trained as language models. 
This means they have been trained on large amounts of raw text in a **self-supervised fashion**. 
Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model. 
That means that humans are not needed to label the data!

This type of model develops a statistical understanding of the language it has been trained on,
but it’s not very useful for specific practical tasks. 
Because of this, the general pretrained model then goes through a process called **transfer learning**. 
During this process, the model is fine-tuned in a **supervised way** — that is, using human-annotated labels — on a given task.


#### 三种典型框架
The model is primarily composed of two blocks:

- Encoder : The encoder receives an input and builds a representation of it (its features). 
This means that the model is optimized to acquire understanding from the input.
- Decoder : The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence.
This means that the model is optimized for generating outputs.

Each of these parts can be used independently, depending on the task:

- Encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
Encoder models use only the encoder of a Transformer model. At each stage, the attention layers can access _**all the words in the initial sentence**_. 
These models are often characterized as having “bi-directional” attention, and are often called auto-encoding models.

The pretraining of these models usually revolves around somehow corrupting a given sentence
(for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence.

Encoder models are best suited for tasks requiring an understanding of the full sentence, such as sentence classification,
named entity recognition (and more generally word classification), and extractive question answering.
- Decoder-only models: Good for generative tasks such as text generation.
Decoder models use only the decoder of a Transformer model. 
At each stage, for a given word the attention layers _**can only access the words positioned before it in the sentence**_.
These models are often called auto-regressive models.

The pretraining of decoder models usually revolves around predicting the next word in the sentence.

These models are best suited for tasks involving text generation.

- Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.
Encoder-decoder models (also called sequence-to-sequence models) use both parts of the Transformer architecture. At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input.

The pretraining of these models can be done using the objectives of encoder or decoder models, but usually involves something a bit more complex. For instance, T5 is pretrained by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is then to predict the text that this mask word replaces.

Sequence-to-sequence models are best suited for tasks revolving around generating new sentences depending on a given input, such as summarization, translation, or generative question answering
#### Attention layers

this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.

## Using transformers

### Preprocessing with a tokenizer
- Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
- Mapping each token to an integer
- Adding additional inputs that may be useful to the model

```python
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
```
The output itself is a dictionary containing two keys, input_ids and attention_mask.
input_ids contains two rows of integers (one for each sentence) that are the unique identifiers of the tokens in each sentence. 
 given some inputs, it outputs what we’ll call hidden states, also known as features.
### Postprocessing the output
softmax
```python
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

## Finetune
### Process data
#### load dataset
```python
from datasets import load_dataset
```

 example: MRPC (Microsoft Research Paraphrase Corpus) dataset, 
 The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing). 

这样返回的结果：

DatasetDict

#### Preprocessing a dataset
需要将sentence1、sentence2作为pairs输入。

```jupyter
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs如下：
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```
attention_mask ：和padding配合使用
token_type_ids：新的变量，标记了哪些是sentence1，哪些是2.如果decode后会在中间用[SEP]隔开
In general, you don’t need to worry about whether or not there are token_type_ids in your tokenized inputs: 
as long as you use the same checkpoint for the tokenizer and the model, 
everything will be fine as the tokenizer knows what to provide to its model.

**_Note！_**

如何创建一个tokenize_dataset？ use the Dataset.map() method.

原始代码:
```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```
问题在于，这样需要一次性处理大量列表，需要很多内存。因此改为使用：
```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(function=tokenize_function, 
                                      batched=True)
```
这样就将原来的文本数据集转化为了tokenize化后的数据集。（新增三个特征）

features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids']
#### Dynamic padding
pad all the examples to the length of the longest element when we batch elements together — a technique we refer to as dynamic padding.
```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
DataCollatorWithPadding在稍后的训练步骤中起作用，而不是在这里立即对tokenized_datasets进行动态填充。

DataCollatorWithPadding实际上是一个处理批次数据的工具，它将在数据加载器（DataLoader）创建时被传入。当数据加载器从数据集中抽取数据批次并准备输入模型时，DataCollatorWithPadding会根据当前批次的最大序列长度进行动态填充。
用法：1：使用Trainer包装
```python
from transformers import Trainer
from transformers import TrainingArguments
training_args = TrainingArguments("test-trainer")
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```
2：
```python
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True, collate_fn=data_collator)
```

### use Trainer API
```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
```
上面的：training_args是超参数，compute_metrics是评价函数。
#### evaluate
compare those preds to the labels. To build our compute_metric() function, we will rely on the metrics from the Evaluate library.
```python
import evaluate

metric = evaluate.load()
```
这样便实例化了metric，在[这里](https://huggingface.co/evaluate-metric)查看load支持的方法。

- 实际上原来使用了 ```from datasets import load_metric```，现在迁移到了evaluate。

实例化后， 使用metric.compute。




















## Accelerate
Accelerate is a library that enables the same PyTorch code to be run across any distributed configuration by adding just four lines of code! In short, training and inference at scale made simple, efficient and adaptable.
 #### core code
```python
from accelerate import Accelerator
accelerator = Accelerator()

model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)
    accelerator.backward(loss)
```

#### 一个例子，从原始代码转变到Accelerate
```python
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from accelerate import Accelerator  # ++
accelerator = Accelerator() # ++

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer(["Hello, I'm a single sentence!", "And another sentence"], padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor([1, 0])
dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # --
model.to(device) # --
model, dataloader = accelerator.prepare(model, dataloader) # ++

optimizer = AdamW(model.parameters())

for epoch in range(3):
    for batch in dataloader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch] #--
        input_ids, attention_mask, labels = batch # ++
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward() # --
        accelerator.backward(loss) # ++
        optimizer.step()
        optimizer.zero_grad()

print("Training without Accelerate is done.")

```
#### 在使用Accelerate库时，可以利用其内置的分布式训练功能来支持单机多卡的训练。
要在多张GPU上训练，你只需使用accelerate命令行工具配置训练设置，然后稍微调整你的训练代码即可。
```bash
accelerate launch --config_file=accelerate_config.yml --num_processes=5 your_training_script.py
```
关键在num_processes，这样可以使用5张GPU

示例：加上deepspeed_plugin的使用，更好的编辑超参数

首先，在你的项目目录下创建一个名为accelerate_config.yml的配置文件，内容如下：
```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config: ds_config.json

```
然后，在同一个目录下创建一个名为ds_config.json的DeepSpeed配置文件，内容如下:
```json
{
  "train_batch_size": 10,
  "gradient_accumulation_steps": 1,
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "last_batch_iteration": -1,
      "total_num_steps": 500,
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 100
    }
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "cpu_offload": true
  },
  "fp16": {
    "enabled": true
  },
  "amp": {
    "enabled": false
  }
}

```

然后：
```python
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from accelerate import Accelerator, DeepSpeedPlugin

# 初始化 Accelerator 和 DeepSpeedPlugin
accelerator = Accelerator(fp16=True, deepspeed_plugin=DeepSpeedPlugin("ds_config.json"))
device = accelerator.device

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer(["Hello, I'm a single sentence!", "And another sentence"], padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor([1, 0])
dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=2)

# 准备模型和数据加载器
model, dataloader = accelerator.prepare(model, dataloader)

optimizer = AdamW(model.parameters())

for epoch in range(3):
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

print("Training with Accelerate on multiple GPUs is done.")

```
