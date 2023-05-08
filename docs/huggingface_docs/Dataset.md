## Dataset Library
Core:   ```from datasets import load_dataset```

```python
from datasets import load_dataset

my_data = load_dataset()
my_data.map()
```
- 查看数据库属性

A good practice when doing any sort of data analysis is to grab a small random sample to get a quick feel for the type of data you’re working with. In 🤗 Datasets, we can create a random sample by chaining the Dataset.shuffle() and Dataset.select() functions together:
```python
drug_sample = my_data["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
drug_sample[:3]
```
#### 可以使用的方法
1. rename_column
重命名属性
2. filter
过滤为None的值，或者一些其他的要求
3. add_column
新增一个属性。也可以用map代替
4. map


### big data -hard to load ->memory-mapped file
这是Dataset自动使用的方法，可以降低RAM的使用。对于一般的dataset，这样的降低内存的方法完全足够了

但是如果dataset过于巨大，那么可以使用streaming=True的方法，动态下载数据集。
```python
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)
```
这样，会返回一个IterableDataset ，输出是逐个返回的。因此需要用上：
```python
next(iter(tokenized_dataset))
```





