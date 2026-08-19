# pip install transformers datasets torch scikit-learn

# 假设我们有一些意图分类数据，格式为 (文本, 标签)
texts = [
    "我要退款",
    "这个订单我不想要了",
    "怎么取消订单",
    "我的订单到哪了",
    "查一下物流",
    "地址填错了想修改",
    "帮我改一下收货地址",
    "商品有问题要退货",
    "快递到哪了",
    "申请退款"
]

labels = [
    "refund",
    "refund",
    "cancel_order",
    "query_order",
    "query_logistics",
    "modify_address",
    "modify_address",
    "refund",
    "query_logistics",
    "refund"
]

# 将标签转换为数字索引
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels_encoded = label_encoder.transform(labels)  # 变成 0,1,2...
print("标签编码:", labels_encoded)
num_labels = len(label_encoder.classes_)
print("类别数:", num_labels)

# 加载预训练 BERT 模型和 Tokenizer
# BertForSequenceClassification 会在 BERT 的输出层（[CLS] 向量）后自动添加一个 dropout 和线性分类层，输出维度等于 num_labels
from transformers import BertTokenizer, BertForSequenceClassification

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False
)

# 文本预处理（Tokenization）
# BERT 的 tokenizer 会将文本转换为 input_ids、attention_mask 等。需要统一长度，通常设置最大长度为 64 或 128（意图识别文本一般较短）
max_length = 64

def tokenize_function(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

encodings = tokenize_function(texts)
print(encodings["input_ids"].shape)  # (样本数, max_length)
print(encodings["attention_mask"].shape)

# 创建 Dataset 和 DataLoader
# 我们可以直接使用 HuggingFace 的 Dataset 对象
import torch
from torch.utils.data import Dataset, DataLoader

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 去掉 batch 维度
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# 创建数据集
dataset = IntentDataset(texts, labels_encoded, tokenizer)
print("数据集sample:", dataset[0:5])  # 查看第一个样本的内容
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print("数据集大小:", len(dataset))
print("批次数量:", len(dataloader))

# 设置训练参数并训练
# 使用 HuggingFace Trainer
from transformers import Trainer, TrainingArguments
from datasets import Dataset as HFDataset

# 转换为 HuggingFace Dataset
hf_dataset = HFDataset.from_dict({
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"],
    "labels": labels_encoded
})

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,               # 小数据可以多训练几轮
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=20,
    weight_decay=0.01,
    # logging_dir="./logs",
    logging_steps=5,
    eval_strategy="epoch",             # 每个 epoch 评估一次
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (labels == preds).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    eval_dataset=hf_dataset,          # 简单起见用同一份数据演示
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# 预测新文本
def predict_intent(text):
    model.eval()
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    ).to('cpu')

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()

    intent = label_encoder.inverse_transform([pred])[0]
    return intent

# 测试
print(predict_intent("我要申请退款"))
print(predict_intent("我的快递到哪了"))