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

