import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# TF-IDF + 逻辑回归文本分类器示例
# Term Frequency-Inverse Document Frequency

# ========== 1. 准备训练数据 ==========
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
    "refund",           # 退款
    "refund",
    "cancel_order",     # 取消订单
    "query_order",      # 查询订单
    "query_logistics",  # 查询物流
    "modify_address",   # 修改地址
    "modify_address",
    "refund",
    "query_logistics",
    "refund"
]

# ========== 2. 定义中文分词函数 ==========
def tokenizer(text):
    """使用 jieba 分词，返回词列表"""
    return jieba.lcut(text)   # lcut 返回 list

# ========== 3. 构建 Pipeline：TF-IDF 向量化 + 逻辑回归分类器 ==========
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer=tokenizer,   # 自定义分词器
        token_pattern=None,    # 停用默认的英文正则分词
        lowercase=False        # 中文无需小写
    )),
    ('clf', LogisticRegression())  # 逻辑回归
])

# ========== 4. 训练模型 ==========
pipeline.fit(texts, labels)

# ========== 5. 预测新文本 ==========
new_texts = [
    "我要申请退款",
    "我的快递到哪了",
    "地址写错了帮我改一下",
    "这个商品有问题，想退",
    "哈哈"
]

preds = pipeline.predict(new_texts)

# predict_proba 返回每个类别的概率（行=文本，列=类别，每行和为 1）
probas = pipeline.predict_proba(new_texts)

for text, pred, proba in zip(new_texts, preds, probas):
    max_prob = proba.max()   # 最高概率 = 置信度
    print(f"文本：{text}  →  预测意图：{pred}（置信度 {max_prob:.2f}）")