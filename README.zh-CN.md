# rag-hybrid-retrieval

[English](README.md) | 简体中文

基于合同语料的引用溯源式文档问答。BM25 + 向量混合检索、按条款边界切块、默认拒答，以及一个能复现本文所有数字的离线评测脚本。

零运行时依赖。Python 3.10+。

```bash
python cli.py index
python cli.py ask "How much notice is required to terminate for convenience?"
python cli.py eval
```

---

## 这个项目解决什么问题

一家法律科技客户有约 12 万份内部合同，客服团队一天大部分时间都在照着这些合同回答重复问题。他们最初的做法是一个朴素的 RAG demo：把每个 512 token 的窗口做 embedding，余弦相似度取 top-k，塞进 prompt。

它失败的方式很具体，也很贵：**它会非常自信地搞错条款编号。** 一份服务协议里，两条条款可能读起来几乎一样，含义却完全相反 —— 上一条写客户可以随时无理由终止，下一条写不得无因终止。向量相似度区分不了这两者。它们的余弦距离极小，但法律意义上的差别是根本性的。

那次失败催生了三个设计决定：

1. **按条款边界切块，而不是按 token 窗口。** 引用必须能落到「第 4 条」，而不是「第 4 节里某个 512 token 的窗口」。
2. **保留精确的词法匹配（BM25）。** 当问题说的是 "notice period"、条款写的也是 "notice period" 时，词法匹配直接赢。向量检索反而会稀释这个信号。
3. **没有内容越过分数下限时，拒答。** 拒答的代价很低。一个带着引用、但引用是错的答案，代价不低。

---

## 实测结果

评测集：22 个问题，覆盖 3 份文档（MSA、双向 NDA、GDPR DPA）中的 20 条条款。18 个是域内问题、带标注的正确条款；4 个是域外问题，必须拒答。

| 检索方式 | recall@1 | recall@3 | recall@5 | MRR@10 | 拒答率 |
|---|---|---|---|---|---|
| 仅 BM25 (1.0 / 0.0) | 0.6818 | 0.7727 | 0.8182 | 0.7386 | 1.00 |
| 仅向量 (0.0 / 1.0) | 0.7273 | 0.7727 | 0.7727 | 0.7500 | 1.00 |
| **混合 (0.4 / 0.6)** | **0.7727** | **0.7727** | 0.7727 | **0.7778** | 1.00 |

用 `python cli.py eval` 可复现；权重扫描见 `rag.evaluate.sweep`。

关于诚实性，有两点必须说明 —— 因为这些数字很容易被夸大：

- **这里的 embedding 后端是字符 n-gram 哈希，不是真实的模型。** 它存在的目的是让测试套件能离线运行、不需要 API key。设置 `OPENAI_API_KEY` 后，`rag/vector.py` 会切换到 `text-embedding-3-small`，语义分支会明显变好 —— 但我在这个仓库里没有测过，所以我不给任何数字。
- **第一版比纯 BM25 还差。** 词级哈希会让 "terminate" 和 "termination" 之间的相似度为 0，所以语义分支纯属噪声：recall@1 只有 0.2273。是换成字符 3-gram 之后，混合检索才同时超过两个基线。这段历史保留在 `_hash_embed` 的 docstring 里，没有清理掉。

只报告获胜配置的检索基准，那是营销文档。这个仓库把失败的运行也留着。

---

## 工作原理

```
data/contracts/*.txt
      │
      ├─ corpus.py      clause-aware chunking  →  Chunk(ref="msa_2023#s001", text=...)
      │
      ├─ bm25.py        Okapi BM25 (k1=1.5, b=0.75)      ─┐
      │                                                   ├─ fusion.py  RRF (k=60, 0.4/0.6)
      └─ vector.py      embeddings + cosine              ─┘
                                                          │
                                              answer.py  grounding floor + citation audit
                                                          │
                                                    {answer, citations[]}
```

**为什么用 RRF 而不是加权分数平均。** BM25 的分数是无界的（常见 0–20）；余弦值落在 [-1, 1] 区间内。直接对原始分数取平均，意味着词法分支仅凭量纲就会占主导，跟你以为自己设的权重毫无关系。倒数排名融合（Reciprocal Rank Fusion）只使用排名，而排名天生可比。

**引用审计。** 生成之后，模型输出里的每一个 `[doc#chunk]` 引用，都会跟实际召回的块集合做核对。从未被召回的引用会被标记并丢弃。这拦住了 LLM 文档问答最丢人的一种失败模式 —— 一段流畅的回答，引用了一条看起来真实、但根本不存在于文档里的条款。

**拒答下限。** 如果融合后的最高分低于 `0.015`，系统抛出 `RefusalError` 而不是作答。这个阈值是在 4 个域外问题上调出来的（4 个全部拒答，18 个域内问题零误拒）。该下限是 `rag/answer.py` 顶部的一个常量；任何改动它的提交，都应该让评测输出里的 `refusal_rate` 发生变化。

---

## 目录结构

```
rag/
  tokenizer.py   lowercase + stopword tokenizer, sentence splitter
  corpus.py      clause-aware + sentence-packing chunking
  bm25.py        Okapi BM25, pure Python
  vector.py      embedding client (hash / openai) + cosine index
  fusion.py      Reciprocal Rank Fusion
  answer.py      context assembly, refusal floor, citation audit
  evaluate.py    recall@k, MRR, refusal rate, weight sweep
cli.py           index / ask / eval
tests/           8 tests, 0.03s, no network
data/contracts/  3 synthetic contracts + gold.json
```

`data/contracts` 里的合同是合成的 —— 专为这个仓库编写，这样评测集可以一并提交，任何人 clone 下来都能复现这些数字。

---

## 运行

```bash
# offline, no keys needed
python cli.py index
python cli.py ask "What interest rate applies to late payments?" -k 3
python cli.py eval
python -m unittest discover -s tests

# with a real embedding model
export OPENAI_API_KEY=sk-...
python cli.py eval
```

```
$ python cli.py eval
{
  "n_cases": 22,
  "recall@1": 0.7727,
  "recall@3": 0.7727,
  "recall@5": 0.7727,
  "mrr@10": 0.7778,
  "refusal_rate": 1.0
}
```

---

## 如果要上生产，还缺什么

按代价从高到低排列的缺口：

1. **真实 embedding。** `text-embedding-3-small`，或者本地 BGE 模型。哈希后端只是 CI 的便利设施，不是语义模型。
2. **重排器（Reranker）。** 在融合后的 top 20 候选上跑一个 cross-encoder，带来的准确率提升超过任何融合参数微调。
3. **向量存储。** 在 Python list 上做内存余弦，对 20 个块没问题，到 12 万份文档就崩了。该上 pgvector 配 HNSW 索引，或者 Qdrant。
4. **纯文本以外的格式的条款切分。** 真实合同是 DOCX 和 PDF 来的，编号格式并不统一；`corpus.py` 里的正则只适配干净的 ASCII 编号。
5. **评测集规模。** 22 个问题足够发现回归，但远不够声称准确率。客户那边的集合是约 2000 道人工标注题；那才是有意义的数字。
