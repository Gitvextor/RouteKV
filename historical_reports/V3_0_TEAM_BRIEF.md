# Router V3.0 队友说明

## 1. 我们在做什么

Router V3.0 是一个 **representation capability pilot**，目标是验证：

```text
Document + Question
→ Semantic Features + Structural Features
→ MLP
→ simpshort / simplong / compshort / complong
```

它目前只预测 DocMath 四类任务，不选择 KV-cache/H2O policy，因此还不是正式的 H2O Router。

## 2. 模型架构

- Semantic Features：复用 Router V3 的 Frozen BGE、Chunk Builder 和 BM25/Anchor Selector。
- 每条样本最多选择 8 个相关 chunks，形成 1920 维 semantic representation。
- Structural Features：16 维，包括 prompt tokens、model-context load、文档/问题长度、数字、公式、表格等统计。
- Primary Hybrid MLP：

```text
1936
→ Linear(128) → LayerNorm → GELU → Dropout
→ Linear(64)  → LayerNorm → GELU → Dropout
→ z64
→ Linear(64, 4)
```

`z64` backbone 和四分类 head 独立保存，后续可以替换成正式 H2O routing head。

## 3. 数据与训练

- 原始候选：3,200 rows / 2,408 documents。
- 完整 testmini 锁：800 rows / 745 documents，仅用于去重，Router 不读取原始 testmini。
- 排除重叠：440 rows / 314 documents。
- 最终数据：2,760 rows / 2,094 documents。
- Document-level split：

| Split | Rows | Documents |
|---|---:|---:|
| Train | 1,931 | 1,466 |
| Validation | 420 | 314 |
| Internal Test | 409 | 314 |

五个模型均先完成训练、early stopping 和 checkpoint 冻结，之后统一打开 Internal Test 一次。

## 4. 当前结果

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| Structural Logistic | 0.9095 | 0.8954 |
| Structural MLP | 0.9169 | 0.9056 |
| Semantic MLP | 0.9584 | 0.9451 |
| **Hybrid MLP Primary** | **0.9682** | **0.9623** |
| Hybrid Logistic | 0.9658 | 0.9554 |

主要观察：

- Hybrid MLP 相比 Structural-only MLP，Macro-F1 提升 **0.0567**，说明 Semantic Features 有明显增益。
- Hybrid MLP 相比 Semantic-only MLP，提升 **0.0172**，说明 Structural Features 也有补充价值。
- Hybrid MLP 相比 Hybrid Logistic，提升 **0.0069**，MLP 非线性建模有小幅收益。
- Short vs Long Accuracy 为 0.9902；Complexity-Proxy Accuracy 为 0.9731。

## 5. 如何理解结果

这些结果说明模型学到了有用的 semantic 和 structural **数据集信号**，但不能解释为模型理解了真实的 intrinsic reasoning difficulty。

Semantic 部分只编码检索选择后的 chunks，不是完整全文编码；source、template、formatting 和 length confounding 仍然存在。Length-matched Complexity-Proxy 因匹配文档不足，目前不可估计。

## 6. 下一步

1. 保留当前 `z64` backbone，检查跨数据集和不同 source/template 下的稳定性。
2. 增加更严格的 length/source-matched 分析，判断当前成绩中有多少来自 shortcut。
3. 将四分类 head 替换为正式 H2O policy/outcome head。
4. 使用真实 H2O 运行结果训练 routing objective，并保持 correctness、truncation、latency 分离。
5. 最终与 Full、Recent-only、固定 H2O 和 Oracle 等策略进行同预算比较。

当前完整报告与 checkpoint 位于：

```text
router_v3_0/outputs/docmath_4class_capability_v1_gpu_full/
```
