# Router 项目详细总结：V3.0 到 V4.0.2

## 1. 项目目标

本项目希望训练一个轻量 Router，在大模型开始回答前，根据
Document + Question 判断应该采用 Full Attention 还是某个 H2O KV
压缩策略。

最终目标是：

```text
Document + Question
→ Frozen Semantic / Structural Representation
→ Policy Risk Prediction
→ Quality-First Policy Selection
→ 尽量保持回答正确，同时降低 Peak Attention KV
```

Action Space V1 固定为：

```text
FULL
SAFE_H2O_R100_I32
SAFE_H2O_R080_I32
SAFE_H2O_R060_I32
SAFE_H2O_R040_I32
SAFE_H2O_R020_I32
```

其中 r100 仍然是 H2O Policy，不能解释为 Full/no-prune。

## 2. 为什么需要 Router

固定使用 Full 最安全，但没有 KV 压缩收益；固定使用激进 H2O 可以
节约 KV，却会让部分原本能答对的题变错。

因此 Router 要解决的是：

> 哪些题可以安全压缩？如果能压缩，最激进可以到哪一级？

这是风险预测问题，不只是普通多分类问题。对本项目而言，
Compression-Induced Failure（CIF）定义为：

```text
Full acceptable = true
且被选择的 H2O Policy acceptable = false
```

Full 本身失败时，不允许用“H2O 恰好答对”来奖励主动压缩；这些 rescue
样本只做分析。

## 3. 数据现状

V4 系列共使用：

| 项目 | 数量 |
|---|---:|
| Questions | 780 |
| 独立 Documents | 661 |
| 完整六策略 Panel | 500 |
| Full/r80/r60 Outcome | 各 780 |
| r100/r40/r20 Outcome | 各 500 |
| Context-truncated | 102 |
| testmini-document overlap | 154 |
| Full acceptable | 528 |
| Full unacceptable | 252 |
| Any-policy acceptable | 554 |
| No verified safe policy | 226 |
| Full-failure/H2O-rescue | 26 |

Conditional CIF 数量：

| Policy | CIF |
|---|---:|
| r100 | 11 |
| r80 | 33 |
| r60 | 48 |
| r40 | 33 |
| r20 | 38 |

这些 CIF 是最重要、也最稀缺的训练信号。尤其 r100 只有 11 条，
导致风险模型很难稳定学习。

### 数据使用规则

- 缺失的 Policy Outcome 使用 mask，不伪造标签。
- Full Failure Head 使用全部 780 条。
- CIF Risk Head 仅使用 Full acceptable、Policy observed、非截断样本。
- Cost Head 使用所有已观测 Peak-KV。
- 截断样本保留训练和切片统计，但运行时固定回退 Full。
- cohort、路径、protocol provenance、correctness、Policy outcome 和
  Peak-KV target 都不得进入模型输入。

## 4. 表征基础：Router V3.0

V3.0 首先验证 Document + Question 表征是否可学习：

```text
Frozen BGE semantic features：1920维
+ Structural features：16维
= hybrid1936

1936 → 128 → 64
得到 z64 Router representation
```

V3.0 的 DocMath 四分类结果较高：

- Accuracy：0.9682；
- Macro-F1：0.9623；
- Semantic 相对 Structural-only MLP 的 Macro-F1 增益：+0.0567；
- Structural 相对 Semantic-only MLP 的 Macro-F1 增益：+0.0172。

这证明 z64 能捕获 DocMath 数据集信号，但不能解释为理解了真实
reasoning difficulty，也不能直接证明它包含 H2O Policy 风险信号。

## 5. 架构演进

### 5.1 Router V3.0.1

```text
Frozen z64
→ Linear(64,3)
→ Full / r80 / r60
```

小样本 Pilot 中，Learned Router 没有明显优于最佳 Fixed Policy，
且对少数 FULL 样本不稳定。结论是：z64 有一定 Policy signal，但
数据不足，直接三分类容易受类别不平衡影响。

### 5.2 Router V4.0.0

V4.0.0 把任务改成绝对 Policy Outcome：

```text
z64 + structural16 → 六策略 Failure Head
structural16 → 五策略 Cost Head
```

固定 Final Test Core 结果：

- Fixed Full Acceptable：65.00%；
- Router Acceptable：65.00%；
- CIF：0；
- H2O Route：7.00%；
- Safe Aggregate Peak-KV Saving：0.33%。

正确率目标达到，但 Router 太保守；因为预注册的最低路由率为 10%，
所以整体实验仍标记为 finished with limits。

### 5.3 Router V4.0.1

为了更充分利用 780 条数据，V4.0.1 改成 document-grouped Nested OOF，
并比较 Linear、Conditional MLP 和 Partial Fine-tune。

Complete-6 结果：

| 模型 | Acceptable | H2O Route | Safe KV Saving | CIF |
|---|---:|---:|---:|---:|
| Legacy Absolute Linear | 66.00% | 35.60% | 2.45% | 5 |
| Frozen Conditional MLP | 65.40% | 30.60% | 1.95% | 7 |
| Partial Fine-tune | 65.40% | 34.00% | 2.57% | 7 |
| Fixed Full | 66.60% | 0% | 0% | 0 |

V4.0.1 提高了压缩率，但新增错误仍然明显。更大的模型或解冻 Backbone
没有解决核心问题。

### 5.4 Router V4.0.2

V4.0.2 不再增加 Backbone 容量，而是把每道题展开为多个
Question × Policy Candidate：

```text
base80 = z64[64] + normalized structural16

risk_input =
base80[80]
+ policy_one_hot[5]
+ policy_order_coordinate[1]
+ base80 × policy_order_coordinate[80]
= 166维
```

三个独立模型：

```text
Full Failure Head:
LayerNorm(80) → Linear(80,1)

Shared CIF Risk Head:
LayerNorm(166) → Linear(166,1)

Shared Cost Head:
LayerNorm(22)
→ Linear(22,16) → GELU → Linear(16,1)
```

这样所有 H2O Policy 共享风险学习，同时保留 Policy 身份和保守顺序。

## 6. V4.0.2 训练和决策

### Nested OOF

- 复用 V4.0.1 的 5 个 Outer document folds；
- 每个 Outer Test 为 156 条；
- Outer Development 再做 3-fold document-grouped cross-fit；
- 每个模型由 10 个 document-bootstrap members 组成；
- 每条题只获得一次没有见过其 document 的 Outer OOF prediction；
- 780 条 OOF prediction 全覆盖；
- document leakage = 0。

训练参数：

```yaml
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-2
epochs: 50
batch_size_questions: 32
batch_size_candidates: 128
bootstrap_members: 10
gradient_clip_norm: 1.0
cost_loss: Huber
```

### 校准和风险上界

- Full Failure：独立 Platt calibration；
- CIF：shared-slope + policy-intercept Platt calibration；
- 计算 10 members 的 mean/std；
- 风险上界：

```text
risk_ucb = calibrated_mean + 1.645 × bootstrap_std
```

- 对 r100→r20 做单调风险投影，更激进 Policy 的风险不得低于更保守
  Policy。

### Selector

1. 特征/checkpoint/Action Space/协议无效 → Full；
2. context truncated → Full；
3. Full 风险或不确定性过高 → Full；
4. CIF 风险过高的候选被排除；
5. Saving <5% 的候选被排除；
6. 在剩余安全候选中选择预测 Peak-KV 最低者；
7. 1% cost tie 内选择更保守 Policy；
8. 无安全候选 → Full；
9. 禁止主动 H2O rescue。

## 7. V4.0.2 核心结果

### Primary Complete-6：500条

| Method | Acceptable | H2O Route | Safe Aggregate Saving | CIF |
|---|---:|---:|---:|---:|
| Fixed Full | 66.60% | 0.00% | 0.00% | 0 |
| V4.0.2 | 66.20% | 10.40% | 0.67% | 3 |
| Oracle | 66.60% | 47.60% | 19.90% | 0 |

Router Policy 分布：

```text
FULL  = 448
r100  = 18
r80   = 24
r60   = 0
r40   = 10
r20   = 0
```

Full fallback rate：89.60%。

安全 Oracle opportunity capture：3.34%。

### Secondary Common-3：780条

| Method | Acceptable | H2O Route | Safe Aggregate Saving | CIF |
|---|---:|---:|---:|---:|
| Fixed Full | 67.69% | 0.00% | 0.00% | 0 |
| V4.0.2 | 67.31% | 6.92% | 0.48% | 4 |
| Oracle | 67.69% | 47.05% | 15.79% | 0 |

### 与 V4.0.1 比较

V4.0.2 把 Complete-6 CIF 从 5--7 条降低到 3 条，说明：

- shared candidate risk 建模方向有效；
- document bootstrap/UCB 让模型更加保守；
- 但代价是 H2O Route 和 KV Saving 明显降低；
- 当前仍未同时达到“零新增错误 + 明显压缩”。

### Risk Model 能力

| Policy | AUROC | AUPRC | Failure Recall |
|---|---:|---:|---:|
| Full | 0.716 | 0.578 | 86.51% |
| r100 | 0.396 | 0.027 | 50.00% |
| r80 | 0.584 | 0.078 | 61.29% |
| r60 | 0.631 | 0.115 | 81.40% |
| r40 | 0.652 | 0.131 | 86.67% |
| r20 | 0.677 | 0.162 | 97.06% |

r100/r80 的 AUPRC 很低，说明稀少 CIF 仍然难以识别。

### Fold 稳定性

Complete-6 不同 Outer Fold 的 H2O Route 为 1%--22%，CIF 为 0--2。
这种波动说明当前风险阈值仍对训练文档组合敏感。

### 事后 Risk-Coverage 诊断

Outer OOF 曲线中存在：

```text
H2O Route = 3.80%
CIF = 0
Safe Aggregate Saving = 0.15%
```

但这个点是看到 Outer OOF outcome 后的事后诊断，不能反向替代冻结规则，
否则会产生测试集调参。

## 8. Quality Gate Override 与最终模型

原始要求是：

```text
Complete-6 CIF = 0
Common-3 CIF = 0
Acceptable Count >= Fixed Full
```

V4.0.2 未达到这些质量目标。之后经明确授权：

- 将这些质量门禁降级为警告；
- 保留真实的 3/4 条 CIF；
- document leakage、截断回退、协议兼容、未观测选择、artifact hash 等
  技术门禁继续强制执行；
- 使用全部 780 条训练最终 research-only bundle。

最终训练：

```text
Questions = 780
Observed candidates = 3060
CIF training candidates = 1967
Bootstrap members = 10
Epochs = 50
```

Checkpoint：

```text
router_v4_0_2/output/research_candidate/ensemble.pt
```

SHA-256：

```text
2551427a1cae5bf095b39b7a1b6ccc52719b07c27e7b72774b5062c6f425ca7c
```

全量模型没有在这 780 条上重新报告性能，因为那会成为训练集成绩。
性能证据仍然是上面的 Nested OOF。

## 9. 测试与完整性

最终验证：

| 测试 | 结果 |
|---|---:|
| V4.0.2 + V4.0.1 + V4.0.0 | 65 passed |
| V3.0 + V3 regression | 75 passed |
| V4.0.2 最终局部复测 | 11 passed |
| CUDA tiny smoke | Passed |
| Checkpoint reload | Passed |
| Artifact manifest errors | 0 |
| Outer OOF rows / unique samples | 780 / 780 |
| Document leakage | 0 |

## 10. 如何理解实验

### 已经证明的

- Frozen BGE + V3 Backbone 可以提供稳定的 z64 表征。
- 数据中确实存在大量安全压缩机会：Oracle 在 Complete-6 可安全压缩
  47.60%，Safe Aggregate Saving 约 19.90%。
- Shared candidate risk 比 V4.0.1 更能减少新增错误。
- 完整的数据、mask、document-grouped OOF、协议和 artifact 流程可以运行。

### 尚未证明的

- 尚未证明 Router 能在全新 blind documents 上保持当前表现。
- 尚未达到零新增错误。
- 尚未证明 780 条训练出的 checkpoint 可以直接部署。
- Peak Attention KV Saving 不等于整机显存节省或端到端加速。
- V3 四分类高分不等于理解真实 reasoning difficulty。

### 当前最大瓶颈

1. CIF 样本太少，尤其 r100 只有 11 条。
2. z64 来源于 DocMath 四分类预训练，不完全针对 H2O failure。
3. 同一 campaign 的数据已被多轮开发观察，不能作为正式 blind evidence。
4. CompLong 的 Full acceptable 比例很低，Router 几乎没有可靠压缩空间。
5. 风险阈值在不同 folds 间波动较大。

## 11. 推荐下一步

1. 用全新 document-disjoint 数据做一次真正 blind evaluation。
2. 优先增加“Full 正确但 H2O 失败”的独立文档，而不是只增加安全样本。
3. 新数据冻结前先固定 Router、threshold 和报告模板。
4. 同时保留 Fixed Full、Fixed Policies、Router、Oracle 对比。
5. 如果未来继续改模型，优先研究 H2O failure 专用表征或轻量 backbone
   fine-tune，不要单纯扩大 MLP。
6. production 前必须重新启用可接受的质量门禁，并在 blind test 上验证。

## 12. 最终状态

```text
research_only=true
routing_enabled=false
production_checkpoint=false
formal_result_eligible=false
external_blind_test_required=true
```

一句话总结：

> V4.0.2 已经把 Router 从“直接猜 Policy”推进到“对每个 Policy 预测风险”，
> 并明显减少了新增错误；但现有数据仍不足以同时实现零错误和高压缩，
> 最终 checkpoint 适合继续研究和 blind 验证，不适合直接部署。
