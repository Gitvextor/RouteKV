# Router V4.0.2 队友交接包

日期：2026-07-30

这是一个可复核的研究交接包，汇总了 Router 从 V3.0 表征学习到
V4.0.2 共享 Policy 风险建模的设计、数据、代码、实验结果和最终
research-only checkpoint。

## 建议阅读顺序

1. `01_ROUTER_PROJECT_SUMMARY.md`：项目总览、架构演进、实验结果和结论。
2. `historical_reports/`：V3.0、V4.0.0、V4.0.1 的原始阶段报告。
3. `router_v4_0_2/final_report.md`：V4.0.2 的最终完整报告。
4. `router_v4_0_2/routing_evaluation.json`：完整机器可读指标。
5. `router_v4_0_2/source/`：V4.0.2 源码、配置、Schema 与测试。
6. `router_v4_0_2/output/`：OOF 预测、决策、模型、审计和图表。

## 最重要的结论

- V4.0.2 使用 780 Questions / 661 Documents。
- Complete-6 Nested OOF：
  - Acceptable Rate：66.20%；
  - H2O Route Rate：10.40%；
  - Safe Aggregate Peak-KV Saving：0.67%；
  - Compression-Induced Failures：3。
- Shared Candidate Risk 相比 V4.0.1 将 CIF 从 5--7 条降到 3 条，
  但仍未达到原始“零新增错误”目标。
- 经明确授权，正确率门禁被降级为质量警告；技术完整性门禁仍保留。
- 已使用全部 780 条数据训练 10-member、50-epoch 的
  `research_only` ensemble。
- 该模型不是 production checkpoint，也不是外部 blind 结果。

## 最终 checkpoint

相对路径：

```text
router_v4_0_2/output/research_candidate/ensemble.pt
```

SHA-256：

```text
2551427a1cae5bf095b39b7a1b6ccc52719b07c27e7b72774b5062c6f425ca7c
```

## 验证

在 PowerShell 中运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\VERIFY_PACKAGE.ps1
```

脚本会依据 `PACKAGE_SHA256.csv` 检查包内文件是否缺失或被修改。

## 状态边界

```text
research_only=true
routing_enabled=false
production_checkpoint=false
formal_result_eligible=false
external_blind_test_required=true
```

原始 OOF 中的 3/4 条新增错误被完整保留。Quality Gate Override 只允许
继续研究训练，不代表这些错误消失，也不代表模型已经达到部署标准。
