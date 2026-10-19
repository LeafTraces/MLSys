
# CMU 15-442/642 Machine Learning Systems · Labs

## 进度

| Lab               | 内容                                   | 状态      | 结果                                   |
| ----------------- | -------------------------------------- | --------- | -------------------------------------- |
| [Lab 1](labs/lab1) | 反向模式自动微分框架 + 逻辑回归        | ✅ 完成   | 公开测试 28/28 通过；测试准确率 0.9611 |
| Lab 2             | 分布式训练：ZeRO-3 参数分片 + 张量并行 | ⏳ 未开始 | —                                     |
| Lab 3             | Blackwell GPU 上的 FP16 GEMM 优化      | ⏳ 未开始 | —                                     |

---

## Lab 说明

### Lab 1：自动微分框架 + 逻辑回归

实现一个 TF1 风格（先定义后运行）的反向模式自动微分框架，并用它训练逻辑回归模型识别手写数字。

- **Part 1：自动微分框架**（`auto_diff.py`）
  - Task 1：各算子的前向计算 `compute`
  - Task 2：`Evaluator.run`，按拓扑序执行计算图
  - Task 3：各算子的反向规则 `gradient`
  - Task 4：`gradients`，通过扩展计算图构造反向图，支持梯度的梯度
- **Part 2：逻辑回归**（`logistic_regression.py`）
  - Task 5：前向图 Z = XW + b（新增广播算子）
  - Task 6：softmax 交叉熵损失（新增求和、exp、log 算子）
  - Task 7：mini-batch SGD 训练循环

运行方式：

```bash
cd labs/lab1
python3 -m pytest -l -v tests/
python3 logistic_regression.py
```

### Lab 2：分布式训练

用 MPI 和 NumPy 实现 ZeRO Stage 3 的参数分片，以及张量模型并行的通信逻辑。

### Lab 3：Blackwell GEMM 优化

用 TIRx DSL，在 NVIDIA Blackwell（B200）上从零编写并逐步优化 FP16 矩阵乘法 kernel。需要通过 Modal 在云端 GPU 上运行。
