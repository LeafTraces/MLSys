# Lab 1 · 自动微分框架 + 逻辑回归 学习笔记

> **课程**：CMU 15-442/642 Machine Learning Systems（Spring 2026）
> **作业**：<https://github.com/mlsyscourse/assignment1>
> **对应讲次**：Lecture 3（编程抽象 / TF1 风格计算图）、Lecture 4（自动微分）
> **最终结果**：Part 1 全部 28 个公开测试通过；逻辑回归在手写数字数据集上测试准确率 **0.9611**（与 notebook 参考值一致）
>
> ⚠️ 课程要求不要把作业答案公开到 GitHub 等公共网站。本笔记只记录思路、坑和方法，不包含完整答案代码。

---

## 0. 一句话总结

我从零实现了一个 **TF1 风格的反向模式自动微分框架**：用 `Node` 和 `Op` 搭计算图，用 `Evaluator` 执行图，用 `gradients` 通过「扩展计算图」生成反向图。然后在这个框架上新增了广播、求和、exp、log 四个 Op，训练出一个准确率约 96% 的逻辑回归模型。

---

## 1. 框架的核心结构

### 1.1 两个角色：Op 与 Node

| | `Op`（运算类型） | `Node`（图上的一个节点） |
|---|---|---|
| 是什么 | 「乘法」「矩阵乘」这类运算本身 | 图中具体的某一次运算 |
| 有几个 | 每种只有一个全局实例（如 `mul = MulOp()`） | 每建一次运算就新建一个 |
| 存什么 | 不存数据，只提供方法 | `inputs`、`op`、`attrs`、`name` |
| Logisim 类比 | 元件库里的「AND 门」类型 | 画布上摆下的某个 AND 门 |

**建图调用链**：`x1 * x2` → `Node.__mul__` → `mul(x1, x2)` → `MulOp.__call__` → 返回一个新 Node（不计算任何数）。

> ⭐ **节点本身不携带数值，也不携带形状。** 数值只存在于 `Evaluator.run` 里的字典中。

### 1.2 四个 Task 的 2×2 地图

| | **单个节点** | **整张图** |
|---|---|---|
| **算数值**（操作 numpy 数组） | `Op.compute`（Task 1） | `Evaluator.run`（Task 2） |
| **建反向图**（操作 Node） | `Op.gradient`（Task 3） | `gradients`（Task 4） |

> ⭐ **最重要的一条规则：`compute` 里只有 numpy 数组；`gradient` 里只有 Node。** 我在 Part 1 和 Part 2 里有好几次把两者混用（在 `gradient` 里用 `np.transpose`、`np.exp`，在 `sgd_epoch` 里用 `ad.broadcast`），都是违反了这一条。

### 1.3 与课程的对应

| 作业里 | 课程概念 |
|---|---|
| `ad.Variable` | TF1 的 `tf.placeholder` |
| `Evaluator.run` | `sess.run`：找依赖 → 喂数据 → 按拓扑序执行 |
| `gradients` | `tf.gradients`：扩展计算图（Lecture 4 第 15–20 页） |
| `gradient` 的 `output_grad` | 伴随值 v̄ᵢ |
| `gradient` 的返回列表 | 部分伴随值 v̄_{k→i}，每个输入一份 |
| `node_to_grad` 字典 | Lecture 4 第 14 页伪代码里的同名变量 |
| `sgd_epoch` 里 numpy 更新 W | 图外的「写回」，相当于 `tf.assign` |

---

## 2. Part 1 各 Task 要点

### Task 1：`compute`

- 用 numpy 实现前向计算，大多一两行。
- `assert len(input_values) == N` 的 N 是**输入个数**（乘法是 2，乘常数是 1）。
- 常数存在 `attrs` 里，用 `node.constant` 取。
- MatMul 要先看 `node.trans_A` / `node.trans_B` 决定是否转置。numpy 的转置是 `.T` 或 `np.transpose`，不是 PyTorch 的 `.t()`。

### Task 2：`Evaluator.run`

- **拓扑排序**：DFS 后序遍历，先递归访问所有输入，再加入自己；用 visited 集合去重。起点是**要计算的节点** `self.eval_nodes`，因为 DFS 是沿 `inputs` 往回走的。
- **按拓扑序填「草稿纸」字典**：
  - placeholder：从 `input_values` 抄值，找不到就抛 `ValueError`；
  - 其他节点：按 `node.inputs` 的**顺序**，从草稿纸（不是 `input_values`！）取出输入值，调用 `compute`，结果写回草稿纸。
- 返回 `[草稿纸[n] for n in eval_nodes]`。

### Task 3：`gradient`

套路：在纸上推局部导数 → 用 `output_grad`、`node.inputs[...]`、`attrs` 里的常数，通过 `+ - * /` 或 `matmul(...)` 把它「搭」出来。

| Op | 局部导数要点 |
|---|---|
| Mul | 乘法的导数是**另一个因子** |
| MulByConst | 另一个因子是**常数**，不是输入节点 |
| Div | 对分母：−a/b² |
| MatMul | 记住 Z = A′B′ 时 Ā′ = Z̄B′ᵀ、B̄′ = A′ᵀZ̄，再处理转置；用 (PQ)ᵀ = QᵀPᵀ 把「乘积再转置」改写成 `matmul` 的转置参数 |

### Task 4：`gradients`

- 反向拓扑序：`topo_sort([output_node])[::-1]`。
- 种子必须是 **Node**：`ones_like(output_node)`，和输出同形状。
- 对每个节点：先把收到的各份加起来，记录为它的伴随值；placeholder 用 `continue` 跳过；其他节点调用 `gradient`，用 `zip(node.inputs, input_grads)` 把每一份送给对应的输入。
- 返回的梯度本身是一张由普通 Op 组成的图，所以能**再求一次导**（`test_gradient_of_gradient` 通过）。

---

## 3. Part 2 各 Task 要点

### 3.1 新增的四个 Op

| Op | compute | gradient 的关键 |
|---|---|---|
| **Broadcast**(b, 参照节点) | `np.broadcast_to(b, 参照.shape)` | 给 b：沿 axis=0 求和（b 被复制到每一行，多条路径相加）；给参照节点：`zeros_like`（它只提供形状，值不影响结果） |
| **Sum**(x, dim, keepdims=False) | `np.sum(x, axis=dim, keepdims=...)` | 把 output_grad **广播回输入的形状**（每个被加的元素局部导数都是 1） |
| **Exp** | `np.exp` | `output_grad × node`：exp 的导数就是它自己的**输出**，复用前向值 |
| **Log** | `np.log` | `output_grad / 输入` |

### 3.2 Task 5：`logistic_regression`

Z = XW + broadcast(b, XW)

- 广播的参照节点必须是 **XW**，不能是 Z：Z 依赖广播结果，用 Z 会形成**环**。
- ⚠️ 曾经写成 `add(xw, b)`：numpy 在前向**悄悄自动广播**，结果看起来正确，但 b̄ 的形状变成 (B, k)，训练会乱。是靠「先预测 b̄ 应为 (2,)」发现的。

### 3.3 Task 6：`softmax_loss`

$$
\text{loss} = \frac{1}{B}\sum_b\Big(\log\sum_j e^{Z_{bj}} - \sum_j Y_{bj}Z_{bj}\Big)
$$

拆解为：exp → 沿 axis=1 求和（keepdims）→ log；Z×Y → 沿 axis=1 求和（keepdims）；相减 → 全部求和（dim=None）→ 除以 batch_size。

- **为什么 axis=1 要 keepdims**：反向时 `SumOp.gradient` 要把 (B,) 广播回 (B, k)，按右对齐规则会失败；(B, 1) 才能广播。如果 B 恰好等于 k，它不报错，而是**悄悄算错**。
- **为什么最后用 dim=None 而不是 dim=0**：前面加了 keepdims，形状是 (B, 1)。dim=0 得到 (1,)，dim=None 才得到要求的 `()`。
- **为什么要传 batch_size**：1/B 必须在建图时就存进 `div_by_const` 的 attrs，而建图时 Node 不知道自己的形状。
- 不需要自己写 SubOp：`Node.__sub__` 已经把减法拆成 `+` 和 `×(−1)`，推导出的梯度和手写 SubOp 完全一样。

### 3.4 Task 7：`sgd_epoch`

- **W 像接力棒一样传递**：W₀ →batch 0→ W₁ →batch 1→ W₂ …，`f_run_model` 要在循环**里面**调用，每次传入最新的 W、b，梯度每次重新算。
- **one-hot**：形状 (batch, 类别数)，类别数从 `W.shape[1]` 得到；可以用 `np.eye(k)[labels]` 构造。
- **最后不完整的 batch**（1437 = 28×50 + 37），两种方案都正确：
  - 丢弃（简单；每轮 shuffle，长期看所有数据都会用到）；
  - 保留，并把梯度乘以 scale = 50/37 修正（我最终的版本，参考了 AI 的修正）。之所以成立，是因为梯度对 loss 是线性的。
- **平均 loss**：先求所有样本 loss 的**总和**，再除以样本总数，保证每个样本权重相等；不能直接平均各 batch 的均值。
- `end = min(...)` 的真正作用：保证 `end - start` 是这个 batch 的**真实大小**（切片越界本身不会报错）。

---

## 4. 我踩过的坑（按类型）

### 4.1 类型混淆：「这个变量到底是什么？」

| 错误写法 | 混淆 |
|---|---|
| `node.value`、`node.key` | 以为值存在节点上（其实在字典里） |
| `np.transpose(B)`，B 是 Node | Node 当成了数组 |
| `OnesLikeOp()` | Op 当成了 Node（应调用全局实例 `ones_like(x)`） |
| `node is placeholder` | Node 和 Op 比较（应为 `node.op is placeholder`） |
| `topo_sort(output_node)` | 要列表，传了单个 Node |
| `zeros_like` 返回「0 的 ndarray」 | gradient 里只有 Node |
| `ad.broadcast(y, X)`（在 sgd_epoch 里） | numpy 世界里用了图的 Op |
| `target_node = input_values[1]` | 变量名暗示是 Node，其实是数组 |

### 4.2 Python 词汇

| 想做的事 | 正确写法 | 我踩过的坑 |
|---|---|---|
| 字典写入 / 读取 | `d[k] = v` / `d[k]` | 只写了 `d[k]` 想写入 |
| 遍历字典 | `for k in d` 得到**键** | 以为得到值 |
| 调用方法 | `d.items()` | 漏括号 |
| 生成列表 | `[f(x) for x in xs]` | — |
| 配对遍历 | `zip(xs, ys)` | — |
| 反转 | `xs[::-1]` | — |
| 跳过本轮 | `continue` | — |
| 多个条件 | `a and b` | 写成 `(a, b)`，元组永远为真 |
| 循环 n 次 | `for i in range(n)` | `for i in batch_size`（整数不能迭代） |
| 取负 | `(-1) * node` | `-node`（Node 没有 `__neg__`） |
| 拼写 | `np.broadcast_to`、`keepdims`、`node.inputs` | `braodcast`、`keepdim`、`node.input` |
| 函数返回 | 最后一行写 `return` | **漏了三次** → 返回 None |

### 4.3 形状

- **numpy 广播规则**：从最右一维对齐，每一维要么相等，要么其中一个是 1；短的形状左边自动补维。
- `np.sum(axis=k)` 默认会**去掉**第 k 维；`keepdims=True` 保留为 1。
- 形状例子里让**各维大小互不相同**（如 B=2、k=3），否则形状错误会被巧合掩盖。
- numpy 的「自动广播」会让错误的代码在前向看起来正确，问题只在反向或训练中暴露。

### 4.4 调试与环境

- **读报错从最后一行往上看**：错误类型 → 出错的行 → 调用链 → 推断哪个变量有问题。
- `'NoneType' object has no attribute ...` → 某个函数**漏了 return**，或用的是旧代码。
- **Colab 的 import 缓存**：同一进程里模块只导入一次。改了 `.py` 文件要**重启运行时**（或用 `%autoreload 2`）。判断依据：报错里的 `ipykernel_xxx` 编号没变，说明没重启。`!python3 -m pytest` 每次是新进程，不受影响。
- 怀疑变量不对时：`print(type(x), x)`。

---

## 5. 「理论 → 代码」的五步法

1. **标类型**：每个参数、返回值、中间变量是什么类型、什么形状；顺手抄下 docstring 里的要求（比如要抛 ValueError、返回形状为 `()`）。
2. **手算最小例子**：挑一个 2～5 个元素的小例子（至少包含一个「中间节点被别的节点使用」的情况），假装自己是计算机，用表格写出每一步后各变量的内容。
3. **写注释骨架**：每一步回答三个问题：**从哪个变量取？取的是什么（类型）？结果放到哪里？**
4. **逐行翻译**：卡住时先判断是「不知道做什么」（回到第 2 步）还是「不知道怎么写」（查词汇表）。
5. **写一点，测一点**：运行前**先写下预测**，结果和预测对不上的时候，就是发现 bug 的时候。

> 💡 写函数时先把最后一行 `return ...` 写上，再往前填。

---

## 6. 自测题（先自己答，再展开对照）

**1. 为什么拓扑排序要从 `eval_nodes` 出发，而不是从输入节点出发？**

<details><summary>参考答案</summary>

DFS 是沿着 `node.inputs` 往回走的。从输出出发能找到它依赖的全部节点；从 placeholder 出发，它的 inputs 为空，一步都走不动。

</details>

**2. `gradients` 的种子为什么是 `ones_like(output_node)`，而不是数字 1？**

<details><summary>参考答案</summary>

`gradient` 方法要求 output_grad 是 Node；而且输出可能是矩阵，伴随值要和输出同形状。`ones_like` 正好造出「和输出同形状的全 1」节点。

</details>

**3. Broadcast(b, XW) 的反向，为什么给 b 的那一份是沿 axis=0 求和？给 XW 的那一份为什么是全 0？**

<details><summary>参考答案</summary>

b 被复制到每一行，每一行都是一条影响 loss 的路径，多条路径的伴随值要相加，即沿 batch 维（axis=0）求和。XW 只提供形状，它的值变化不影响广播结果，所以伴随值为 0。

</details>

**4. 写 `Z = add(xw, b)`（忘了用广播后的 b），前向结果是对的，问题出在哪里？**

<details><summary>参考答案</summary>

numpy 在 compute 里自动广播，前向正确；但 `AddOp.gradient` 给 b 的伴随值形状是 (B, k)，和 b 的形状 (k,) 不一致。更新 b 时 numpy 再次广播，b 变成 (B, k)，训练出错。

</details>

**5. 在 `sgd_epoch` 里，为什么 `f_run_model` 必须在循环内调用？**

<details><summary>参考答案</summary>

每个 batch 都要用**当前最新**的 W、b 重新计算梯度。只算一次梯度再重复使用，等于没有做 SGD，只是放大了步长。

</details>

**6. 训练集 1437 个样本、batch_size = 50。若保留最后 37 个样本并乘以 scale 修正，scale 是多少？为什么乘在梯度上也成立？**

<details><summary>参考答案</summary>

scale = 50/37。图里的 loss 是 37 个样本之和除以 50，乘以 50/37 后变成除以 37 的真实平均。梯度对 loss 是线性的，loss 乘以常数，梯度也乘以同一常数。

</details>

---

## 7. 可选的后续练习

- 在 `tests/test_customized_cases.py` 里，为 Exp / Log / Sum / Broadcast 写反向测试；可以用 Lecture 4 的**数值梯度检查**（中心差分）和 `gradients` 的结果对比。
- 处理「和输出无关的节点」：在 `gradients` 返回时，对找不到伴随值的节点返回 `zeros_like(节点)`。
- 思考：如果 `SumOp` 沿 axis=1 求和但**不**加 keepdims，怎样让反向仍然正确？（需要一个能「把被压掉的维度补回来」的 Op）
