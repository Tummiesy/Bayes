# 使用朴素贝叶斯进行短文本分类

## 1）项目概述

本项目构建了一条完整且可复现的多分类短文本实验流水线，使用朴素贝叶斯模型在 **3 个独立数据集** 上进行训练与评估：

- `banking`
- `snips`
- `stackoverflow`

每个数据集都被视为独立任务（不会跨数据集合并训练）。

对于每个数据集，流水线会执行：
1. 加载 `train/dev/test` 三个 TSV 文件。
2. 校验数据模式（schema）与行数据有效性。
3. 进行轻量文本预处理。
4. 训练并比较不同“向量化器 + 朴素贝叶斯”配置。
5. 以 **dev Macro-F1** 选出最佳配置。
6. 采用 **Option B** 模型选择策略：
   - 在 train→dev 搜索中确定最佳配置；
   - 用最佳配置在 **train+dev** 上重新训练；
   - 在 **test** 上只评估一次。
7. 将报告、指标和图表保存到 `outputs/`。

---

## 2）数据集目录结构

期望目录结构：

```text
project_root/
  data/
    banking/
      train.tsv
      dev.tsv
      test.tsv
    snips/
      train.tsv
      dev.tsv
      test.tsv
    stackoverflow/
      train.tsv
      dev.tsv
      test.tsv
```

每个 TSV 至少需要包含以下列：

- `text`
- `label`

> 说明：为了兼容旧结构，代码也支持数据集目录直接位于项目根目录（`./banking`、`./snips`、`./stackoverflow`）。

---

## 3）安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

依赖保持尽量精简：

- pandas
- numpy
- scikit-learn
- matplotlib

---

## 4）运行方式

在项目根目录执行：

```bash
python main.py
```

脚本会在每个数据集上跑完整组合实验，并将结果写入 `outputs/`。

---

## 5）对比的模型与向量化器

### 向量化器
- `CountVectorizer`，`ngram_range=(1,1)`
- `CountVectorizer`，`ngram_range=(1,2)`
- `TfidfVectorizer`，`ngram_range=(1,1)`
- `TfidfVectorizer`，`ngram_range=(1,2)`

### 分类器
- `MultinomialNB`
- `BernoulliNB`
- `ComplementNB`

### 超参数
- `alpha` 取值：`[0.1, 0.5, 1.0, 2.0]`
- `min_df` 取值：`[1, 2, 3]`

无效组合会被安全捕获，并在实验结果表中以 `status=failed` 记录。

---

## 6）预处理

默认预处理策略（轻量且可复现）：
- 转小写（lowercase）
- 去除首尾空格
- 合并重复空白字符

可选开关（位于 `PreprocessConfig`）：
- 去除标点
- 去除数字
- 去除英文停用词

默认不使用词干提取或词形还原。

---

## 7）模型选择逻辑

主选择指标：**dev Macro-F1**。

并列时的决策顺序：
1. dev Accuracy 更高者优先
2. 更简单的 n-gram（优先 `(1,1)` 而非 `(1,2)`）
3. 更简单的向量化器（优先 `Count` 而非 `TF-IDF`）

最终评估策略：**Option B**。
- 在 train/dev 上搜索配置
- 用最佳配置在 **train+dev** 上重训
- 在 test 上仅评估一次

---

## 8）输出产物

每个数据集（`outputs/<dataset>/`）会生成：

1. `all_experiments.csv`
   - 每个配置一行，包含 dev 指标。
2. `best_config.json`
   - 最优配置与对应 dev 指标。
3. `test_metrics.json`
   - 最终 test 指标。
4. `classification_report.txt`
   - 完整的 sklearn 分类报告。
5. `confusion_matrix.png`
   - 最优模型的混淆矩阵图。

全局输出（`outputs/`）：

- `summary_results.csv`（每个数据集的最佳配置与关键指标）
- `best_test_macro_f1_barplot.png`（可选的对比柱状图）
- `analysis.txt`（对最易/最难数据集的简短说明）

---

## 9）为什么短文本任务适合朴素贝叶斯

朴素贝叶斯是短文本分类的强基线，原因包括：

- 对稀疏词袋特征表现良好。
- 训练与预测速度快。
- 在数据量有限、类别较多时依然稳健。
- 配合简单预处理通常也能获得有竞争力的效果。

因此它非常适合意图分类类任务，例如银行问句分类、SNIPS 意图识别和 StackOverflow 题目标签分类。

---

## 10）项目代码结构

```text
src/
  data_loader.py   # 加载 TSV + 数据校验
  preprocess.py    # 文本预处理
  features.py      # 向量化器构建
  models.py        # 朴素贝叶斯模型构建
  evaluate.py      # 指标、报告、混淆矩阵、柱状图
  experiment.py    # 端到端实验循环 + 选择逻辑 + 输出
  utils.py         # IO 与日志辅助
main.py            # 程序入口
```
