# FuxiCTR中的Test Time Adaptation功能

本文档介绍了在FuxiCTR框架中实现的Test Time Adaptation (TTA)功能，该功能允许模型在推理阶段进行自适应调整，以适应测试数据分布的变化。

## 概述

Test Time Adaptation是一种在推理阶段调整模型参数的技术，无需重新训练整个模型。这对于处理训练数据和测试数据之间存在分布偏移的情况特别有用。在CTR预测场景中，这种偏移很常见，例如由于用户行为随时间变化、新物品的出现或季节性趋势等因素。

我们在FuxiCTR的BaseModel类中实现了TTA功能，使其能够与所有基于BaseModel的模型兼容。

## 实现方法

我们在BaseModel中添加了以下TTA相关功能：

1. **参数配置**：在BaseModel的__init__方法中添加了TTA相关参数：
   - `enable_adaptation`: 是否启用TTA功能
   - `adaptation_lr`: 适应过程的学习率
   - `adaptation_steps`: 每个批次上的适应步骤数
   - `adaptation_method`: 适应方法（目前支持"entropy_minimization"、"self_training"和"tent"）

2. **预测方法**：
   - `predict_with_adaptation`: 带有适应功能的预测方法
   - 重写了原有的`predict`方法，根据`enable_adaptation`参数决定是否使用适应功能

3. **评估方法**：
   - `evaluate_with_adaptation`: 带有适应功能的评估方法
   - 重写了原有的`evaluate`方法，根据`enable_adaptation`参数决定是否使用适应功能

4. **适应核心方法**：
   - `_adapt_on_batch`: 在单个批次上执行适应过程
   - `_entropy_loss`: 计算熵损失，用于某些适应方法

## 适应方法

目前实现了以下几种适应方法：

1. **熵最小化 (Entropy Minimization)**：
   - 通过最小化预测的熵来提高模型的确定性
   - 适用于模型对测试样本的预测不确定性较高的情况

2. **自训练 (Self-Training)**：
   - 使用模型的当前预测生成伪标签
   - 然后使用这些伪标签进行短期适应

3. **Tent**：
   - 基于Test Entropy最小化的方法
   - 源自论文：[Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726)

## 使用方法

### 在模型初始化时启用TTA

```python
model = DeepFM(feature_map, 
               enable_adaptation=True,
               adaptation_lr=1e-4,
               adaptation_steps=5,
               adaptation_method="entropy_minimization",
               **other_params)
```

### 在已加载的模型上启用TTA

```python
model.enable_adaptation = True
model.adaptation_lr = 1e-4
model.adaptation_steps = 5
model.adaptation_method = "entropy_minimization"
```

### 使用TTA进行预测

启用TTA后，`predict`和`evaluate`方法会自动使用适应功能：

```python
# 使用TTA进行预测
predictions = model.predict(test_gen)

# 使用TTA进行评估
metrics = model.evaluate(test_gen)
```

## 示例脚本

我们提供了一个示例脚本`demo/example_test_time_adaptation.py`，展示了如何使用TTA功能：

1. 首先使用常规方式训练模型
2. 在测试集上分别使用和不使用TTA进行评估
3. 比较两种方法的性能差异

## 注意事项

1. TTA会增加推理时间，因为每个批次都需要进行多步适应
2. 适应学习率(`adaptation_lr`)应该设置得比训练学习率小，以避免过度适应
3. 适应步骤数(`adaptation_steps`)需要根据具体任务进行调整，通常5-10步就足够了
4. 不同的适应方法对不同类型的分布偏移有不同的效果，可以根据具体场景选择合适的方法 