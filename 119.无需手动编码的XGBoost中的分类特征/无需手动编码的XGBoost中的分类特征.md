# 无需手动编码的XGBoost中的分类特征

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/02/xgboost-categorical-featured.png)


[XGBoost](https://www.nvidia.com/en-us/glossary/data-science/xgboost/) 是一种基于梯度提升的基于决策树的集成[机器学习](https://www.nvidia.com/en-us/glossary/data-science/machine-learning/)算法。 然而，直到最近，它还没有原生支持分类数据。 在将分类特征用于训练或推理之前，必须对其进行手动编码。

在序数类别的情况下，例如学校成绩，这通常使用标签编码来完成，其中每个类别都被分配一个与该类别的位置相对应的整数。 A、B 和 C 级可以分别指定整数 1、2 和 3。

在类别之间没有顺序关系的基本类别的情况下，例如颜色，这通常使用单热编码来完成。 这是为分类特征包含的每个类别创建新的二元特征的地方。 具有红色、绿色和蓝色类别的单个分类特征将被单热编码为三个二进制特征，一个代表每种颜色。

```python
>>> import pandas as pd
>>> df = pd.DataFrame({"id":[1,2,3,4,5],"color":["red","green","blue","green","blue"]})
>>> print(df)
  id  color
0   1    red
1   2  green
2   3   blue
3   4  green
4   5   blue

>>> print(pd.get_dummies(df))
  id  color_blue  color_green  color_red
0   1           0            0          1
1   2           0            1          0
2   3           1            0          0
3   4           0            1          0
4   5           1            0          0

```

这意味着具有大量类别的分类特征可能会产生数十个甚至数百个额外特征。 因此，遇到内存池和最大 DataFrame 大小限制是很常见的。

对于像 XGBoost 这样的树学习器来说，这也是一种特别糟糕的方法。 决策树通过找到所有特征的分裂点及其可能的值来训练，这将导致纯度的最大增加。

由于具有许多类别的 one-hot 编码分类特征往往是稀疏的，因此拆分算法通常会忽略 one-hot 特征，而倾向于稀疏程度较低的特征，这些特征可以带来更大的纯度增益。

现在，XGBoost 1.7 包含一项实验性功能，使您能够直接在分类数据上训练和运行模型，而无需手动编码。 这包括让 XGBoost 自动标记编码或单热编码数据的选项，以及一种最佳分区算法，用于有效地对分类数据执行拆分，同时避免单热编码的陷阱。 1.7 版还包括对缺失值的支持和最大类别阈值以避免过度拟合。

这篇文章简要介绍了如何在包含多个分类特征的示例数据集上实际使用新特征。

## 使用 XGBoost 的分类支持预测星型
要使用新功能，您必须先加载一些数据。 对于这个例子，我使用了 [Kaggle 型预测数据集](https://www.kaggle.com/datasets/deepu1109/star-dataset)

```python
>>> import pandas as pd
>>> import xgboost as xgb
>>> from sklearn.model_selection import train_test_split
>>> data = pd.read_csv("6 class csv.csv")
>>> print(data.head())
```
```python
  Temperature (K)  Luminosity(L/Lo)  Radius(R/Ro)  Absolute magnitude(Mv)  \
0             3068          0.002400        0.1700                   16.12  
1             3042          0.000500        0.1542                   16.60  
2             2600          0.000300        0.1020                   18.70  
3             2800          0.000200        0.1600                   16.65  
4             1939          0.000138        0.1030                   20.06  

   Star type Star color Spectral Class 
0          0        Red              M 
1          0        Red              M 
2          0        Red              M 
3          0        Red              M 
4          0        Red              M

```
然后，将目标列（星型）提取到自己的系列中，并将数据集拆分为训练和测试数据集。

```python
>>> X = data.drop("Star type", axis=1)
>>> y = data["Star type"]
>>> X_train, X_test, y_train, y_test = train_test_split(X, y)
```
接下来，将分类特征指定为类别 dtype。

```python
>>> y_train = y_train.astype("category")
>>> X_train["Star color"] = X_train["Star color"].astype("category")
>>> X_train["Spectral Class"] = X_train["Spectral Class"].astype("category")
```
现在，要使用新功能，您必须在创建 XGBClassifier 对象时将 enable_categorical 参数设置为 True。 之后，像训练 XGBoost 模型时一样继续。 这适用于 CPU 和 GPU tree_methods。

```python

>>> clf = xgb.XGBClassifier(
    tree_method="gpu_hist", enable_categorical=True, max_cat_to_onehot=1
)
>>> clf.fit(X_train, y_train)

XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=True,
              eval_metric=None, gamma=0, gpu_id=0, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=6, max_leaves=0,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1, 
              objective='multi:softprob', predictor='auto', random_state=0, 
              reg_alpha=0, ...)
```
最后，您可以使用您的模型生成预测，而无需一次性编码或以其他方式对分类特征进行编码。

```python
>>> X_test["Star color"] = X_test["Star color"]
    .astype("category")
    .cat.set_categories(X_train["Star color"].cat.categories)
>>> X_test["Spectral Class"] = X_test["Spectral Class"]
    .astype("category")
    .cat.set_categories(X_train["Spectral Class"].cat.categories)
>>> print(clf.predict(X_test))
[1 0 3 3 2 5 1 1 2 1 4 3 4 0 0 4 1 5 2 4 4 1 4 5 5 3 1 4 5 2 0 2 5 5 4 2 5
 0 3 3 0 2 3 3 1 0 4 2 0 4 5 2 0 0 3 2 3 4 4 4]
```
## 总结
我们演示了如何使用 XGBoost 对分类特征的实验支持来改进 XGBoost 在分类数据上的训练和推理体验。 有关更多示例，请参阅以下资源：

* [分类数据入门](https://xgboost.readthedocs.io/en/stable/python/examples/categorical.html#sphx-glr-python-examples-categorical-py)
* [使用 cat_in_the_dat 数据集训练 XGBoost](https://xgboost.readthedocs.io/en/stable/python/examples/cat_in_the_dat.html#sphx-glr-python-examples-cat-in-the-dat-py)


我们很高兴听到这项新功能如何让您的生活更轻松。 在接下来的几个月中，我们将进行更多工作，帮助您更好地理解基于树的算法的本质，这将在您未来的工作中大有裨益。

RAPIDS 团队始终与开源社区合作，以了解和解决新出现的需求。 如果您是开源维护者，有兴趣将 GPU 加速引入您的项目，请访问 GitHub 或 Twitter。 RAPIDS 团队很想了解潜在的新算法或工具包将如何影响您的工作。




































