# Purchase-Redemption 资金流入流出预测系统

本项目基于天池“资金流入流出预测”赛题，实现余额宝每日申购额和赎回额预测。项目已整理为适合提交到 GitHub 的版本：**代码、报告、说明文档可以提交；原始数据文件不提交**。

## 一、项目说明

目标：根据 2013-07-01 至 2014-08-31 的历史数据，预测 2014-09-01 至 2014-09-30 每日资金申购总额 `purchase` 和赎回总额 `redeem`。

本项目包含：

- 数据读取与预处理
- 每日申购/赎回聚合
- 日期特征、滞后特征、滚动统计特征构造
- LightGBM 回归建模，若本地无法安装 LightGBM，则自动降级为随机森林
- 2014 年 8 月验证集评估
- 2014 年 9 月滚动预测
- Streamlit 可视化展示
- 课程需求分析文档
- 原项目脚本归档

## 二、目录结构

```text
Purchase-Redemption/
├── Purchase Redemption Data/
│   └── README.txt                         # 数据放置说明，原始数据不提交 GitHub
├── src/
│   ├── config.py                          # 路径配置
│   ├── data_loader.py                     # 数据读取
│   ├── preprocess.py                      # 预处理与特征工程
│   ├── train.py                           # 模型训练与验证
│   ├── predict.py                         # 9 月预测与提交文件生成
│   └── evaluate.py                        # 评估函数
├── app.py                                 # Streamlit 可视化系统
├── run_all.py                             # 一键运行完整流程
├── requirements.txt                       # Python 依赖
├── output/
│   └── README.txt                         # 运行后生成预测和验证结果
├── models/
│   └── README.txt                         # 运行后保存模型文件
├── legacy_original_project/               # 你原本项目中的脚本和说明归档
├── 项目二_需求分析与最小方案.md
├── 项目二_需求分析与最小方案_目录更新版.pdf
└── .gitignore
```

## 三、数据放置

由于上传文件和 GitHub 仓库大小限制，本仓库不包含原始赛题数据。运行前请把天池数据解压到：

```text
Purchase Redemption Data/
```

推荐文件名如下：

```text
user_profile_table.csv
user_balance_table.csv
mfd_day_share_interest.csv
mfd_bank_shibor.csv
```

其中最重要的是：

```text
user_balance_table.csv
```

如果缺少这个文件，模型无法训练。

## 四、安装依赖

建议使用 Python 3.8 或以上版本。

```bash
pip install -r requirements.txt
```

如果 `lightgbm` 安装失败，程序会自动使用 `RandomForestRegressor`，仍可跑完整流程。

## 五、一键运行

在项目根目录执行：

```bash
python run_all.py
```

运行完成后会生成：

```text
output/daily_features.csv
output/validation_prediction.csv
output/tc_comp_predict_table.csv
output/tc_comp_predict_table_with_header.csv
models/model_purchase.pkl
models/model_redeem.pkl
models/feature_columns.json
```

文件说明：

| 文件 | 作用 |
|---|---|
| `daily_features.csv` | 预处理和特征工程后的日级训练数据 |
| `validation_prediction.csv` | 2014 年 8 月验证集预测结果，用于评估模型效果 |
| `tc_comp_predict_table.csv` | 2014 年 9 月最终预测提交文件，默认无表头 |
| `tc_comp_predict_table_with_header.csv` | 带表头版本，便于查看和展示 |

## 六、启动可视化系统

```bash
streamlit run app.py
```

页面包括：

1. 数据探索：展示历史申购/赎回趋势
2. 模型评估：展示 8 月验证集真实值与预测值对比
3. 预测结果：展示 2014 年 9 月预测结果，并下载无表头提交文件
4. 误差分析：展示验证集每日相对误差

如果首次运行 Streamlit 出现邮箱提示，直接按回车跳过即可。

## 七、如何判断预测是否可靠

2014 年 9 月真实值属于隐藏测试集，本地无法直接知道最终预测是否完全正确。因此本项目使用 2014 年 8 月作为验证集，模拟未来 30 天预测，并计算申购和赎回的相对误差。

评估结果主要查看：

- 申购 MAPE
- 赎回 MAPE
- 模拟总分
- 真实值与预测值折线图是否接近

最终 `tc_comp_predict_table.csv` 可用于赛题平台或课程结果展示。

## 八、提交到 GitHub

本项目已经配置 `.gitignore`，默认不会提交：

- 原始数据文件
- 运行生成的模型文件
- 运行生成的输出结果
- 压缩包
- Python 缓存文件

可以按下面流程提交：

```bash
git init
git add .
git commit -m "Initial commit: purchase redemption prediction project"
git branch -M main
git remote add origin 你的GitHub仓库地址
git push -u origin main
```

如果你的目录原本已经是 Git 仓库，则从 `git add .` 开始即可。

## 九、原项目文件说明

`legacy_original_project/` 中保留了你原本项目里的脚本和文档，便于查阅：

- `01_data_exploration.py`
- `02_feature_engineering.py`
- `03_train_data_prepare.py`
- `README_original.md`
- 部分原项目输出结果

当前推荐运行主流程为：

```bash
python run_all.py
streamlit run app.py
```
