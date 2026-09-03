运行 python run_all.py 后，输出文件会生成在这里。
最重要的最终提交文件为：tc_comp_predict_table.csv

运行 python -m src.backtest 后会额外生成：
- rolling_backtest_predictions.csv：各回测月逐日真实值、预测值与相对误差
- rolling_backtest_metrics.csv：各月份和总体回测指标
