from src.preprocess import main as preprocess_main
from src.train import main as train_main
from src.predict import main as predict_main
from src.backtest import main as backtest_main
from src.calendar_forecast import main as calendar_main
from src.monthly_shape_forecast import main as monthly_shape_main
from src.event_distance_forecast import main as event_distance_main
from src.online_ablation import main as online_ablation_main
from src.production import main as production_main

if __name__ == "__main__":
    print("[1/9] 数据预处理与特征工程...")
    preprocess_main()
    print("[2/9] 训练内部滚动基线...")
    train_main()
    print("[3/9] 生成内部9月基线...")
    predict_main()
    print("[4/9] 执行多月滚动回测...")
    backtest_main()
    print("[5/9] 选择日历趋势基线...")
    calendar_main()
    print("[6/9] 优化月度总量与日内形状...")
    monthly_shape_main()
    print("[7/9] 融合节假日距离与周周期信号...")
    event_distance_main()
    print("[8/9] 固定线上验证最优结构（申购月度形状 + 赎回事件距离）...")
    online_ablation_main()
    print("[9/9] 发布正式结果并生成匹配验证集...")
    production_main()
    print("全部流程已完成。正式结果：output/tc_comp_predict_table.csv")
