from src.preprocess import main as preprocess_main
from src.train import main as train_main
from src.predict import main as predict_main

if __name__ == "__main__":
    print("[1/3] 数据预处理与特征工程...")
    preprocess_main()
    print("[2/3] 模型训练与验证...")
    train_main()
    print("[3/3] 预测 2014 年 9 月 30 天结果...")
    predict_main()
    print("全部流程已完成。请查看 output/tc_comp_predict_table.csv")
