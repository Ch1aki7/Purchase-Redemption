"""一键可重复运行，按数据 -> 特征 -> 回测 -> 训练 -> 预测顺序执行。"""
from src.preprocess import preprocess
from src.feature_engineering import make_features,validate_no_leakage
from src.config import PROCESSED_DATA_DIR, OUTPUT_DIR
from src.eda import eda_figures
from src.backtest import run_backtest
from src.train import train_final
from src.predict import predict_final

def main():
    """每步失败立即停止，避免拿旧结果伪装成功。"""
    d=preprocess()
    x=make_features(d,True)
    validate_no_leakage(d,features=x,enriched=True)
    x.to_csv(PROCESSED_DATA_DIR/'features.csv',index_label='date')
    for name, fig in eda_figures(d).items():
        fig.write_html(OUTPUT_DIR/(name+".html"), include_plotlyjs=True)
    run_backtest(d)
    train_final()
    print(predict_final())

if __name__=='__main__':
    main()
