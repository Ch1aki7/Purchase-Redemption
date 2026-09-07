"""首轮检查入口，复用正式审计以正确识别无表头提交样例。"""
from src.data_loader import load_tables

if __name__ == '__main__':
    load_tables()
