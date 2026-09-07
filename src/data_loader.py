"""数据发现、语义映射和完整原始数据审计。"""
import logging
import shutil
import pandas as pd
from .config import ROOT, DATA_DIR, RAW_DATA_DIR, OUTPUT_DIR
from .utils import setup_logging, save_json
ALIASES = {
 'date': ['report_date', 'mfd_date', 'date', 'trade_date'],
 'user_id': ['user_id', 'userid', 'uid'],
 'purchase': ['total_purchase_amt', 'total_purchase', 'purchase'],
 'redeem': ['total_redeem_amt', 'total_redeem', 'redeem'],
 'balance': ['tbalance', 'today_balance', 'balance'],
 'previous_balance': ['ybalance', 'yesterday_balance', 'previous_balance'],
 'yield_rate': ['mfd_daily_yield', 'yield_rate', 'daily_yield'],
 'yield_7d': ['mfd_7daily_yield', 'yield_7d'],
 'shibor_overnight': ['interest_o_n', 'shibor_overnight'],
 **{f'shibor_{n}{u.lower()}': [f'interest_{n}_{u.lower()}', f'shibor_{n}{u.lower()}'] for n,u in [(1,'W'),(2,'W'),(1,'M'),(3,'M'),(6,'M'),(9,'M'),(1,'Y')]}}
TABLES = ('user_profile', 'user_balance', 'share_interest', 'bank_shibor')

def discover():
    """优先发现 data 下 CSV；首次将用户原始文件复制到 data/raw，保留源文件。"""
    setup_logging()
    found = {}
    for key in TABLES + ('comp_predict',):
        paths = [p for p in DATA_DIR.rglob('*.csv') if key in p.stem and 'processed' not in p.parts]
        if not paths:
            source = [p for p in (ROOT / 'Purchase Redemption Data').glob('*.csv') if key in p.stem]
            if source:
                dest = RAW_DATA_DIR / source[0].name
                shutil.copy2(source[0], dest)
                paths = [dest]
                logging.info('原始文件复制到 %s，源文件保持不变', dest)
        if len(paths) > 1:
            raise ValueError(f'{key} 匹配多个文件，请只保留一个输入: {paths}')
        if not paths and key in TABLES:
            raise FileNotFoundError(f'缺少 {key} CSV，请放入 data/raw/')
        if paths:
            found[key] = paths[0]
    return found

def map_fields(df, key):
    """按明确语义别名映射，未知必要字段报错而不猜测。"""
    lower = {str(c).strip().lower(): c for c in df}
    mapping = {}
    for canonical, aliases in ALIASES.items():
        hits = [lower[a] for a in aliases if a in lower]
        if len(hits) > 1:
            raise ValueError(f'{key}: {canonical} 字段映射歧义 {hits}')
        if hits:
            mapping[hits[0]] = canonical
    needed = {'user_balance': ['date','user_id','purchase','redeem','balance'], 'user_profile':['user_id'], 'share_interest':['date','yield_rate'], 'bank_shibor':['date','shibor_overnight']}[key]
    missing = set(needed) - set(mapping.values())
    if missing:
        raise ValueError(f'{key} 缺少可识别字段 {missing}；实际字段 {list(df)}')
    logging.info('%s 字段映射: %s', key, mapping)
    return df.rename(columns=mapping), mapping

def load_tables(audit=True):
    """读取全表并记录 shape、类型、缺失、重复、统计、原始日期范围。"""
    tables, reports = {}, {}
    for key, path in discover().items():
        if key == 'comp_predict':
            d = pd.read_csv(path, header=None, names=['date','purchase','redeem'])
            reports[key] = {'file':path.name,'shape':list(d.shape),'head':d.to_dict('records'),'role':'无表头格式样例，隔离，绝不参与训练或评分'}
            continue
        raw = pd.read_csv(path)
        d, mapping = map_fields(raw, key)
        report = {'file':path.name,'shape':list(raw.shape),'columns':list(raw),'dtypes':raw.dtypes.astype(str).to_dict(),'head':raw.head().to_dict('records'),'missing_rate':raw.isna().mean().to_dict(),'duplicates':int(raw.duplicated().sum()),'mapping':mapping,'statistics':raw.describe(include='all').to_dict()}
        if 'date' in d:
            d['date'] = pd.to_datetime(d['date'].astype(str), format='%Y%m%d', errors='coerce')
            report['date_range'] = [str(d.date.min()), str(d.date.max())]
            report['invalid_dates'] = int(d.date.isna().sum())
            if d.date.isna().any():
                raise ValueError(f'{key} 含无效日期，请检查原始数据')
        reports[key], tables[key] = report, d
        logging.info('%s shape=%s 日期=%s 缺失=%s 重复=%s', key, raw.shape, report.get('date_range'), raw.isna().sum().to_dict(), report['duplicates'])
    if audit:
        save_json(reports, OUTPUT_DIR / 'data_quality_report.json')
    return tables
