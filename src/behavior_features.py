"""日级行为与有限维度静态画像聚合。"""
import pandas as pd

def aggregate_behavior(balance, profile):
    """聚合原始金额，不删除异常；画像只用已有性别，避免全样本筛城市。"""
    b = balance.copy()
    b['has_purchase'] = b.purchase > 0
    b['has_redeem'] = b.redeem > 0
    b['active'] = b.has_purchase | b.has_redeem
    b['is_new_observed'] = b.date.eq(b.groupby('user_id').date.transform('min'))
    d = b.groupby('date').agg(total_purchase=('purchase','sum'), total_redeem=('redeem','sum'), daily_user_count=('user_id','nunique'), purchase_user_count=('has_purchase','sum'), redeem_user_count=('has_redeem','sum'), daily_active_users=('active','sum'), total_balance=('balance','sum'), avg_balance=('balance','mean'), median_balance=('balance','median'), new_observed_user_ratio=('is_new_observed','mean'))
    d['net_flow'] = d.total_purchase-d.total_redeem
    d['avg_purchase'] = d.total_purchase/d.daily_user_count
    d['avg_redeem'] = d.total_redeem/d.daily_user_count
    d['purchase_user_ratio'] = d.purchase_user_count/d.daily_user_count
    d['redeem_user_ratio'] = d.redeem_user_count/d.daily_user_count
    d['purchase_redeem_ratio'] = d.total_purchase/d.total_redeem.clip(lower=1)
    if 'sex' in profile:
        merged = b.loc[b.active,['user_id','date']].merge(profile[['user_id','sex']], on='user_id', how='left', validate='many_to_one')
        merged['sex_1'] = merged.sex.eq(1).where(merged.sex.notna())
        d['active_sex_1_ratio'] = merged.groupby('date').sex_1.mean()
    return d.sort_index()
