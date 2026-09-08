from streamlit.testing.v1 import AppTest
from pathlib import Path
import json
pages=['首页 / 项目概览','数据探索','特征分析','模型训练','模型评估','滚动回测','未来预测','误差分析']
results={}
a=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=45)
for page in pages:
 a.sidebar.radio[0].set_value(page).run(timeout=45)
 results[page]=[str(e.value) for e in a.exception]
 print(page,results[page])
a.sidebar.radio[0].set_value('模型训练').run(timeout=45)
a.button[0].click().run(timeout=60)
results['交互训练']= [str(e.value) for e in a.exception]
print('交互训练',results['交互训练'])
event_button=next(button for button in a.button if '运行正式模型' in button.label)
results['正式模型训练入口']=[] if event_button else ['未找到正式模型按钮']
print('正式模型训练入口',results['正式模型训练入口'])
for name in ['tc_comp_predict_table_event_purchase_only.csv','tc_comp_predict_table_event_redeem_only.csv']:
 assert (Path('output')/name).exists(),name
Path('output/web_smoke_test.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
assert not any(results.values())

