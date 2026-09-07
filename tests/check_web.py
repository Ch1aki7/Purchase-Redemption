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
Path('output/web_smoke_test.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
assert not any(results.values())

