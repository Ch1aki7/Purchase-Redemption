import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
a=AppTest.from_file(str(Path('app.py').resolve())).run(timeout=45)
result={}
for name in ['首页 / 项目概览','峰谷优化','未来预测','误差分析']:
 a.sidebar.radio[0].set_value(name).run(timeout=45)
 result[name]=[str(e.value) for e in a.exception]
 print(name,result[name])
a.sidebar.radio[0].set_value('未来预测').run(timeout=45)
a.radio[0].set_value('峰谷实验候选（已冻结）').run(timeout=45)
result['优化候选切换']=[str(e.value) for e in a.exception]
print('候选切换',result['优化候选切换'])
Path('output/optimization/web_tests.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
assert not any(result.values())
