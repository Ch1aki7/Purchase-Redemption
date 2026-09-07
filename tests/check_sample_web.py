import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
a=AppTest.from_file(str(Path('app.py').resolve())).run(timeout=60)
results={}
for page in ['多步样本验证','未来预测','峰谷优化','首页 / 项目概览']:
 a.sidebar.radio[0].set_value(page).run(timeout=60)
 results[page]=[str(e.value) for e in a.exception]
 print(page,results[page])
a.sidebar.radio[0].set_value('未来预测').run(timeout=60)
a.radio[0].set_value('多步样本研究候选（已冻结）').run(timeout=60)
results['多步候选切换']=[str(e.value) for e in a.exception]
Path('output/multistep/web_tests.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
print(results)
assert not any(results.values())
