"""从当前正文提取短例执行，检查概率链与移位；仅CPU构造例。"""
from pathlib import Path
import contextlib, hashlib, io, json, math, re
from paths import DRAFT, RESEARCH as research
chapter = DRAFT / 'ch01-先定义一台翻译机.md'
source = chapter.read_text()
snippets = re.findall(r'```python\n(.*?)```', source, re.S)
assert len(snippets) == 1
output = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(output):
    exec(compile(snippets[0], str(chapter), 'exec'), namespace)
assert namespace['decoder_input'] == ['<bos>', 'i', 'love', 'cats']
assert namespace['target'] == ['i', 'love', 'cats', '<eos>']
assert math.isclose(math.prod(namespace['correct_probs']), .3024)
assert math.isclose(-math.log(.3024), -sum(map(math.log,namespace['correct_probs'])))
report = dict(kind='CPU构造例；不支持翻译性能结论',checks=4,stdout=output.getvalue(),snippet_sha256=hashlib.sha256(snippets[0].encode()).hexdigest())
(research/'ch01-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(output.getvalue())
