import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for file in (root/'workflows').glob('*.json'):
    d=json.loads(file.read_text(encoding='utf-8'))
    n={node['id']:node for node in d['nodes']}
    assert len(n)==len(d['nodes'])
    for link in d['links']:
        k,a,ao,b,bi,t=link
        assert n[a]['outputs'][ao]['type']==t
        assert n[b]['inputs'][bi]['type']==t
        assert k in n[a]['outputs'][ao]['links']
        assert n[b]['inputs'][bi]['link']==k
    print('VALID',file.name)
