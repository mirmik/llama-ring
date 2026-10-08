#!/usr/bin/env python3
"""Fill a shadow context, save/restore while summary is pending, then cross a fold."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:18098')
p.add_argument('--output',required=True)
p.add_argument('--tokens',type=int,default=70000)
p.add_argument('--tail-tokens',type=int,default=26000)
a=p.parse_args()
def post(path, body):
    request=urllib.request.Request(a.url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as r:return json.load(r)
def chat(messages):
    return post('/v1/chat/completions',dict(model='vcortex-shadow',id_slot=0,messages=messages,max_tokens=1,temperature=0))
block='Status record: CEDAR is healthy.\n'
n=len(post('/tokenize',{'content':block})['tokens'])
messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':block*(a.tokens//n)}]
start=time.monotonic();first=chat(messages);prefill=time.monotonic()-start
start=time.monotonic();saved=post('/slots/0?action=save',{'filename':'long-pending'});save_s=time.monotonic()-start
assert first['shadow']['active_tokens']>63079 and first['shadow']['commits']==0
post('/slots/0?action=erase',{})
chat([{'role':'user','content':'Reply OK.'}])
start=time.monotonic();restored=post('/slots/0?action=restore',{'filename':'long-pending'});restore_s=time.monotonic()-start
again=chat(messages)
assert again['cache_reused'] and again['context_state']['context_id']==first['context_state']['context_id']
assert again['choices'][0]['message']==first['choices'][0]['message']
messages += [again['choices'][0]['message'],{'role':'user','content':block*(a.tail_tokens//n)}]
folded=chat(messages)
assert folded['cache_reused'] and folded['shadow']['commits']>=1
assert folded['context_state']['context_id']==first['context_state']['context_id']
result={'passed':True,'prefill_s':prefill,'save_s':save_s,'restore_s':restore_s,'saved':saved,'restored':restored,
        'before':first['shadow'],'after':folded['shadow'],'continuation_equal':True}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('passed','prefill_s','save_s','restore_s','saved')}))
post('/slots/0?action=erase',{})
