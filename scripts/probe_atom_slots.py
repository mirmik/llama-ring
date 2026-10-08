#!/usr/bin/env python3
"""Exercise real llm-proxy SlotManager against an isolated shadow HTTP server."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import aiohttp

async def run(args):
    sys.path.insert(0, args.proxy)
    from llm_proxy.slot_manager import SlotManager, _cache_filename
    sm = SlotManager(args.cache, checkpoint_interval_minutes=None, checkpoint_max_requests=None)
    sm.init_model('shadow-test', 1)
    base = f'http://127.0.0.1:{args.port}'
    records = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as client:
        async def post(path, body):
            async with client.post(base + path, json=body) as r:
                result = await r.json()
                assert r.status == 200, result
                return result
        async def chat(session, history, maximum=256, thinking=True, tools=None):
            slot = await sm.acquire('shadow-test', session, args.port)
            success = False
            try:
                body = dict(model='vcortex-shadow', id_slot=slot, messages=history, max_tokens=maximum,
                            temperature=0, chat_template_kwargs={'enable_thinking': thinking})
                if tools: body['tools'] = tools
                result = await post('/v1/chat/completions', body)
                success = True
                records.append({'session': session, 'reused': result['cache_reused'],
                                'context': result['context_state'], 'finish': result['choices'][0]['finish_reason'],
                                'message': result['choices'][0]['message']})
                return result
            finally:
                await sm.finalize_request('shadow-test', args.port, slot, success=success)
        history = [{'role':'system','content':'You are a helpful assistant. Answer briefly.'},
                   {'role':'user','content':'What is 2+3?'}]
        first = await chat('atom-test', history, 512)
        assert first['choices'][0]['message'].get('reasoning_content'), first
        history += [first['choices'][0]['message'], {'role':'user','content':'Now what is 3+4?'}]
        second = await chat('atom-test', history, 512)
        assert second['cache_reused'], 'Thinking invalidated the first history prefix'
        context_id = second['context_state']['context_id']
        history += [second['choices'][0]['message'], {'role':'user','content':'Now what is 4+5?'}]
        # Complete a different session on the same sole slot.
        await chat('subagent-test', [{'role':'user','content':'Reply with OK.'}], 16, False)
        third = await chat('atom-test', history, 512)
        assert third['cache_reused'] and third['context_state']['context_id'] == context_id, 'Full prefill after slot restore'
        checkpoint = Path(args.cache)/'shadow-test'/_cache_filename('atom-test')
        assert checkpoint.exists() and checkpoint.stat().st_size > 0
        # Corruption and missing files must not delete a healthy resident slot.
        damaged = checkpoint.with_name('damaged')
        data = bytearray(checkpoint.read_bytes());data[-1] ^= 1;damaged.write_bytes(data)
        for filename in ('damaged', 'missing'):
            async with client.post(base+'/slots/0?action=restore', json={'filename': filename}) as r:
                assert r.status == 400
        history += [third['choices'][0]['message'], {'role':'user','content':'Say done.'}]
        fourth = await chat('atom-test', history, 128)
        assert fourth['cache_reused'] and fourth['context_state']['context_id']==context_id
        # Real function parsing, preserving whitespace in arguments.
        tool = {'type':'function','function':{'name':'write_file','description':'Write a file.',
                'parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},'required':['path','content']}}}
        tool_history = [{'role':'user','content':'Call write_file with path /tmp/test.txt and content exactly "  alpha\\n    beta\\n". Do not answer in text.'}]
        result = await chat('tools-test', tool_history, 1024, False, [tool])
        call = result['choices'][0]['message']['tool_calls'][0]
        values = json.loads(call['function']['arguments'])
        assert values == {'path':'/tmp/test.txt','content':'  alpha\n    beta\n'}, values
        return {'passed':True, 'checkpoint_bytes':checkpoint.stat().st_size, 'records':records}

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,default=18098)
    p.add_argument('--proxy', required=True, help='Path to the llm-proxy checkout')
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result=asyncio.run(run(args))
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'checkpoint_bytes':result['checkpoint_bytes']}))
