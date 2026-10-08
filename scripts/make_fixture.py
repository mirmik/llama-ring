#!/usr/bin/env python3
"""Synthetic immutable-code stream; labels are only used by query scoring."""
import argparse
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--seed', type=int, default=29)
args = parser.parse_args()
names = ['Липа', 'Кедр', 'Ива', 'Дуб', 'Клён', 'Сосна', 'Ольха', 'Ель', 'Берёза', 'Осина', 'Ясень', 'Тополь']
codes = random.Random(args.seed).sample(range(1000, 10000), len(names))
blocks = []
for i in range(24):
    fact = i if i < 8 else i - 8 if i < 12 else i - 4 if i < 16 else None
    block = {'text': f'Код объекта {names[fact]}: {codes[fact]}.' if fact is not None else 'Наблюдение: погода ясная, ветер слабый.'}
    if i in (7, 15, 23):
        block['questions'] = [{'text': f'Какой код у объекта {name}? Ответь только четырьмя цифрами.', 'expected': str(code)}
                              for name, code in zip(names[:8 if i == 7 else 12], codes)]
    blocks.append(block)
data = {'seed': args.seed, 'block_tokens': 1024,
        'system': 'Веди журнал сведений об объектах. Отвечай на вопрос о коде только четырьмя цифрами. '
                  'Если код неизвестен, ответь UNKNOWN. Последующие сообщения «Дополнение памяти» '
                  'содержат сведения из прежней части журнала. Учитывай их вместе с текущим журналом.',
        'blocks': blocks}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
