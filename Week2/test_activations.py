"""批量测试不同激活函数的 MLP 模型"""
import subprocess
import re
import json
from pathlib import Path
import sys

from mnist_models import MLP_ACTIVATIONS

ROOT = Path(__file__).resolve().parent
results = []

print('=' * 60)
print('MLP 激活函数对比实验 (2 epochs, MNIST)')
print('=' * 60)

for act in MLP_ACTIVATIONS:
    print(f'\n>>> Testing {act}...', flush=True)
    result = subprocess.run(
        [sys.executable, 'mnist_models.py', '--model', 'mlp', '--activation', act,
         '--epochs', '2', '--output-dir', 'outputs/activations'],
        check=True, stdout=subprocess.PIPE, text=True, cwd=ROOT,
    )
    accuracy = re.findall(r'Validation Accuracy: (\d+\.\d+)%', result.stdout)[-1]
    results.append((act, float(accuracy)))
    print(f'    {act}: {accuracy}%')

print('\n' + '=' * 60)
print('结果汇总 (按准确率排序)')
print('=' * 60)
print(f"{'激活函数':<15} {'准确率':<10}")
print('-' * 30)
for act, acc in sorted(results, key=lambda x: x[1], reverse=True):
    print(f'{act:<15} {acc:.2f}%')

# 保存结果
output = ROOT / 'outputs' / 'activation_comparison.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2))
print(f'\n结果已保存到: {output}')
