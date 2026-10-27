"""批量测试不同激活函数的 MLP 模型"""
import subprocess
import re
import json
from pathlib import Path
import os

activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu', 'gelu', 'silu', 'mish', 'softplus']
results = []

# 获取 uv run 的 python 路径
uv_python = str(Path('.venv/Scripts/python.exe').resolve())

print('=' * 60)
print('MLP 激活函数对比实验 (2 epochs, MNIST)')
print('=' * 60)

for act in activations:
    print(f'\n>>> Testing {act}...', flush=True)
    result = subprocess.run(
        [uv_python, 'mnist_models.py', '--model', 'mlp', '--activation', act, '--epochs', '2'],
        capture_output=True, text=True, cwd='.'
    )
    # 打印错误以便调试
    if result.stderr:
        print(f"stderr: {result.stderr[:200]}")
    
    # 提取最后一行准确率
    for line in result.stdout.split('\n'):
        if 'Validation Accuracy' in line:
            acc = re.search(r'(\d+\.\d+)%', line)
            if acc:
                results.append((act, float(acc.group(1))))
                print(f'    {act}: {acc.group(1)}%')

print('\n' + '=' * 60)
print('结果汇总 (按准确率排序)')
print('=' * 60)
print(f"{'激活函数':<15} {'准确率':<10}")
print('-' * 30)
for act, acc in sorted(results, key=lambda x: x[1], reverse=True):
    print(f'{act:<15} {acc:.2f}%')

# 保存结果
output = Path('outputs/activation_comparison.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2))
print(f'\n结果已保存到: {output}')
