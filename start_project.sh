#!/bin/bash

# Hikyuu Factor项目启动脚本
# 确保使用正确的Hikyuu版本（pip安装的2.6.8）

echo "🚀 启动Hikyuu Factor项目..."

# 临时清除PYTHONPATH中的本地hikyuu路径
export PYTHONPATH="$(pwd)/src"

# 检查Hikyuu版本
echo "📊 检查Hikyuu版本..."
python -c "
import sys
sys.path = [p for p in sys.path if '/Users/zhenkunliu/hikyuu' not in p]
import hikyuu as hku
print(f'Hikyuu version: {hku.__version__}')
print(f'Hikyuu path: {hku.__file__}')
if 'site-packages' in hku.__file__:
    print('✅ 使用pip安装的版本')
else:
    print('⚠️  使用本地编译版本')
"

echo ""
echo "🎯 可用的启动命令:"
echo "  数据管理Agent:    python -m agents.data_manager_agent"
echo "  因子计算Agent:    python -m agents.factor_calculation_agent"
echo "  验证Agent:        python -m agents.validation_agent"
echo "  信号生成Agent:    python -m agents.signal_generation_agent"
echo ""
echo "📝 项目路径: $(pwd)"
echo "🐍 Python路径: $(which python)"
echo "📦 PYTHONPATH: $PYTHONPATH"