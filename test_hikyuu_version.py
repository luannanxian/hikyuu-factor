#!/usr/bin/env python3
"""
测试Hikyuu版本脚本
"""
import sys
import os

# 临时清除PYTHONPATH中的hikyuu路径
original_path = sys.path.copy()
sys.path = [p for p in sys.path if '/Users/zhenkunliu/hikyuu' not in p]

try:
    import hikyuu as hku
    print(f"✅ Hikyuu version: {hku.__version__}")
    print(f"📍 Hikyuu path: {hku.__file__}")

    # 检查是否是pip安装的版本
    if 'site-packages' in hku.__file__:
        print("✅ 使用的是pip安装的Hikyuu版本")
    else:
        print("⚠️  使用的是本地编译的Hikyuu版本")

except ImportError as e:
    print(f"❌ 无法导入Hikyuu: {e}")

# 恢复原始路径
sys.path = original_path