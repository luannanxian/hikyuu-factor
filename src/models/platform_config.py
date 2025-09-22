"""
Platform Configuration Models
平台配置相关的数据模型，包括平台检测、优化配置等
"""

import platform
import enum
import subprocess
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class PlatformType(enum.Enum):
    """平台类型枚举"""
    APPLE_SILICON = "apple_silicon"
    X86_64 = "x86_64"
    ARM64 = "arm64"
    UNKNOWN = "unknown"

    @classmethod
    def detect(cls) -> 'PlatformType':
        """检测当前平台类型"""
        machine = platform.machine().lower()
        system = platform.system().lower()

        if system == "darwin":
            if machine in ["arm64", "aarch64"]:
                return cls.APPLE_SILICON
            elif machine in ["x86_64", "amd64"]:
                return cls.X86_64
        elif system == "linux":
            if machine in ["arm64", "aarch64"]:
                return cls.ARM64
            elif machine in ["x86_64", "amd64"]:
                return cls.X86_64

        return cls.UNKNOWN

    def get_features(self) -> Dict[str, bool]:
        """获取平台支持的特性"""
        if self == self.APPLE_SILICON:
            return {
                "simd": True,
                "neon": True,
                "avx2": False,
                "sse4": False,
                "native_acceleration": True
            }
        elif self == self.X86_64:
            return {
                "simd": True,
                "neon": False,
                "avx2": True,
                "sse4": True,
                "native_acceleration": True
            }
        else:
            return {
                "simd": False,
                "neon": False,
                "avx2": False,
                "sse4": False,
                "native_acceleration": False
            }

    def get_optimization_flags(self) -> List[str]:
        """获取编译优化标志"""
        if self == self.APPLE_SILICON:
            return ["-march=armv8-a+simd", "-mtune=apple-a14", "-O3"]
        elif self == self.X86_64:
            return ["-march=native", "-mavx2", "-msse4.1", "-O3"]
        else:
            return ["-O2"]


@dataclass
class CPUInfo:
    """CPU信息"""
    architecture: str
    cores: int
    threads: int
    cache_size: Optional[str] = None
    features: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def detect(cls) -> 'CPUInfo':
        """检测CPU信息"""
        import psutil

        cpu_count = psutil.cpu_count(logical=False) or 1
        thread_count = psutil.cpu_count(logical=True) or 1
        arch = platform.machine()

        # 尝试获取CPU特性
        features = {}
        try:
            if platform.system() == "Darwin":
                # macOS系统信息
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.features"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    cpu_features = result.stdout.strip().split()
                    features = {
                        "sse4": any("SSE4" in f for f in cpu_features),
                        "avx2": "AVX2" in cpu_features,
                        "neon": arch.lower() in ["arm64", "aarch64"]
                    }
            elif platform.system() == "Linux":
                # Linux CPU信息
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    features = {
                        "sse4": "sse4_1" in cpuinfo or "sse4_2" in cpuinfo,
                        "avx2": "avx2" in cpuinfo,
                        "neon": "asimd" in cpuinfo or "neon" in cpuinfo
                    }
        except Exception:
            # 如果检测失败，使用默认值
            pass

        return cls(
            architecture=arch,
            cores=cpu_count,
            threads=thread_count,
            features=features
        )


@dataclass
class OptimizationConfig:
    """优化配置类"""
    platform_type: PlatformType
    enable_simd: bool = True
    enable_parallel: bool = True
    thread_count: Optional[int] = None
    memory_limit_mb: Optional[int] = None
    cache_size_mb: int = 256
    batch_size: int = 1000
    custom_flags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """初始化后验证"""
        self.validate()

    def validate(self) -> None:
        """验证配置参数"""
        if self.thread_count is not None and self.thread_count < 1:
            raise ValueError("thread_count must be >= 1")

        if self.memory_limit_mb is not None and self.memory_limit_mb < 128:
            raise ValueError("memory_limit_mb must be >= 128")

        if self.cache_size_mb < 64:
            raise ValueError("cache_size_mb must be >= 64")

        if self.batch_size < 100:
            raise ValueError("batch_size must be >= 100")

    @classmethod
    def auto_detect(cls, custom_overrides: Optional[Dict[str, Any]] = None) -> 'OptimizationConfig':
        """自动检测最佳配置"""
        platform_type = PlatformType.detect()
        cpu_info = CPUInfo.detect()

        # 默认配置
        config = {
            "platform_type": platform_type,
            "enable_simd": platform_type in [PlatformType.APPLE_SILICON, PlatformType.X86_64],
            "enable_parallel": True,
            "thread_count": min(cpu_info.cores, 8),  # 限制最大线程数
            "cache_size_mb": 256
        }

        # 根据平台调整
        if platform_type == PlatformType.APPLE_SILICON:
            config.update({
                "batch_size": 2000,  # Apple Silicon处理能力更强
                "cache_size_mb": 512,
                "custom_flags": ["-DUSE_NEON", "-DAPPLE_SILICON"]
            })
        elif platform_type == PlatformType.X86_64:
            config.update({
                "batch_size": 1500,
                "cache_size_mb": 384,
                "custom_flags": ["-DUSE_AVX2", "-DUSE_SSE4"]
            })

        # 应用自定义覆盖
        if custom_overrides:
            config.update(custom_overrides)

        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "platform_type": self.platform_type.value,
            "enable_simd": self.enable_simd,
            "enable_parallel": self.enable_parallel,
            "thread_count": self.thread_count,
            "memory_limit_mb": self.memory_limit_mb,
            "cache_size_mb": self.cache_size_mb,
            "batch_size": self.batch_size,
            "custom_flags": self.custom_flags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """从字典创建配置"""
        data = data.copy()
        if "platform_type" in data:
            data["platform_type"] = PlatformType(data["platform_type"])
        return cls(**data)

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'OptimizationConfig':
        """从JSON字符串创建配置"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, file_path: Union[str, Path]) -> None:
        """保存配置到文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'OptimizationConfig':
        """从文件加载配置"""
        with open(file_path, 'r') as f:
            return cls.from_json(f.read())

    def get_hikyuu_params(self) -> Dict[str, Any]:
        """获取Hikyuu框架的参数配置"""
        params = {
            "cpu_num": self.thread_count or CPUInfo.detect().cores,
            "use_parallel": self.enable_parallel,
            "cache_dir": f"cache_{self.cache_size_mb}mb"
        }

        if self.memory_limit_mb:
            params["max_memory_mb"] = self.memory_limit_mb

        return params

    def get_performance_multiplier(self) -> float:
        """获取性能倍数估算"""
        base_multiplier = 1.0

        # 平台基础性能
        if self.platform_type == PlatformType.APPLE_SILICON:
            base_multiplier *= 1.5  # Apple Silicon通常更快
        elif self.platform_type == PlatformType.X86_64:
            base_multiplier *= 1.2

        # SIMD加速
        if self.enable_simd:
            base_multiplier *= 1.3

        # 并行加速
        if self.enable_parallel and self.thread_count:
            # 并行效率通常不是线性的
            parallel_efficiency = min(self.thread_count * 0.8, 4.0)
            base_multiplier *= parallel_efficiency

        return base_multiplier


class PlatformDetector:
    """平台检测器"""

    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """获取完整的平台信息"""
        platform_type = PlatformType.detect()
        cpu_info = CPUInfo.detect()

        return {
            "platform_type": platform_type.value,
            "platform_features": platform_type.get_features(),
            "cpu_info": {
                "architecture": cpu_info.architecture,
                "cores": cpu_info.cores,
                "threads": cpu_info.threads,
                "features": cpu_info.features
            },
            "system": {
                "os": platform.system(),
                "version": platform.version(),
                "python_version": platform.python_version()
            },
            "optimization_flags": platform_type.get_optimization_flags()
        }

    @staticmethod
    def benchmark_platform() -> Dict[str, float]:
        """简单的平台性能基准测试"""
        import time
        import numpy as np

        # CPU计算测试
        start_time = time.time()
        data = np.random.randn(10000, 100)
        result = np.mean(data @ data.T)
        cpu_time = time.time() - start_time

        # 内存访问测试
        start_time = time.time()
        large_array = np.zeros((5000, 5000))
        large_array.fill(1.0)
        memory_time = time.time() - start_time

        return {
            "cpu_benchmark_seconds": cpu_time,
            "memory_benchmark_seconds": memory_time,
            "performance_score": 1.0 / (cpu_time + memory_time)
        }