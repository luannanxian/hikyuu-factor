"""
Factor Definition Models
因子定义相关的数据模型，支持Hikyuu框架集成和版本管理
"""

import enum
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import uuid


class FactorType(enum.Enum):
    """因子类型枚举"""
    MOMENTUM = "momentum"  # 动量因子
    TECHNICAL = "technical"  # 技术因子
    VALUATION = "valuation"  # 估值因子
    QUALITY = "quality"  # 质量因子
    GROWTH = "growth"  # 成长因子
    SENTIMENT = "sentiment"  # 情绪因子
    RISK = "risk"  # 风险因子
    MACRO = "macro"  # 宏观因子
    CROSS_SECTIONAL = "cross_sectional"  # 横截面因子
    TIME_SERIES = "time_series"  # 时间序列因子
    COMPOSITE = "composite"  # 复合因子
    CUSTOM = "custom"  # 自定义因子


class FactorDataType(enum.Enum):
    """因子数据类型"""
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CATEGORY = "category"
    PERCENTAGE = "percentage"
    RATIO = "ratio"


class FactorFrequency(enum.Enum):
    """因子计算频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    INTRADAY = "intraday"


class FactorStatus(enum.Enum):
    """因子状态"""
    DRAFT = "draft"  # 草稿
    TESTING = "testing"  # 测试中
    ACTIVE = "active"  # 激活
    DEPRECATED = "deprecated"  # 已弃用
    ARCHIVED = "archived"  # 已归档


@dataclass
class FactorParameter:
    """因子参数定义"""
    name: str
    data_type: str  # int, float, str, bool, list
    default_value: Any
    description: str
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True

    def validate_value(self, value: Any) -> bool:
        """验证参数值"""
        if value is None and self.required:
            return False

        if value is None and not self.required:
            return True

        # 类型检查
        if self.data_type == "int" and not isinstance(value, int):
            return False
        elif self.data_type == "float" and not isinstance(value, (int, float)):
            return False
        elif self.data_type == "str" and not isinstance(value, str):
            return False
        elif self.data_type == "bool" and not isinstance(value, bool):
            return False
        elif self.data_type == "list" and not isinstance(value, list):
            return False

        # 范围检查
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False

        # 允许值检查
        if self.allowed_values is not None and value not in self.allowed_values:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "default_value": self.default_value,
            "description": self.description,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "allowed_values": self.allowed_values,
            "required": self.required
        }


@dataclass
class FactorDependency:
    """因子依赖定义"""
    dependency_type: str  # factor, indicator, data_source
    dependency_id: str
    version: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dependency_type": self.dependency_type,
            "dependency_id": self.dependency_id,
            "version": self.version,
            "parameters": self.parameters
        }


@dataclass
class FactorMetadata:
    """因子元数据"""
    author: str
    created_at: datetime
    updated_at: datetime
    description: str
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source_paper: Optional[str] = None  # 来源论文
    implementation_notes: Optional[str] = None
    performance_notes: Optional[str] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "source_paper": self.source_paper,
            "implementation_notes": self.implementation_notes,
            "performance_notes": self.performance_notes,
            "custom_fields": self.custom_fields
        }


@dataclass
class FactorVersion:
    """因子版本信息"""
    version: str
    created_at: datetime
    changes: str
    checksum: str
    status: FactorStatus = FactorStatus.DRAFT

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "changes": self.changes,
            "checksum": self.checksum,
            "status": self.status.value
        }


@dataclass
class FactorDefinition:
    """因子定义主类"""
    factor_id: str
    factor_name: str
    factor_type: FactorType
    data_type: FactorDataType
    frequency: FactorFrequency
    parameters: List[FactorParameter] = field(default_factory=list)
    dependencies: List[FactorDependency] = field(default_factory=list)
    hikyuu_formula: Optional[str] = None  # Hikyuu公式表达式
    custom_function: Optional[str] = None  # 自定义函数名
    metadata: Optional[FactorMetadata] = None
    versions: List[FactorVersion] = field(default_factory=list)
    current_version: str = "1.0.0"
    status: FactorStatus = FactorStatus.DRAFT

    def __post_init__(self):
        """初始化后处理"""
        if not self.factor_id:
            self.factor_id = self._generate_factor_id()

        if not self.versions:
            self._create_initial_version()

        if not self.metadata:
            self.metadata = FactorMetadata(
                author="system",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                description=f"Factor {self.factor_name}"
            )

    def _generate_factor_id(self) -> str:
        """生成因子ID"""
        # 基于因子名称和类型生成唯一ID
        content = f"{self.factor_name}_{self.factor_type.value}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _create_initial_version(self) -> None:
        """创建初始版本"""
        checksum = self._calculate_checksum()
        initial_version = FactorVersion(
            version=self.current_version,
            created_at=datetime.now(),
            changes="Initial version",
            checksum=checksum,
            status=self.status
        )
        self.versions.append(initial_version)

    def _calculate_checksum(self) -> str:
        """计算因子定义的校验和"""
        # 包含所有关键字段的内容
        content = {
            "factor_name": self.factor_name,
            "factor_type": self.factor_type.value,
            "data_type": self.data_type.value,
            "frequency": self.frequency.value,
            "parameters": [p.to_dict() for p in self.parameters],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "hikyuu_formula": self.hikyuu_formula,
            "custom_function": self.custom_function
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def add_parameter(self, parameter: FactorParameter) -> None:
        """添加参数"""
        # 检查参数名是否重复
        if any(p.name == parameter.name for p in self.parameters):
            raise ValueError(f"Parameter {parameter.name} already exists")
        self.parameters.append(parameter)

    def add_dependency(self, dependency: FactorDependency) -> None:
        """添加依赖"""
        self.dependencies.append(dependency)

    def validate_parameters(self, param_values: Dict[str, Any]) -> List[str]:
        """验证参数值，返回错误信息列表"""
        errors = []

        # 检查必需参数
        for param in self.parameters:
            if param.required and param.name not in param_values:
                errors.append(f"Required parameter {param.name} is missing")
                continue

            if param.name in param_values:
                value = param_values[param.name]
                if not param.validate_value(value):
                    errors.append(f"Invalid value for parameter {param.name}: {value}")

        return errors

    def get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数值"""
        return {param.name: param.default_value for param in self.parameters}

    def create_new_version(self, changes: str) -> str:
        """创建新版本"""
        # 解析当前版本号
        current_parts = self.current_version.split('.')
        major, minor, patch = map(int, current_parts)

        # 递增版本号（这里简单地递增patch版本）
        new_version = f"{major}.{minor}.{patch + 1}"

        # 计算新的校验和
        checksum = self._calculate_checksum()

        # 创建新版本记录
        new_version_record = FactorVersion(
            version=new_version,
            created_at=datetime.now(),
            changes=changes,
            checksum=checksum,
            status=FactorStatus.TESTING
        )

        self.versions.append(new_version_record)
        self.current_version = new_version

        # 更新元数据
        if self.metadata:
            self.metadata.updated_at = datetime.now()

        return new_version

    def get_version(self, version: str) -> Optional[FactorVersion]:
        """获取指定版本"""
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_latest_version(self) -> FactorVersion:
        """获取最新版本"""
        if not self.versions:
            self._create_initial_version()
        return max(self.versions, key=lambda v: v.created_at)

    def activate_version(self, version: str) -> bool:
        """激活指定版本"""
        version_record = self.get_version(version)
        if not version_record:
            return False

        # 将其他版本状态设为非激活
        for v in self.versions:
            if v.status == FactorStatus.ACTIVE:
                v.status = FactorStatus.DEPRECATED

        # 激活指定版本
        version_record.status = FactorStatus.ACTIVE
        self.current_version = version
        self.status = FactorStatus.ACTIVE

        return True

    def get_hikyuu_expression(self, parameters: Optional[Dict[str, Any]] = None) -> str:
        """获取Hikyuu表达式"""
        if not self.hikyuu_formula:
            return ""

        # 使用提供的参数或默认参数
        params = parameters or self.get_default_parameters()

        # 简单的模板替换
        expression = self.hikyuu_formula
        for name, value in params.items():
            placeholder = f"{{{name}}}"
            expression = expression.replace(placeholder, str(value))

        return expression

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "factor_id": self.factor_id,
            "factor_name": self.factor_name,
            "factor_type": self.factor_type.value,
            "data_type": self.data_type.value,
            "frequency": self.frequency.value,
            "parameters": [p.to_dict() for p in self.parameters],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "hikyuu_formula": self.hikyuu_formula,
            "custom_function": self.custom_function,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "versions": [v.to_dict() for v in self.versions],
            "current_version": self.current_version,
            "status": self.status.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorDefinition':
        """从字典创建"""
        # 解析参数
        parameters = [FactorParameter(**p) for p in data.get("parameters", [])]

        # 解析依赖
        dependencies = [FactorDependency(**d) for d in data.get("dependencies", [])]

        # 解析元数据
        metadata = None
        if data.get("metadata"):
            metadata_data = data["metadata"].copy()
            metadata_data["created_at"] = datetime.fromisoformat(metadata_data["created_at"])
            metadata_data["updated_at"] = datetime.fromisoformat(metadata_data["updated_at"])
            metadata = FactorMetadata(**metadata_data)

        # 解析版本
        versions = []
        for v_data in data.get("versions", []):
            v_data_copy = v_data.copy()
            v_data_copy["created_at"] = datetime.fromisoformat(v_data_copy["created_at"])
            v_data_copy["status"] = FactorStatus(v_data_copy["status"])
            versions.append(FactorVersion(**v_data_copy))

        return cls(
            factor_id=data["factor_id"],
            factor_name=data["factor_name"],
            factor_type=FactorType(data["factor_type"]),
            data_type=FactorDataType(data["data_type"]),
            frequency=FactorFrequency(data["frequency"]),
            parameters=parameters,
            dependencies=dependencies,
            hikyuu_formula=data.get("hikyuu_formula"),
            custom_function=data.get("custom_function"),
            metadata=metadata,
            versions=versions,
            current_version=data.get("current_version", "1.0.0"),
            status=FactorStatus(data.get("status", "draft"))
        )

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'FactorDefinition':
        """从JSON字符串创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, file_path: Union[str, Path]) -> None:
        """保存到文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'FactorDefinition':
        """从文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())


class FactorRegistry:
    """因子注册表"""

    def __init__(self):
        self.factors: Dict[str, FactorDefinition] = {}

    def register(self, factor: FactorDefinition) -> bool:
        """注册因子"""
        if factor.factor_id in self.factors:
            return False

        self.factors[factor.factor_id] = factor
        return True

    def get(self, factor_id: str) -> Optional[FactorDefinition]:
        """获取因子"""
        return self.factors.get(factor_id)

    def list_factors(self, factor_type: Optional[FactorType] = None,
                     status: Optional[FactorStatus] = None) -> List[FactorDefinition]:
        """列出因子"""
        factors = list(self.factors.values())

        if factor_type:
            factors = [f for f in factors if f.factor_type == factor_type]

        if status:
            factors = [f for f in factors if f.status == status]

        return factors

    def search(self, query: str) -> List[FactorDefinition]:
        """搜索因子"""
        query_lower = query.lower()
        results = []

        for factor in self.factors.values():
            if (query_lower in factor.factor_name.lower() or
                query_lower in factor.factor_id.lower() or
                (factor.metadata and query_lower in factor.metadata.description.lower())):
                results.append(factor)

        return results

    def update(self, factor: FactorDefinition) -> bool:
        """更新因子"""
        if factor.factor_id not in self.factors:
            return False

        self.factors[factor.factor_id] = factor
        return True

    def remove(self, factor_id: str) -> bool:
        """移除因子"""
        if factor_id not in self.factors:
            return False

        del self.factors[factor_id]
        return True

    def export_registry(self, file_path: Union[str, Path]) -> None:
        """导出注册表"""
        data = {
            "factors": [f.to_dict() for f in self.factors.values()],
            "exported_at": datetime.now().isoformat()
        }

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_registry(self, file_path: Union[str, Path]) -> int:
        """导入注册表，返回导入的因子数量"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for factor_data in data.get("factors", []):
            factor = FactorDefinition.from_dict(factor_data)
            if self.register(factor):
                count += 1

        return count