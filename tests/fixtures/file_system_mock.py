"""
文件系统Mock - File System Mock

模拟文件系统操作，避免测试污染真实文件系统
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from unittest.mock import Mock, patch, mock_open
import json
import csv
from datetime import datetime


class MockFileSystem:
    """
    文件系统模拟器

    在内存中模拟文件系统操作，支持文件读写、目录操作等
    """

    def __init__(self):
        """初始化文件系统模拟器"""
        self.files = {}  # 模拟文件内容
        self.directories = set()  # 模拟目录
        self.current_dir = "/"
        self.operations = []  # 操作历史

    def create_file(self, path: str, content: str = ""):
        """创建文件"""
        path = self._normalize_path(path)
        self.files[path] = content
        self._log_operation("create_file", path)

        # 确保父目录存在
        parent_dir = str(Path(path).parent)
        if parent_dir != "/":
            self.directories.add(parent_dir)

    def write_file(self, path: str, content: str, mode: str = "w"):
        """写入文件"""
        path = self._normalize_path(path)

        if mode == "w":
            self.files[path] = content
        elif mode == "a":
            existing_content = self.files.get(path, "")
            self.files[path] = existing_content + content
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self._log_operation("write_file", path)

    def read_file(self, path: str) -> str:
        """读取文件"""
        path = self._normalize_path(path)

        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")

        self._log_operation("read_file", path)
        return self.files[path]

    def delete_file(self, path: str):
        """删除文件"""
        path = self._normalize_path(path)

        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")

        del self.files[path]
        self._log_operation("delete_file", path)

    def create_directory(self, path: str):
        """创建目录"""
        path = self._normalize_path(path)
        self.directories.add(path)
        self._log_operation("create_directory", path)

    def delete_directory(self, path: str, recursive: bool = False):
        """删除目录"""
        path = self._normalize_path(path)

        if not recursive:
            # 检查目录是否为空
            for file_path in self.files:
                if file_path.startswith(path + "/"):
                    raise OSError(f"Directory not empty: {path}")

            for dir_path in self.directories:
                if dir_path.startswith(path + "/") and dir_path != path:
                    raise OSError(f"Directory not empty: {path}")

        if recursive:
            # 删除目录下的所有文件和子目录
            files_to_delete = [f for f in self.files if f.startswith(path + "/")]
            for file_path in files_to_delete:
                del self.files[file_path]

            dirs_to_delete = [d for d in self.directories if d.startswith(path + "/")]
            for dir_path in dirs_to_delete:
                self.directories.discard(dir_path)

        self.directories.discard(path)
        self._log_operation("delete_directory", path)

    def list_directory(self, path: str = None) -> List[str]:
        """列出目录内容"""
        if path is None:
            path = self.current_dir

        path = self._normalize_path(path)

        if path not in self.directories and path != "/":
            raise FileNotFoundError(f"Directory not found: {path}")

        items = []

        # 添加文件
        for file_path in self.files:
            if Path(file_path).parent == Path(path):
                items.append(Path(file_path).name)

        # 添加子目录
        for dir_path in self.directories:
            if Path(dir_path).parent == Path(path):
                items.append(Path(dir_path).name)

        self._log_operation("list_directory", path)
        return sorted(items)

    def file_exists(self, path: str) -> bool:
        """检查文件是否存在"""
        path = self._normalize_path(path)
        return path in self.files

    def directory_exists(self, path: str) -> bool:
        """检查目录是否存在"""
        path = self._normalize_path(path)
        return path in self.directories or path == "/"

    def get_file_size(self, path: str) -> int:
        """获取文件大小"""
        path = self._normalize_path(path)

        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")

        return len(self.files[path].encode('utf-8'))

    def _normalize_path(self, path: str) -> str:
        """标准化路径"""
        if not path.startswith("/"):
            path = self.current_dir + "/" + path

        return str(Path(path).resolve())

    def _log_operation(self, operation: str, path: str):
        """记录操作历史"""
        self.operations.append({
            "operation": operation,
            "path": path,
            "timestamp": datetime.now().isoformat()
        })

    def get_operations_history(self) -> List[Dict]:
        """获取操作历史"""
        return self.operations.copy()

    def reset(self):
        """重置文件系统"""
        self.files.clear()
        self.directories.clear()
        self.operations.clear()
        self.current_dir = "/"


def mock_file_operations(mock_fs: MockFileSystem = None) -> Dict[str, Mock]:
    """
    创建文件操作mock

    Args:
        mock_fs: 模拟文件系统实例

    Returns:
        包含各种文件操作mock的字典
    """
    if mock_fs is None:
        mock_fs = MockFileSystem()

    def mock_open_func(filename, mode='r', *args, **kwargs):
        """模拟open函数"""
        if 'r' in mode:
            if not mock_fs.file_exists(filename):
                raise FileNotFoundError(f"File not found: {filename}")
            content = mock_fs.read_file(filename)
            return mock_open(read_data=content)()
        elif 'w' in mode or 'a' in mode:
            def write_content(content):
                mock_fs.write_file(filename, content, mode.replace('b', ''))

            mock_file = Mock()
            mock_file.write = write_content
            mock_file.__enter__ = lambda self: self
            mock_file.__exit__ = lambda self, *args: None
            return mock_file

    def mock_os_path_exists(path):
        """模拟os.path.exists"""
        return mock_fs.file_exists(path) or mock_fs.directory_exists(path)

    def mock_os_listdir(path):
        """模拟os.listdir"""
        return mock_fs.list_directory(path)

    def mock_os_makedirs(path, exist_ok=False):
        """模拟os.makedirs"""
        if mock_fs.directory_exists(path) and not exist_ok:
            raise FileExistsError(f"Directory already exists: {path}")
        mock_fs.create_directory(path)

    def mock_os_remove(path):
        """模拟os.remove"""
        mock_fs.delete_file(path)

    def mock_shutil_rmtree(path):
        """模拟shutil.rmtree"""
        mock_fs.delete_directory(path, recursive=True)

    return {
        "open": mock_open_func,
        "os.path.exists": mock_os_path_exists,
        "os.listdir": mock_os_listdir,
        "os.makedirs": mock_os_makedirs,
        "os.remove": mock_os_remove,
        "shutil.rmtree": mock_shutil_rmtree,
        "mock_fs": mock_fs
    }


def mock_directory_operations(base_path: str = "/tmp/test") -> Dict[str, Any]:
    """
    创建目录操作mock

    Args:
        base_path: 基础路径

    Returns:
        目录操作mock字典
    """
    mock_fs = MockFileSystem()

    def create_directory_structure(structure: Dict[str, Any], parent_path: str = base_path):
        """创建目录结构"""
        for name, content in structure.items():
            path = f"{parent_path}/{name}"

            if isinstance(content, dict):
                # 这是一个目录
                mock_fs.create_directory(path)
                create_directory_structure(content, path)
            else:
                # 这是一个文件
                mock_fs.create_file(path, str(content))

    def get_directory_tree(path: str = base_path) -> Dict[str, Any]:
        """获取目录树结构"""
        tree = {}

        # 获取直接子项
        try:
            items = mock_fs.list_directory(path)
        except FileNotFoundError:
            return tree

        for item in items:
            item_path = f"{path}/{item}"

            if mock_fs.directory_exists(item_path):
                # 递归获取子目录
                tree[item] = get_directory_tree(item_path)
            else:
                # 文件
                tree[item] = mock_fs.read_file(item_path)

        return tree

    return {
        "create_structure": create_directory_structure,
        "get_tree": get_directory_tree,
        "mock_fs": mock_fs,
        "base_path": base_path
    }


def create_temp_workspace(prefix: str = "hikyuu_test_") -> Path:
    """
    创建临时工作空间

    Args:
        prefix: 临时目录前缀

    Returns:
        临时目录路径
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))

    # 创建常见的测试目录结构
    directories = [
        "data",
        "logs",
        "config",
        "cache",
        "temp"
    ]

    for dir_name in directories:
        (temp_dir / dir_name).mkdir(exist_ok=True)

    # 创建一些基础配置文件
    config_file = temp_dir / "config" / "test_config.json"
    config_data = {
        "database": {
            "url": "sqlite:///test.db",
            "echo": False
        },
        "logging": {
            "level": "INFO",
            "file": "logs/test.log"
        },
        "cache": {
            "type": "memory",
            "size": 1000
        }
    }

    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    return temp_dir


class FileWatcher:
    """
    文件监控模拟器

    模拟文件系统事件监控
    """

    def __init__(self, path: str):
        """
        初始化文件监控器

        Args:
            path: 监控路径
        """
        self.path = path
        self.events = []
        self.is_watching = False

    def start_watching(self):
        """开始监控"""
        self.is_watching = True
        self.events.clear()

    def stop_watching(self):
        """停止监控"""
        self.is_watching = False

    def simulate_event(self, event_type: str, file_path: str):
        """模拟文件事件"""
        if self.is_watching:
            event = {
                "type": event_type,  # created, modified, deleted, moved
                "path": file_path,
                "timestamp": datetime.now().isoformat()
            }
            self.events.append(event)

    def get_events(self) -> List[Dict]:
        """获取监控到的事件"""
        return self.events.copy()

    def clear_events(self):
        """清除事件历史"""
        self.events.clear()


def create_test_files(directory: Path, file_specs: Dict[str, str]):
    """
    在指定目录创建测试文件

    Args:
        directory: 目标目录
        file_specs: 文件规格 {文件名: 内容}
    """
    directory.mkdir(parents=True, exist_ok=True)

    for filename, content in file_specs.items():
        file_path = directory / filename

        # 根据文件扩展名处理不同格式
        if filename.endswith('.json'):
            with open(file_path, 'w') as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    json.dump(content, f, indent=2)
        elif filename.endswith('.csv'):
            with open(file_path, 'w') as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    # 假设content是包含CSV数据的列表
                    writer = csv.writer(f)
                    for row in content:
                        writer.writerow(row)
        else:
            with open(file_path, 'w') as f:
                f.write(str(content))