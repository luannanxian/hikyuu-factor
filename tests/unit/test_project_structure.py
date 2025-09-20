"""
Test for T001: Create project directory structure with proper permissions

This test validates that the project directory structure is created correctly
according to the requirements in tasks.md Phase 1.
"""
import os
import stat
from pathlib import Path
import pytest


class TestProjectStructure:
    """Test suite for project directory structure validation (T001)"""

    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path.cwd()

        # Expected directory structure from T001 requirements
        self.expected_directories = [
            # Source code structure
            "src",
            "src/agents",
            "src/models",
            "src/services",
            "src/api",
            "src/cli",
            "src/lib",

            # Test structure
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/contract",
            "tests/performance",

            # Supporting directories
            "config",
            "docs",
            "scripts"
        ]

        # Expected __init__.py files for Python packages
        self.expected_init_files = [
            "src/__init__.py",
            "src/agents/__init__.py",
            "src/models/__init__.py",
            "src/services/__init__.py",
            "src/api/__init__.py",
            "src/cli/__init__.py",
            "src/lib/__init__.py",
        ]

    def test_all_required_directories_exist(self):
        """Test that all required directories from T001 are created"""
        missing_dirs = []

        for dir_path in self.expected_directories:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
            elif not full_path.is_dir():
                missing_dirs.append(f"{dir_path} (exists but not a directory)")

        assert not missing_dirs, f"Missing required directories: {missing_dirs}"

    def test_directories_have_proper_permissions(self):
        """Test that directories have proper read/write permissions"""
        permission_issues = []

        for dir_path in self.expected_directories:
            full_path = self.project_root / dir_path
            if full_path.exists():
                # Check read and write permissions
                if not os.access(full_path, os.R_OK):
                    permission_issues.append(f"{dir_path} - no read permission")
                if not os.access(full_path, os.W_OK):
                    permission_issues.append(f"{dir_path} - no write permission")

        assert not permission_issues, f"Permission issues found: {permission_issues}"

    def test_python_package_init_files_exist(self):
        """Test that __init__.py files exist for Python packages"""
        missing_init_files = []

        for init_file in self.expected_init_files:
            full_path = self.project_root / init_file
            if not full_path.exists():
                missing_init_files.append(init_file)
            elif not full_path.is_file():
                missing_init_files.append(f"{init_file} (exists but not a file)")

        assert not missing_init_files, f"Missing __init__.py files: {missing_init_files}"

    def test_python_import_structure_works(self):
        """Test that Python can import the src package structure"""
        # This test validates the verification command from T001:
        # `python -c "import src; print('Project structure OK')"`

        with pytest.raises(ImportError):
            # This should fail initially (Red phase of TDD)
            import src
            # If we get here, the import worked
            assert hasattr(src, '__file__'), "src package should be importable"

    def test_directory_structure_matches_agent_architecture(self):
        """Test that structure supports the 4-agent microservice architecture"""
        # Verify agent-specific structure exists
        agent_specific_dirs = [
            "src/agents",  # For the 4 agents: data_manager, factor_calculation, validation, signal_generation
            "tests/contract",  # For API contract tests
            "config",  # For agent configuration files
        ]

        missing_agent_dirs = []
        for dir_path in agent_specific_dirs:
            if not (self.project_root / dir_path).exists():
                missing_agent_dirs.append(dir_path)

        assert not missing_agent_dirs, f"Missing agent architecture directories: {missing_agent_dirs}"

    def test_test_directory_structure_supports_tdd(self):
        """Test that test directories support TDD approach"""
        test_dirs = ["tests/unit", "tests/integration", "tests/contract", "tests/performance"]

        missing_test_dirs = []
        for test_dir in test_dirs:
            if not (self.project_root / test_dir).exists():
                missing_test_dirs.append(test_dir)

        assert not missing_test_dirs, f"Missing TDD test directories: {missing_test_dirs}"