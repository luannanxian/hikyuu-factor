"""
Test for T004: Setup pyproject.toml with build system and development tools

This test validates that pyproject.toml is properly configured with:
- Build system configuration
- Development tool settings (black, mypy, flake8, isort)
- Project metadata
- Dependencies management
"""
import tomllib
from pathlib import Path
import pytest


class TestPyprojectTomlConfiguration:
    """Test suite for pyproject.toml configuration validation (T004)"""

    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path.cwd()
        self.pyproject_path = self.project_root / "pyproject.toml"

        # Expected sections in pyproject.toml
        self.expected_sections = [
            "build-system",
            "project",
            "tool.black",
            "tool.isort",
            "tool.mypy",
            "tool.pytest.ini_options"
        ]

        # Expected build system
        self.expected_build_system = {
            "requires": ["setuptools>=61.0", "wheel"],
            "build-backend": "setuptools.build_meta"
        }

        # Expected project metadata fields
        self.expected_project_fields = [
            "name",
            "version",
            "description",
            "authors",
            "license",
            "readme",
            "requires-python",
            "dependencies",
            "optional-dependencies"
        ]

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml file exists"""
        assert self.pyproject_path.exists(), "pyproject.toml file should exist"
        assert self.pyproject_path.is_file(), "pyproject.toml should be a file"

    def test_pyproject_toml_valid_format(self):
        """Test that pyproject.toml is valid TOML format"""
        try:
            with open(self.pyproject_path, "rb") as f:
                config = tomllib.load(f)
                assert isinstance(config, dict), "pyproject.toml should parse to a dictionary"
        except tomllib.TOMLDecodeError as e:
            pytest.fail(f"pyproject.toml is not valid TOML: {e}")
        except FileNotFoundError:
            pytest.fail("pyproject.toml file does not exist")

    def test_required_sections_exist(self):
        """Test that all required sections exist in pyproject.toml"""
        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        missing_sections = []

        for section in self.expected_sections:
            # Handle nested sections like "tool.black"
            section_parts = section.split(".")
            current = config

            for part in section_parts:
                if part not in current:
                    missing_sections.append(section)
                    break
                current = current[part]

        assert not missing_sections, f"Missing required sections: {missing_sections}"

    def test_build_system_configuration(self):
        """Test that build-system is properly configured"""
        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        build_system = config.get("build-system", {})

        # Check required fields
        assert "requires" in build_system, "build-system.requires should be defined"
        assert "build-backend" in build_system, "build-system.build-backend should be defined"

        # Check build backend
        assert build_system["build-backend"] == self.expected_build_system["build-backend"], \
            f"Expected build-backend: {self.expected_build_system['build-backend']}"

        # Check that setuptools is in requires
        requires = build_system["requires"]
        assert any("setuptools" in req for req in requires), \
            "setuptools should be in build-system.requires"

    def test_project_metadata_complete(self):
        """Test that project metadata is complete"""
        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        project = config.get("project", {})
        missing_fields = []

        for field in self.expected_project_fields:
            if field not in project:
                missing_fields.append(field)

        assert not missing_fields, f"Missing project metadata fields: {missing_fields}"

        # Validate specific fields
        assert project["name"] == "hikyuu-factor", "Project name should be 'hikyuu-factor'"
        assert "量化因子" in project["description"] or "quantitative factor" in project["description"], \
            "Description should mention quantitative factors"
        assert project["requires-python"] >= "3.11", "Should require Python 3.11+"

    def test_development_tools_configured(self):
        """Test that development tools are properly configured"""
        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        tool = config.get("tool", {})

        # Test Black configuration
        black_config = tool.get("black", {})
        assert "line-length" in black_config, "Black line-length should be configured"
        assert black_config["line-length"] <= 120, "Line length should be reasonable (<=120)"

        # Test isort configuration
        isort_config = tool.get("isort", {})
        assert "profile" in isort_config, "isort profile should be configured"
        assert isort_config["profile"] == "black", "isort should be compatible with black"

        # Test mypy configuration
        mypy_config = tool.get("mypy", {})
        assert "python_version" in mypy_config, "mypy python_version should be specified"
        assert "strict" in mypy_config, "mypy strict mode should be configured"

        # Test pytest configuration
        pytest_config = tool.get("pytest", {}).get("ini_options", {})
        assert "testpaths" in pytest_config, "pytest testpaths should be configured"
        assert "tests" in pytest_config["testpaths"], "tests directory should be in testpaths"

    def test_dependencies_include_hikyuu_requirements(self):
        """Test that dependencies include Hikyuu framework requirements"""
        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        project = config.get("project", {})
        dependencies = project.get("dependencies", [])

        required_deps = ["hikyuu", "fastapi", "sqlalchemy", "pandas", "numpy"]
        missing_deps = []

        for dep in required_deps:
            if not any(dep in dependency.lower() for dependency in dependencies):
                missing_deps.append(dep)

        assert not missing_deps, f"Missing required dependencies: {missing_deps}"

    def test_optional_dependencies_for_development(self):
        """Test that optional dependencies include development tools"""
        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        project = config.get("project", {})
        optional_deps = project.get("optional-dependencies", {})

        # Should have development dependencies
        assert "dev" in optional_deps, "Should have 'dev' optional dependencies"

        dev_deps = optional_deps["dev"]
        dev_tools = ["pytest", "black", "mypy", "flake8", "isort"]

        missing_dev_tools = []
        for tool in dev_tools:
            if not any(tool in dep.lower() for dep in dev_deps):
                missing_dev_tools.append(tool)

        assert not missing_dev_tools, f"Missing development tools: {missing_dev_tools}"

    def test_supports_editable_install(self):
        """Test that configuration supports editable installation"""
        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        # Should have proper build system for editable installs
        build_system = config.get("build-system", {})
        assert "setuptools" in str(build_system.get("requires", [])), \
            "setuptools required for editable installs"

        # Project should have proper package discovery
        project = config.get("project", {})
        assert "name" in project, "Project name required for editable installs"