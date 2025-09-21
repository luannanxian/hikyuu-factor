"""
Test for T006: Setup pytest configuration in pytest.ini with markers and test discovery

This test validates that pytest is properly configured with:
- Custom pytest.ini configuration file
- Test markers for different test types
- Proper test discovery settings
- Coverage configuration
- Performance test settings
"""
import configparser
from pathlib import Path
import pytest
import subprocess
import sys


class TestPytestConfiguration:
    """Test suite for pytest configuration validation (T006)"""

    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path.cwd()
        self.pytest_ini_path = self.project_root / "pytest.ini"

        # Expected pytest markers for hikyuu-factor system
        self.expected_markers = [
            "unit",
            "integration",
            "contract",
            "performance",
            "slow",
            "requires_hikyuu",
            "requires_mysql",
            "requires_redis"
        ]

        # Expected pytest configuration sections
        self.expected_config_options = [
            "testpaths",
            "python_files",
            "python_functions",
            "python_classes",
            "addopts",
            "markers"
        ]

    def test_pytest_ini_exists(self):
        """Test that pytest.ini configuration file exists"""
        assert self.pytest_ini_path.exists(), "pytest.ini file should exist"
        assert self.pytest_ini_path.is_file(), "pytest.ini should be a file"

    def test_pytest_ini_valid_format(self):
        """Test that pytest.ini is valid INI format"""
        try:
            config = configparser.ConfigParser()
            config.read(self.pytest_ini_path)
            assert config.sections(), "pytest.ini should have sections"
        except configparser.Error as e:
            pytest.fail(f"pytest.ini is not valid INI format: {e}")
        except FileNotFoundError:
            pytest.fail("pytest.ini file does not exist")

    def test_pytest_tool_section_exists(self):
        """Test that [tool:pytest] section exists"""
        config = configparser.ConfigParser()
        config.read(self.pytest_ini_path)

        # Check for pytest section (can be [tool:pytest] or [pytest])
        has_pytest_section = any(
            section in ["tool:pytest", "pytest"] for section in config.sections()
        )

        assert has_pytest_section, "pytest.ini should have [tool:pytest] or [pytest] section"

    def test_required_configuration_options_exist(self):
        """Test that all required configuration options exist"""
        config = configparser.ConfigParser()
        config.read(self.pytest_ini_path)

        # Find pytest section
        pytest_section = None
        for section in config.sections():
            if section in ["tool:pytest", "pytest"]:
                pytest_section = section
                break

        assert pytest_section, "No pytest configuration section found"

        missing_options = []
        for option in self.expected_config_options:
            if not config.has_option(pytest_section, option):
                missing_options.append(option)

        assert not missing_options, f"Missing pytest configuration options: {missing_options}"

    def test_test_markers_properly_defined(self):
        """Test that all required test markers are defined"""
        config = configparser.ConfigParser()
        config.read(self.pytest_ini_path)

        # Find pytest section
        pytest_section = None
        for section in config.sections():
            if section in ["tool:pytest", "pytest"]:
                pytest_section = section
                break

        markers_option = config.get(pytest_section, "markers", fallback="")

        missing_markers = []
        for marker in self.expected_markers:
            if marker not in markers_option:
                missing_markers.append(marker)

        assert not missing_markers, f"Missing pytest markers: {missing_markers}"

    def test_test_discovery_configuration(self):
        """Test that test discovery is properly configured"""
        config = configparser.ConfigParser()
        config.read(self.pytest_ini_path)

        pytest_section = None
        for section in config.sections():
            if section in ["tool:pytest", "pytest"]:
                pytest_section = section
                break

        # Check test discovery patterns
        python_files = config.get(pytest_section, "python_files", fallback="")
        python_functions = config.get(pytest_section, "python_functions", fallback="")
        python_classes = config.get(pytest_section, "python_classes", fallback="")

        assert "test_*.py" in python_files, "Should discover test_*.py files"
        assert "test_*" in python_functions, "Should discover test_* functions"
        assert "Test*" in python_classes, "Should discover Test* classes"

        # Check testpaths
        testpaths = config.get(pytest_section, "testpaths", fallback="")
        assert "tests" in testpaths, "Should include tests directory in testpaths"

    def test_pytest_addopts_configuration(self):
        """Test that pytest addopts are properly configured"""
        config = configparser.ConfigParser()
        config.read(self.pytest_ini_path)

        pytest_section = None
        for section in config.sections():
            if section in ["tool:pytest", "pytest"]:
                pytest_section = section
                break

        addopts = config.get(pytest_section, "addopts", fallback="")

        # Check for important addopts
        expected_addopts = [
            "--strict-markers",  # Ensure all markers are defined
            "--strict-config",   # Strict configuration validation
            "--tb=short",        # Short traceback format
            "-v",                # Verbose output
        ]

        missing_addopts = []
        for opt in expected_addopts:
            if opt not in addopts:
                missing_addopts.append(opt)

        assert not missing_addopts, f"Missing important addopts: {missing_addopts}"

    def test_pytest_can_collect_tests(self):
        """Test that pytest can collect tests from the project"""
        try:
            # Run pytest --collect-only to test discovery
            result = subprocess.run([
                sys.executable, "-m", "pytest", "--collect-only", "-q"
            ], capture_output=True, text=True, cwd=self.project_root)

            # Should not fail with configuration errors
            assert result.returncode == 0 or "no tests ran" in result.stdout.lower(), \
                f"Pytest collection failed: {result.stderr}"

        except FileNotFoundError:
            pytest.skip("pytest not available in environment")

    def test_pytest_markers_work_correctly(self):
        """Test that pytest markers can be used for test selection"""
        try:
            # Test that we can select by marker (even if no tests match)
            for marker in ["unit", "integration", "contract"]:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", "-m", marker, "--collect-only", "-q"
                ], capture_output=True, text=True, cwd=self.project_root)

                # Should not fail with marker errors
                assert "INTERNALERROR" not in result.stderr, \
                    f"Marker '{marker}' caused internal error: {result.stderr}"

        except FileNotFoundError:
            pytest.skip("pytest not available in environment")

    def test_configuration_supports_hikyuu_requirements(self):
        """Test that configuration supports Hikyuu-specific requirements"""
        config = configparser.ConfigParser()
        config.read(self.pytest_ini_path)

        pytest_section = None
        for section in config.sections():
            if section in ["tool:pytest", "pytest"]:
                pytest_section = section
                break

        markers_option = config.get(pytest_section, "markers", fallback="")

        # Should have Hikyuu-specific markers
        hikyuu_markers = ["requires_hikyuu", "requires_mysql", "requires_redis"]
        missing_hikyuu_markers = []

        for marker in hikyuu_markers:
            if marker not in markers_option:
                missing_hikyuu_markers.append(marker)

        assert not missing_hikyuu_markers, \
            f"Missing Hikyuu-specific markers: {missing_hikyuu_markers}"

    def test_configuration_supports_performance_testing(self):
        """Test that configuration supports performance testing"""
        config = configparser.ConfigParser()
        config.read(self.pytest_ini_path)

        pytest_section = None
        for section in config.sections():
            if section in ["tool:pytest", "pytest"]:
                pytest_section = section
                break

        markers_option = config.get(pytest_section, "markers", fallback="")

        # Should have performance testing markers
        performance_markers = ["performance", "slow"]
        missing_performance_markers = []

        for marker in performance_markers:
            if marker not in markers_option:
                missing_performance_markers.append(marker)

        assert not missing_performance_markers, \
            f"Missing performance testing markers: {missing_performance_markers}"

    def test_pytest_ini_consistent_with_pyproject_toml(self):
        """Test that pytest.ini is consistent with pyproject.toml pytest config"""
        # Read pyproject.toml pytest configuration
        pyproject_path = self.project_root / "pyproject.toml"

        if not pyproject_path.exists():
            pytest.skip("pyproject.toml not found")

        try:
            import tomllib
            with open(pyproject_path, "rb") as f:
                pyproject_config = tomllib.load(f)

            pytest_config = pyproject_config.get("tool", {}).get("pytest", {}).get("ini_options", {})

            if not pytest_config:
                # If no pytest config in pyproject.toml, that's okay
                return

            # Read pytest.ini
            ini_config = configparser.ConfigParser()
            ini_config.read(self.pytest_ini_path)

            pytest_section = None
            for section in ini_config.sections():
                if section in ["tool:pytest", "pytest"]:
                    pytest_section = section
                    break

            # Check consistency for key options
            key_options = ["testpaths", "markers"]
            inconsistencies = []

            for option in key_options:
                if option in pytest_config:
                    ini_value = ini_config.get(pytest_section, option, fallback="")
                    toml_value = pytest_config[option]

                    # Basic consistency check (not exact match due to format differences)
                    if option == "testpaths":
                        if isinstance(toml_value, list):
                            toml_value = " ".join(toml_value)
                        if "tests" in toml_value and "tests" not in ini_value:
                            inconsistencies.append(f"{option}: pytest.ini missing 'tests'")

            assert not inconsistencies, f"Inconsistencies between pytest.ini and pyproject.toml: {inconsistencies}"

        except ImportError:
            pytest.skip("tomllib not available (Python 3.11+ required)")
        except Exception as e:
            pytest.fail(f"Error checking consistency: {e}")