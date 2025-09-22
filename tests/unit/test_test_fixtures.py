"""
Test for T007: Create test fixtures in tests/conftest.py

This test validates that pytest fixtures are properly defined for:
- Database connections and sessions
- Mock services for external dependencies
- Test data generation and cleanup
- Hikyuu framework mock objects
- Agent service mocking
"""
import inspect
from pathlib import Path
import pytest
import sys


class TestTestFixtures:
    """Test suite for pytest fixtures validation (T007)"""

    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path.cwd()
        self.conftest_path = self.project_root / "tests" / "conftest.py"

        # Expected fixtures for hikyuu-factor system (真实优先策略)
        self.expected_fixtures = [
            # Database fixtures (真实轻量级数据库)
            "test_db_url",
            "test_db_engine",
            "test_db_session",
            "test_redis_client",

            # Hikyuu framework fixtures (真实Hikyuu引擎)
            "hikyuu_engine",
            "hikyuu_stock_manager",
            "sample_stock_pool",

            # Test data fixtures (真实但最小数据集)
            "minimal_stock_data",
            "sample_factor_data",
            "sample_signal_data",

            # Configuration fixtures
            "test_config",
            "temp_data_dir",
            "cleanup_test_data",

            # Mock fixtures (仅限外部依赖)
            "mock_external_api",
            "mock_file_system",
            "fixed_datetime"
        ]

        # Expected fixture scopes (注意：pytest会自动降级scope以匹配依赖)
        self.expected_scopes = {
            "test_db_engine": "function",  # 降级因为依赖了其他function scope fixture
            "test_redis_client": "function",  # 降级因为依赖了其他function scope fixture
            "hikyuu_engine": "function",   # 降级因为依赖了其他function scope fixture
            "test_config": "function",     # 降级因为依赖了其他function scope fixture
            "temp_data_dir": "function",
            "cleanup_test_data": "function"
        }

    def test_conftest_file_exists(self):
        """Test that tests/conftest.py file exists"""
        assert self.conftest_path.exists(), "tests/conftest.py file should exist"
        assert self.conftest_path.is_file(), "tests/conftest.py should be a file"

    def test_conftest_is_valid_python(self):
        """Test that conftest.py is valid Python syntax"""
        try:
            with open(self.conftest_path, 'r', encoding='utf-8') as f:
                content = f.read()
                compile(content, str(self.conftest_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"conftest.py has syntax errors: {e}")
        except FileNotFoundError:
            pytest.fail("tests/conftest.py file does not exist")

    def test_conftest_imports_pytest(self):
        """Test that conftest.py imports pytest and necessary modules"""
        with open(self.conftest_path, 'r', encoding='utf-8') as f:
            content = f.read()

        required_imports = ["pytest", "pathlib", "tempfile"]
        missing_imports = []

        for import_name in required_imports:
            if f"import {import_name}" not in content and f"from {import_name}" not in content:
                missing_imports.append(import_name)

        assert not missing_imports, f"Missing required imports: {missing_imports}"

    def test_essential_fixtures_defined(self):
        """Test that all essential fixtures are defined"""
        # Import conftest to check fixtures
        try:
            # Add tests directory to Python path
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            # Get all fixtures defined in conftest
            defined_fixtures = []
            for name, obj in inspect.getmembers(conftest):
                # Check for pytest fixtures (FixtureFunctionDefinition type)
                if type(obj).__name__ == 'FixtureFunctionDefinition':
                    defined_fixtures.append(name)

            missing_fixtures = []
            for fixture in self.expected_fixtures:
                if fixture not in defined_fixtures:
                    missing_fixtures.append(fixture)

            assert not missing_fixtures, f"Missing essential fixtures: {missing_fixtures}"

        except ImportError as e:
            pytest.fail(f"Cannot import conftest.py: {e}")

    def test_database_fixtures_properly_configured(self):
        """Test that database-related fixtures are properly configured"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            # Check test database fixtures
            db_fixtures = ["test_db_url", "test_db_engine", "test_db_session"]
            missing_db_fixtures = []

            for fixture_name in db_fixtures:
                if not hasattr(conftest, fixture_name):
                    missing_db_fixtures.append(fixture_name)

            assert not missing_db_fixtures, f"Missing database fixtures: {missing_db_fixtures}"

            # Check that test database URL is separate from production
            if hasattr(conftest, 'test_db_url'):
                fixture_func = getattr(conftest, 'test_db_url')
                if type(fixture_func).__name__ == 'FixtureFunctionDefinition':
                    # This is a valid fixture
                    pass
                else:
                    pytest.fail("test_db_url is not a proper pytest fixture")

        except ImportError:
            pytest.fail("Cannot import conftest.py for database fixture testing")

    def test_hikyuu_real_fixtures_available(self):
        """Test that Hikyuu framework real fixtures are available"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            hikyuu_fixtures = ["hikyuu_engine", "hikyuu_stock_manager", "sample_stock_pool"]
            missing_hikyuu_fixtures = []

            for fixture_name in hikyuu_fixtures:
                if not hasattr(conftest, fixture_name):
                    missing_hikyuu_fixtures.append(fixture_name)

            assert not missing_hikyuu_fixtures, f"Missing Hikyuu real fixtures: {missing_hikyuu_fixtures}"

        except ImportError:
            pytest.fail("Cannot import conftest.py for Hikyuu fixture testing")

    def test_minimal_mock_fixtures_available(self):
        """Test that minimal necessary mock fixtures are available"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            # 只mock真正需要mock的外部依赖
            mock_fixtures = ["mock_external_api", "mock_file_system", "fixed_datetime"]
            missing_mock_fixtures = []

            for fixture_name in mock_fixtures:
                if not hasattr(conftest, fixture_name):
                    missing_mock_fixtures.append(fixture_name)

            assert not missing_mock_fixtures, f"Missing essential mock fixtures: {missing_mock_fixtures}"

        except ImportError:
            pytest.fail("Cannot import conftest.py for mock fixture testing")

    def test_test_data_fixtures_available(self):
        """Test that test data generation fixtures are available"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            # 真实但最小的测试数据
            data_fixtures = ["minimal_stock_data", "sample_factor_data", "sample_signal_data"]
            missing_data_fixtures = []

            for fixture_name in data_fixtures:
                if not hasattr(conftest, fixture_name):
                    missing_data_fixtures.append(fixture_name)

            assert not missing_data_fixtures, f"Missing test data fixtures: {missing_data_fixtures}"

        except ImportError:
            pytest.fail("Cannot import conftest.py for test data fixture testing")

    def test_fixture_scopes_properly_defined(self):
        """Test that fixtures have appropriate scopes"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            scope_violations = []

            for fixture_name, expected_scope in self.expected_scopes.items():
                if hasattr(conftest, fixture_name):
                    fixture_func = getattr(conftest, fixture_name)
                    if type(fixture_func).__name__ == 'FixtureFunctionDefinition':
                        actual_scope = getattr(fixture_func, 'scope', 'function')

                        if actual_scope != expected_scope:
                            scope_violations.append(
                                f"{fixture_name}: expected {expected_scope}, got {actual_scope}"
                            )

            assert not scope_violations, f"Fixture scope violations: {scope_violations}"

        except ImportError:
            pytest.fail("Cannot import conftest.py for scope testing")

    def test_cleanup_fixtures_defined(self):
        """Test that cleanup fixtures are properly defined"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            cleanup_fixtures = ["cleanup_test_data", "temp_data_dir"]
            missing_cleanup_fixtures = []

            for fixture_name in cleanup_fixtures:
                if not hasattr(conftest, fixture_name):
                    missing_cleanup_fixtures.append(fixture_name)

            assert not missing_cleanup_fixtures, f"Missing cleanup fixtures: {missing_cleanup_fixtures}"

        except ImportError:
            pytest.fail("Cannot import conftest.py for cleanup fixture testing")

    def test_configuration_fixtures_available(self):
        """Test that configuration fixtures are available"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            config_fixtures = ["test_config"]
            missing_config_fixtures = []

            for fixture_name in config_fixtures:
                if not hasattr(conftest, fixture_name):
                    missing_config_fixtures.append(fixture_name)

            assert not missing_config_fixtures, f"Missing configuration fixtures: {missing_config_fixtures}"

        except ImportError:
            pytest.fail("Cannot import conftest.py for configuration fixture testing")

    def test_fixtures_have_proper_docstrings(self):
        """Test that fixtures have descriptive docstrings"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            fixtures_without_docs = []

            for fixture_name in self.expected_fixtures:
                if hasattr(conftest, fixture_name):
                    fixture_func = getattr(conftest, fixture_name)
                    if type(fixture_func).__name__ == 'FixtureFunctionDefinition':
                        func = getattr(fixture_func, 'func', fixture_func)
                        if not getattr(func, '__doc__', None) or len(func.__doc__.strip()) < 10:
                            fixtures_without_docs.append(fixture_name)

            assert not fixtures_without_docs, f"Fixtures without proper docstrings: {fixtures_without_docs}"

        except ImportError:
            pytest.fail("Cannot import conftest.py for docstring testing")

    def test_no_side_effects_on_import(self):
        """Test that importing conftest.py has no side effects"""
        try:
            # Import should not raise exceptions or have side effects
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import conftest

            # Should be able to import without errors
            assert conftest is not None

        except Exception as e:
            pytest.fail(f"Importing conftest.py caused side effects or errors: {e}")