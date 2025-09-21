"""
Test for T008: Setup test utilities in tests/utils.py

This test validates that test utilities are properly implemented including:
- Custom assertions for quantitative finance
- Helper functions for data manipulation
- Mock factories for test objects
- Database testing utilities
- Hikyuu testing helpers
"""
import inspect
from pathlib import Path
import pytest
import sys


class TestTestUtilities:
    """Test suite for test utilities validation (T008)"""

    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path.cwd()
        self.utils_path = self.project_root / "tests" / "utils.py"

        # Expected utility functions for hikyuu-factor system
        self.expected_assertions = [
            "assert_factor_values_equal",
            "assert_signals_equal",
            "assert_dataframes_equal",
            "assert_price_data_valid",
            "assert_performance_within_bounds"
        ]

        self.expected_helpers = [
            "create_test_stock_data",
            "create_test_factor_data",
            "setup_test_database_tables",
            "cleanup_test_database",
            "wait_for_calculation",
            "compare_factor_results"
        ]

        self.expected_mock_factories = [
            "MockStockFactory",
            "MockFactorFactory",
            "MockSignalFactory",
            "MockHikyuuKDataFactory"
        ]

        self.expected_db_utilities = [
            "create_test_tables",
            "truncate_test_tables",
            "insert_test_data",
            "verify_database_state"
        ]

        # Expected imports in utils.py
        self.expected_imports = [
            "pandas",
            "numpy",
            "pytest",
            "datetime",
            "typing"
        ]

    def test_utils_file_exists(self):
        """Test that tests/utils.py file exists"""
        assert self.utils_path.exists(), "tests/utils.py file should exist"
        assert self.utils_path.is_file(), "tests/utils.py should be a file"

    def test_utils_is_valid_python(self):
        """Test that utils.py is valid Python syntax"""
        try:
            with open(self.utils_path, 'r', encoding='utf-8') as f:
                content = f.read()
                compile(content, str(self.utils_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"utils.py has syntax errors: {e}")
        except FileNotFoundError:
            pytest.fail("tests/utils.py file does not exist")

    def test_utils_imports_required_modules(self):
        """Test that utils.py imports all required modules"""
        with open(self.utils_path, 'r', encoding='utf-8') as f:
            content = f.read()

        missing_imports = []
        for import_name in self.expected_imports:
            if f"import {import_name}" not in content and f"from {import_name}" not in content:
                missing_imports.append(import_name)

        assert not missing_imports, f"Missing required imports: {missing_imports}"

    def test_custom_assertions_defined(self):
        """Test that custom assertion functions are defined"""
        try:
            # Add tests directory to Python path
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            missing_assertions = []
            for assertion_name in self.expected_assertions:
                if not hasattr(utils, assertion_name):
                    missing_assertions.append(assertion_name)

            assert not missing_assertions, f"Missing custom assertions: {missing_assertions}"

        except ImportError as e:
            pytest.fail(f"Cannot import utils.py: {e}")

    def test_helper_functions_defined(self):
        """Test that helper functions are defined"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            missing_helpers = []
            for helper_name in self.expected_helpers:
                if not hasattr(utils, helper_name):
                    missing_helpers.append(helper_name)

            assert not missing_helpers, f"Missing helper functions: {missing_helpers}"

        except ImportError:
            pytest.fail("Cannot import utils.py for helper function testing")

    def test_mock_factories_defined(self):
        """Test that mock factory classes are defined"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            missing_factories = []
            for factory_name in self.expected_mock_factories:
                if not hasattr(utils, factory_name):
                    missing_factories.append(factory_name)

            assert not missing_factories, f"Missing mock factories: {missing_factories}"

        except ImportError:
            pytest.fail("Cannot import utils.py for mock factory testing")

    def test_database_utilities_defined(self):
        """Test that database utility functions are defined"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            missing_db_utils = []
            for db_util_name in self.expected_db_utilities:
                if not hasattr(utils, db_util_name):
                    missing_db_utils.append(db_util_name)

            assert not missing_db_utils, f"Missing database utilities: {missing_db_utils}"

        except ImportError:
            pytest.fail("Cannot import utils.py for database utility testing")

    def test_assertion_functions_are_callable(self):
        """Test that assertion functions are callable"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            non_callable_assertions = []
            for assertion_name in self.expected_assertions:
                if hasattr(utils, assertion_name):
                    assertion_func = getattr(utils, assertion_name)
                    if not callable(assertion_func):
                        non_callable_assertions.append(assertion_name)

            assert not non_callable_assertions, f"Non-callable assertions: {non_callable_assertions}"

        except ImportError:
            pytest.fail("Cannot import utils.py for callable testing")

    def test_mock_factories_are_classes(self):
        """Test that mock factories are proper classes"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            non_class_factories = []
            for factory_name in self.expected_mock_factories:
                if hasattr(utils, factory_name):
                    factory_obj = getattr(utils, factory_name)
                    if not inspect.isclass(factory_obj):
                        non_class_factories.append(factory_name)

            assert not non_class_factories, f"Non-class factories: {non_class_factories}"

        except ImportError:
            pytest.fail("Cannot import utils.py for class testing")

    def test_functions_have_docstrings(self):
        """Test that utility functions have proper docstrings"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            functions_without_docs = []
            all_expected_functions = (
                self.expected_assertions +
                self.expected_helpers +
                self.expected_db_utilities
            )

            for func_name in all_expected_functions:
                if hasattr(utils, func_name):
                    func_obj = getattr(utils, func_name)
                    if callable(func_obj):
                        if not func_obj.__doc__ or len(func_obj.__doc__.strip()) < 10:
                            functions_without_docs.append(func_name)

            assert not functions_without_docs, f"Functions without proper docstrings: {functions_without_docs}"

        except ImportError:
            pytest.fail("Cannot import utils.py for docstring testing")

    def test_assertion_functions_handle_edge_cases(self):
        """Test that assertion functions properly handle edge cases"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            # Test factor values assertion
            if hasattr(utils, 'assert_factor_values_equal'):
                func = utils.assert_factor_values_equal

                # Check function signature
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                expected_params = ['actual', 'expected']
                missing_params = [p for p in expected_params if p not in params]
                assert not missing_params, f"assert_factor_values_equal missing parameters: {missing_params}"

        except ImportError:
            pytest.fail("Cannot import utils.py for edge case testing")

    def test_helper_functions_support_hikyuu_integration(self):
        """Test that helper functions support Hikyuu framework integration"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            # Check for Hikyuu-specific helpers
            hikyuu_helpers = ["create_test_stock_data", "MockHikyuuKDataFactory"]
            missing_hikyuu_helpers = []

            for helper_name in hikyuu_helpers:
                if not hasattr(utils, helper_name):
                    missing_hikyuu_helpers.append(helper_name)

            assert not missing_hikyuu_helpers, f"Missing Hikyuu integration helpers: {missing_hikyuu_helpers}"

        except ImportError:
            pytest.fail("Cannot import utils.py for Hikyuu integration testing")

    def test_database_utilities_support_mysql(self):
        """Test that database utilities support MySQL operations"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            # Check for MySQL-specific functionality
            if hasattr(utils, 'create_test_tables'):
                func = utils.create_test_tables

                # Should accept database engine parameter
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                assert 'engine' in params or 'session' in params, \
                    "create_test_tables should accept database engine or session parameter"

        except ImportError:
            pytest.fail("Cannot import utils.py for MySQL support testing")

    def test_utils_supports_performance_testing(self):
        """Test that utilities support performance testing requirements"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            # Check for performance testing utilities
            performance_utils = ["assert_performance_within_bounds", "wait_for_calculation"]
            missing_performance_utils = []

            for util_name in performance_utils:
                if not hasattr(utils, util_name):
                    missing_performance_utils.append(util_name)

            assert not missing_performance_utils, f"Missing performance testing utilities: {missing_performance_utils}"

        except ImportError:
            pytest.fail("Cannot import utils.py for performance testing validation")

    def test_no_side_effects_on_import(self):
        """Test that importing utils.py has no side effects"""
        try:
            # Import should not raise exceptions or have side effects
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            # Should be able to import without errors
            assert utils is not None

        except Exception as e:
            pytest.fail(f"Importing utils.py caused side effects or errors: {e}")

    def test_utils_integrates_with_conftest_fixtures(self):
        """Test that utils.py integrates properly with conftest.py fixtures"""
        try:
            tests_dir = str(self.project_root / "tests")
            if tests_dir not in sys.path:
                sys.path.insert(0, tests_dir)

            import utils

            # Check for functions that should work with fixtures
            fixture_compatible_functions = ["setup_test_database_tables", "cleanup_test_database"]

            for func_name in fixture_compatible_functions:
                if hasattr(utils, func_name):
                    func = getattr(utils, func_name)
                    sig = inspect.signature(func)

                    # Should accept session or engine parameters for database operations
                    params = list(sig.parameters.keys())
                    has_db_param = any(param in ['session', 'engine', 'db'] for param in params)

                    assert has_db_param, f"{func_name} should accept database session/engine parameter"

        except ImportError:
            pytest.fail("Cannot import utils.py for fixture integration testing")