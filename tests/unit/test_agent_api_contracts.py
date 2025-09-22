"""
Test for T011: Create Agent API contract tests

This test validates Agent API contracts for the hikyuu-factor system:
- DataManager Agent API contract
- FactorCalculation Agent API contract
- Validation Agent API contract
- SignalGeneration Agent API contract
- Agent communication contracts
- REST API endpoint contracts
"""
import inspect
import json
from pathlib import Path
import pytest
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime


class TestAgentAPIContracts:
    """Test suite for Agent API contract validation (T011)"""

    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path.cwd()
        self.contracts_dir = self.project_root / "tests" / "contract"

        # Expected Agent contract test files
        self.expected_contract_files = [
            "test_data_manager_agent_contract.py",
            "test_factor_calculation_agent_contract.py",
            "test_validation_agent_contract.py",
            "test_signal_generation_agent_contract.py",
            "test_agent_communication_contract.py",
            "test_rest_api_contract.py"
        ]

        # Expected Agent API endpoints
        self.expected_agent_endpoints = {
            "data_manager": [
                "/api/v1/data/stocks",
                "/api/v1/data/prices",
                "/api/v1/data/calendar",
                "/api/v1/data/status"
            ],
            "factor_calculation": [
                "/api/v1/factors/calculate",
                "/api/v1/factors/status",
                "/api/v1/factors/results",
                "/api/v1/factors/list"
            ],
            "validation": [
                "/api/v1/validation/run",
                "/api/v1/validation/status",
                "/api/v1/validation/results",
                "/api/v1/validation/reports"
            ],
            "signal_generation": [
                "/api/v1/signals/generate",
                "/api/v1/signals/status",
                "/api/v1/signals/list",
                "/api/v1/signals/confirm"
            ]
        }

        # Expected contract test patterns
        self.expected_contract_patterns = [
            "test_api_endpoint_exists",
            "test_request_schema_validation",
            "test_response_schema_validation",
            "test_error_handling_contract",
            "test_authentication_contract",
            "test_rate_limiting_contract"
        ]

    def test_contract_tests_directory_exists(self):
        """Test that tests/contract/ directory exists"""
        assert self.contracts_dir.exists(), "tests/contract/ directory should exist"
        assert self.contracts_dir.is_dir(), "tests/contract/ should be a directory"

    def test_contract_init_file_exists(self):
        """Test that tests/contract/__init__.py exists"""
        init_file = self.contracts_dir / "__init__.py"
        assert init_file.exists(), "tests/contract/__init__.py should exist"

    def test_all_agent_contract_files_exist(self):
        """Test that all Agent contract test files exist"""
        missing_files = []

        for contract_file in self.expected_contract_files:
            file_path = self.contracts_dir / contract_file
            if not file_path.exists():
                missing_files.append(contract_file)

        assert not missing_files, f"Missing Agent contract test files: {missing_files}"

    def test_contract_files_are_valid_python(self):
        """Test that all contract test files are valid Python syntax"""
        syntax_errors = []

        for contract_file in self.expected_contract_files:
            file_path = self.contracts_dir / contract_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        compile(content, str(file_path), 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{contract_file}: {e}")

        assert not syntax_errors, f"Syntax errors in contract files: {syntax_errors}"

    def test_data_manager_agent_contract_tests_defined(self):
        """Test that DataManager Agent contract tests are defined"""
        contract_file = self.contracts_dir / "test_data_manager_agent_contract.py"

        if not contract_file.exists():
            pytest.fail("test_data_manager_agent_contract.py does not exist")

        try:
            # Add contract directory to Python path
            contract_dir = str(self.contracts_dir)
            if contract_dir not in sys.path:
                sys.path.insert(0, contract_dir)

            import test_data_manager_agent_contract

            # Check for expected test class
            expected_class = "TestDataManagerAgentContract"
            assert hasattr(test_data_manager_agent_contract, expected_class), \
                f"Missing {expected_class} class"

            # Check for expected test methods
            test_class = getattr(test_data_manager_agent_contract, expected_class)
            methods = [method for method in dir(test_class) if method.startswith('test_')]

            expected_methods = [
                "test_get_stocks_endpoint",
                "test_get_prices_endpoint",
                "test_get_calendar_endpoint",
                "test_data_status_endpoint"
            ]

            missing_methods = [method for method in expected_methods if method not in methods]
            assert not missing_methods, f"Missing DataManager contract test methods: {missing_methods}"

        except ImportError as e:
            pytest.fail(f"Cannot import test_data_manager_agent_contract: {e}")

    def test_factor_calculation_agent_contract_tests_defined(self):
        """Test that FactorCalculation Agent contract tests are defined"""
        contract_file = self.contracts_dir / "test_factor_calculation_agent_contract.py"

        if not contract_file.exists():
            pytest.fail("test_factor_calculation_agent_contract.py does not exist")

        try:
            contract_dir = str(self.contracts_dir)
            if contract_dir not in sys.path:
                sys.path.insert(0, contract_dir)

            import test_factor_calculation_agent_contract

            expected_class = "TestFactorCalculationAgentContract"
            assert hasattr(test_factor_calculation_agent_contract, expected_class), \
                f"Missing {expected_class} class"

            test_class = getattr(test_factor_calculation_agent_contract, expected_class)
            methods = [method for method in dir(test_class) if method.startswith('test_')]

            expected_methods = [
                "test_calculate_factors_endpoint",
                "test_factor_status_endpoint",
                "test_factor_results_endpoint",
                "test_list_factors_endpoint"
            ]

            missing_methods = [method for method in expected_methods if method not in methods]
            assert not missing_methods, f"Missing FactorCalculation contract test methods: {missing_methods}"

        except ImportError as e:
            pytest.fail(f"Cannot import test_factor_calculation_agent_contract: {e}")

    def test_validation_agent_contract_tests_defined(self):
        """Test that Validation Agent contract tests are defined"""
        contract_file = self.contracts_dir / "test_validation_agent_contract.py"

        if not contract_file.exists():
            pytest.fail("test_validation_agent_contract.py does not exist")

        try:
            contract_dir = str(self.contracts_dir)
            if contract_dir not in sys.path:
                sys.path.insert(0, contract_dir)

            import test_validation_agent_contract

            expected_class = "TestValidationAgentContract"
            assert hasattr(test_validation_agent_contract, expected_class), \
                f"Missing {expected_class} class"

            test_class = getattr(test_validation_agent_contract, expected_class)
            methods = [method for method in dir(test_class) if method.startswith('test_')]

            expected_methods = [
                "test_run_validation_endpoint",
                "test_validation_status_endpoint",
                "test_validation_results_endpoint",
                "test_validation_reports_endpoint"
            ]

            missing_methods = [method for method in expected_methods if method not in methods]
            assert not missing_methods, f"Missing Validation contract test methods: {missing_methods}"

        except ImportError as e:
            pytest.fail(f"Cannot import test_validation_agent_contract: {e}")

    def test_signal_generation_agent_contract_tests_defined(self):
        """Test that SignalGeneration Agent contract tests are defined"""
        contract_file = self.contracts_dir / "test_signal_generation_agent_contract.py"

        if not contract_file.exists():
            pytest.fail("test_signal_generation_agent_contract.py does not exist")

        try:
            contract_dir = str(self.contracts_dir)
            if contract_dir not in sys.path:
                sys.path.insert(0, contract_dir)

            import test_signal_generation_agent_contract

            expected_class = "TestSignalGenerationAgentContract"
            assert hasattr(test_signal_generation_agent_contract, expected_class), \
                f"Missing {expected_class} class"

            test_class = getattr(test_signal_generation_agent_contract, expected_class)
            methods = [method for method in dir(test_class) if method.startswith('test_')]

            expected_methods = [
                "test_generate_signals_endpoint",
                "test_signal_status_endpoint",
                "test_list_signals_endpoint",
                "test_confirm_signals_endpoint"
            ]

            missing_methods = [method for method in expected_methods if method not in methods]
            assert not missing_methods, f"Missing SignalGeneration contract test methods: {missing_methods}"

        except ImportError as e:
            pytest.fail(f"Cannot import test_signal_generation_agent_contract: {e}")

    def test_agent_communication_contract_tests_defined(self):
        """Test that Agent communication contract tests are defined"""
        contract_file = self.contracts_dir / "test_agent_communication_contract.py"

        if not contract_file.exists():
            pytest.fail("test_agent_communication_contract.py does not exist")

        try:
            contract_dir = str(self.contracts_dir)
            if contract_dir not in sys.path:
                sys.path.insert(0, contract_dir)

            import test_agent_communication_contract

            expected_class = "TestAgentCommunicationContract"
            assert hasattr(test_agent_communication_contract, expected_class), \
                f"Missing {expected_class} class"

            test_class = getattr(test_agent_communication_contract, expected_class)
            methods = [method for method in dir(test_class) if method.startswith('test_')]

            expected_methods = [
                "test_agent_to_agent_communication",
                "test_message_format_contract",
                "test_authentication_between_agents",
                "test_error_propagation_contract"
            ]

            missing_methods = [method for method in expected_methods if method not in methods]
            assert not missing_methods, f"Missing Agent communication contract test methods: {missing_methods}"

        except ImportError as e:
            pytest.fail(f"Cannot import test_agent_communication_contract: {e}")

    def test_rest_api_contract_tests_defined(self):
        """Test that REST API contract tests are defined"""
        contract_file = self.contracts_dir / "test_rest_api_contract.py"

        if not contract_file.exists():
            pytest.fail("test_rest_api_contract.py does not exist")

        try:
            contract_dir = str(self.contracts_dir)
            if contract_dir not in sys.path:
                sys.path.insert(0, contract_dir)

            import test_rest_api_contract

            expected_class = "TestRESTAPIContract"
            assert hasattr(test_rest_api_contract, expected_class), \
                f"Missing {expected_class} class"

            test_class = getattr(test_rest_api_contract, expected_class)
            methods = [method for method in dir(test_class) if method.startswith('test_')]

            expected_methods = [
                "test_api_versioning_contract",
                "test_content_type_contract",
                "test_http_status_codes_contract",
                "test_error_response_format_contract"
            ]

            missing_methods = [method for method in expected_methods if method not in methods]
            assert not missing_methods, f"Missing REST API contract test methods: {missing_methods}"

        except ImportError as e:
            pytest.fail(f"Cannot import test_rest_api_contract: {e}")

    def test_contract_tests_use_proper_markers(self):
        """Test that contract tests use proper pytest markers"""
        try:
            contract_dir = str(self.contracts_dir)
            if contract_dir not in sys.path:
                sys.path.insert(0, contract_dir)

            # Check that contract tests are marked properly
            for contract_file in self.expected_contract_files:
                if (self.contracts_dir / contract_file).exists():
                    with open(self.contracts_dir / contract_file, 'r') as f:
                        content = f.read()

                        # Should have contract marker
                        assert "@pytest.mark.contract" in content, \
                            f"{contract_file} should use @pytest.mark.contract"

        except Exception as e:
            pytest.fail(f"Error checking contract test markers: {e}")

    def test_contract_tests_include_schema_validation(self):
        """Test that contract tests include schema validation"""
        expected_schema_patterns = [
            "request_schema",
            "response_schema",
            "validate_schema",
            "jsonschema"
        ]

        schema_validation_missing = []

        for contract_file in self.expected_contract_files:
            file_path = self.contracts_dir / contract_file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()

                    has_schema_validation = any(pattern in content for pattern in expected_schema_patterns)
                    if not has_schema_validation:
                        schema_validation_missing.append(contract_file)

        assert not schema_validation_missing, \
            f"Contract files missing schema validation: {schema_validation_missing}"

    def test_contract_tests_include_error_handling(self):
        """Test that contract tests include error handling validation"""
        expected_error_patterns = [
            "test_error",
            "error_handling",
            "status_code",
            "exception"
        ]

        error_handling_missing = []

        for contract_file in self.expected_contract_files:
            file_path = self.contracts_dir / contract_file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()

                    has_error_handling = any(pattern in content for pattern in expected_error_patterns)
                    if not has_error_handling:
                        error_handling_missing.append(contract_file)

        assert not error_handling_missing, \
            f"Contract files missing error handling tests: {error_handling_missing}"

    def test_contract_tests_support_agent_architecture(self):
        """Test that contract tests support Agent architecture requirements"""
        agent_architecture_patterns = [
            "agent",
            "microservice",
            "api",
            "endpoint"
        ]

        architecture_support_missing = []

        for contract_file in self.expected_contract_files:
            file_path = self.contracts_dir / contract_file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read().lower()

                    has_architecture_support = any(pattern in content for pattern in agent_architecture_patterns)
                    if not has_architecture_support:
                        architecture_support_missing.append(contract_file)

        assert not architecture_support_missing, \
            f"Contract files missing Agent architecture support: {architecture_support_missing}"

    def test_contract_tests_have_proper_docstrings(self):
        """Test that contract test classes and methods have proper docstrings"""
        try:
            contract_dir = str(self.contracts_dir)
            if contract_dir not in sys.path:
                sys.path.insert(0, contract_dir)

            files_without_docs = []

            for contract_file in self.expected_contract_files:
                if (self.contracts_dir / contract_file).exists():
                    module_name = contract_file[:-3]  # Remove .py extension
                    try:
                        module = __import__(module_name)

                        # Check module docstring
                        if not module.__doc__ or len(module.__doc__.strip()) < 20:
                            files_without_docs.append(f"{contract_file} (module)")

                        # Check test class docstrings
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if inspect.isclass(attr) and attr_name.startswith('Test'):
                                if not attr.__doc__ or len(attr.__doc__.strip()) < 10:
                                    files_without_docs.append(f"{contract_file} ({attr_name})")

                    except ImportError:
                        continue

            assert not files_without_docs, f"Contract files without proper docstrings: {files_without_docs}"

        except Exception as e:
            pytest.fail(f"Error checking contract test docstrings: {e}")

    def test_no_side_effects_on_contract_import(self):
        """Test that importing contract test modules has no side effects"""
        try:
            contract_dir = str(self.contracts_dir)
            if contract_dir not in sys.path:
                sys.path.insert(0, contract_dir)

            # Import all contract modules - should not cause side effects
            for contract_file in self.expected_contract_files:
                if (self.contracts_dir / contract_file).exists():
                    module_name = contract_file[:-3]  # Remove .py extension
                    try:
                        module = __import__(module_name)
                        assert module is not None
                    except ImportError:
                        # Module might not exist yet, that's ok for this test
                        continue

        except Exception as e:
            pytest.fail(f"Importing contract test modules caused side effects: {e}")