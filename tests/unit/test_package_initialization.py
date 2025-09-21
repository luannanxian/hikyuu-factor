"""
Test for T002: Initialize Python package with __init__.py files in all modules

This test validates that all Python packages are properly initialized
and can be imported correctly.
"""
import importlib
import sys
from pathlib import Path
import pytest


class TestPythonPackageInitialization:
    """Test suite for Python package initialization validation (T002)"""

    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path.cwd()

        # All Python packages that should be importable
        self.expected_packages = [
            "src",
            "src.agents",
            "src.models",
            "src.services",
            "src.api",
            "src.cli",
            "src.lib"
        ]

        # Expected attributes for main package
        self.src_expected_attributes = [
            "__version__",
            "__author__"
        ]

    def test_all_packages_are_importable(self):
        """Test that all expected packages can be imported"""
        import_failures = []

        for package_name in self.expected_packages:
            try:
                importlib.import_module(package_name)
            except ImportError as e:
                import_failures.append(f"{package_name}: {str(e)}")

        assert not import_failures, f"Failed to import packages: {import_failures}"

    def test_main_src_package_has_metadata(self):
        """Test that main src package has proper metadata"""
        import src

        missing_attrs = []
        for attr in self.src_expected_attributes:
            if not hasattr(src, attr):
                missing_attrs.append(attr)

        assert not missing_attrs, f"Missing attributes in src package: {missing_attrs}"

        # Validate attribute values
        assert isinstance(src.__version__, str), "__version__ should be a string"
        assert len(src.__version__) > 0, "__version__ should not be empty"
        assert isinstance(src.__author__, str), "__author__ should be a string"

    def test_subpackages_have_docstrings(self):
        """Test that all subpackages have meaningful docstrings"""
        packages_without_docstrings = []

        for package_name in self.expected_packages:
            if package_name == "src":
                continue  # Skip main package

            try:
                module = importlib.import_module(package_name)
                if not module.__doc__ or len(module.__doc__.strip()) == 0:
                    packages_without_docstrings.append(package_name)
            except ImportError:
                # This will be caught by test_all_packages_are_importable
                pass

        assert not packages_without_docstrings, f"Packages without docstrings: {packages_without_docstrings}"

    def test_package_structure_supports_agent_architecture(self):
        """Test that package structure properly supports 4-agent architecture"""
        try:
            import src.agents

            # Check that agents package is ready for the 4 agents
            expected_docstring_content = [
                "Agent",
                "DataManager",
                "FactorCalculation",
                "Validation",
                "SignalGeneration"
            ]

            missing_content = []
            agents_doc = src.agents.__doc__ or ""

            for content in expected_docstring_content:
                if content not in agents_doc:
                    missing_content.append(content)

            assert not missing_content, f"Agent package docstring missing: {missing_content}"

        except ImportError:
            pytest.fail("src.agents package should be importable")

    def test_import_from_project_root_works(self):
        """Test that imports work correctly from project root directory"""
        # This simulates running from project root
        current_path = str(self.project_root)
        if current_path not in sys.path:
            sys.path.insert(0, current_path)

        try:
            # Test direct import
            import src

            # Test subpackage imports
            from src import agents, models, services, api, cli, lib

            # Test that we can access attributes
            assert hasattr(src, '__version__')
            assert hasattr(agents, '__doc__')

        except ImportError as e:
            pytest.fail(f"Import from project root failed: {e}")

    def test_no_circular_imports(self):
        """Test that there are no circular import issues"""
        import_order = [
            "src",
            "src.lib",      # Should have no dependencies
            "src.models",   # May depend on lib
            "src.services", # May depend on lib, models
            "src.agents",   # May depend on lib, models, services
            "src.api",      # May depend on all above
            "src.cli"       # May depend on all above
        ]

        imported_modules = []

        for module_name in import_order:
            try:
                module = importlib.import_module(module_name)
                imported_modules.append(module_name)

                # Force reload to catch any circular import issues
                importlib.reload(module)

            except ImportError as e:
                pytest.fail(f"Circular import or dependency issue in {module_name}: {e}")

    def test_package_init_files_exist_and_valid(self):
        """Test that all __init__.py files exist and are valid Python"""
        init_files = [
            "src/__init__.py",
            "src/agents/__init__.py",
            "src/models/__init__.py",
            "src/services/__init__.py",
            "src/api/__init__.py",
            "src/cli/__init__.py",
            "src/lib/__init__.py"
        ]

        invalid_files = []

        for init_file in init_files:
            full_path = self.project_root / init_file

            if not full_path.exists():
                invalid_files.append(f"{init_file} - does not exist")
                continue

            try:
                # Try to compile the file to check for syntax errors
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    compile(content, init_file, 'exec')
            except SyntaxError as e:
                invalid_files.append(f"{init_file} - syntax error: {e}")
            except Exception as e:
                invalid_files.append(f"{init_file} - error: {e}")

        assert not invalid_files, f"Invalid __init__.py files: {invalid_files}"