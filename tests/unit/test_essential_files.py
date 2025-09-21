"""
Test for T005: Create basic .gitignore, .env.example, and README.md

This test validates that essential project files are created with proper content:
- .gitignore with Python and project-specific ignores
- .env.example with configuration templates
- README.md with project documentation
"""
import re
from pathlib import Path
import pytest


class TestProjectEssentialFiles:
    """Test suite for essential project files validation (T005)"""

    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path.cwd()

        # Expected files for T005
        self.expected_files = {
            ".gitignore": self.project_root / ".gitignore",
            ".env.example": self.project_root / ".env.example",
            "README.md": self.project_root / "README.md"
        }

        # Required .gitignore patterns
        self.required_gitignore_patterns = [
            # Python specific
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.egg-info",
            ".pytest_cache",
            ".coverage",
            "coverage.xml",
            ".mypy_cache",

            # Environment and secrets
            ".env",
            "*.env",
            ".env.local",

            # IDE and editors
            ".vscode",
            ".idea",
            "*.swp",
            "*.swo",

            # Build and distribution
            "build/",
            "dist/",
            "*.egg-info/",

            # Project specific
            "data/cache",
            "logs/",
        ]

        # Required .env.example sections
        self.required_env_sections = [
            "DATABASE_URL",
            "REDIS_URL",
            "HIKYUU_DATA_DIR",
            "LOG_LEVEL",
            "SECRET_KEY"
        ]

        # Required README.md sections
        self.required_readme_sections = [
            "# hikyuu-factor",
            "## 项目简介",
            "## 安装",
            "## 使用方法",
            "## 开发",
            "## 许可证"
        ]

    def test_all_essential_files_exist(self):
        """Test that all essential files exist"""
        missing_files = []

        for file_name, file_path in self.expected_files.items():
            if not file_path.exists():
                missing_files.append(file_name)
            elif not file_path.is_file():
                missing_files.append(f"{file_name} (exists but not a file)")

        assert not missing_files, f"Missing essential files: {missing_files}"

    def test_gitignore_contains_required_patterns(self):
        """Test that .gitignore contains all required ignore patterns"""
        gitignore_path = self.expected_files[".gitignore"]

        if not gitignore_path.exists():
            pytest.fail(".gitignore file does not exist")

        with open(gitignore_path, 'r', encoding='utf-8') as f:
            gitignore_content = f.read()

        missing_patterns = []
        for pattern in self.required_gitignore_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)

        assert not missing_patterns, f"Missing .gitignore patterns: {missing_patterns}"

    def test_gitignore_follows_best_practices(self):
        """Test that .gitignore follows Python best practices"""
        gitignore_path = self.expected_files[".gitignore"]

        with open(gitignore_path, 'r', encoding='utf-8') as f:
            gitignore_content = f.read()

        # Should have comments explaining sections
        assert "#" in gitignore_content, ".gitignore should have comments"

        # Should not ignore important files
        problematic_patterns = [".py", "src/", "requirements.txt", "pyproject.toml"]
        ignored_important = []

        for pattern in problematic_patterns:
            # Check if pattern appears as a standalone line (would ignore everything)
            if f"\n{pattern}\n" in gitignore_content or gitignore_content.startswith(f"{pattern}\n"):
                ignored_important.append(pattern)

        assert not ignored_important, f"Important files/dirs being ignored: {ignored_important}"

    def test_env_example_contains_required_variables(self):
        """Test that .env.example contains all required environment variables"""
        env_example_path = self.expected_files[".env.example"]

        if not env_example_path.exists():
            pytest.fail(".env.example file does not exist")

        with open(env_example_path, 'r', encoding='utf-8') as f:
            env_content = f.read()

        missing_vars = []
        for var in self.required_env_sections:
            if var not in env_content:
                missing_vars.append(var)

        assert not missing_vars, f"Missing environment variables: {missing_vars}"

    def test_env_example_has_safe_default_values(self):
        """Test that .env.example has safe default values (no secrets)"""
        env_example_path = self.expected_files[".env.example"]

        with open(env_example_path, 'r', encoding='utf-8') as f:
            env_content = f.read()

        # Check for potentially unsafe patterns
        unsafe_patterns = [
            r"password\s*=\s*[^<\n\r]+",  # Real passwords
            r"secret\s*=\s*[a-zA-Z0-9]{20,}",  # Real secrets
            r"token\s*=\s*[a-zA-Z0-9]{20,}",  # Real tokens
        ]

        security_issues = []
        for pattern in unsafe_patterns:
            matches = re.findall(pattern, env_content, re.IGNORECASE)
            if matches:
                security_issues.extend(matches)

        assert not security_issues, f"Potential secrets in .env.example: {security_issues}"

        # Should have placeholder values
        placeholder_patterns = ["<", "example", "your_", "changeme"]
        has_placeholders = any(pattern in env_content.lower() for pattern in placeholder_patterns)

        assert has_placeholders, ".env.example should use placeholder values"

    def test_readme_has_required_sections(self):
        """Test that README.md has all required sections"""
        readme_path = self.expected_files["README.md"]

        if not readme_path.exists():
            pytest.fail("README.md file does not exist")

        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()

        missing_sections = []
        for section in self.required_readme_sections:
            if section not in readme_content:
                missing_sections.append(section)

        assert not missing_sections, f"Missing README.md sections: {missing_sections}"

    def test_readme_mentions_hikyuu_framework(self):
        """Test that README.md properly mentions Hikyuu framework"""
        readme_path = self.expected_files["README.md"]

        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()

        hikyuu_mentions = ["hikyuu", "Hikyuu", "HIKYUU"]
        has_hikyuu = any(mention in readme_content for mention in hikyuu_mentions)

        assert has_hikyuu, "README.md should mention Hikyuu framework"

        # Should mention quantitative factors
        factor_keywords = ["量化因子", "quantitative factor", "因子挖掘", "factor mining"]
        has_factors = any(keyword in readme_content for keyword in factor_keywords)

        assert has_factors, "README.md should mention quantitative factors"

    def test_readme_has_installation_instructions(self):
        """Test that README.md has clear installation instructions"""
        readme_path = self.expected_files["README.md"]

        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()

        # Should mention pip installation
        pip_patterns = ["pip install", "pip3 install", "python -m pip"]
        has_pip = any(pattern in readme_content for pattern in pip_patterns)

        # Should mention requirements.txt or pyproject.toml
        dependency_patterns = ["requirements.txt", "pyproject.toml", "-e ."]
        has_dependencies = any(pattern in readme_content for pattern in dependency_patterns)

        assert has_pip or has_dependencies, "README.md should have installation instructions"

    def test_readme_has_usage_examples(self):
        """Test that README.md has usage examples"""
        readme_path = self.expected_files["README.md"]

        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()

        # Should have code blocks or command examples
        code_indicators = ["```", "`", "python", "hikyuu-factor"]
        has_code_examples = any(indicator in readme_content for indicator in code_indicators)

        assert has_code_examples, "README.md should have usage examples"

    def test_project_files_are_utf8_encoded(self):
        """Test that all project files use UTF-8 encoding"""
        encoding_errors = []

        for file_name, file_path in self.expected_files.items():
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read()
                except UnicodeDecodeError as e:
                    encoding_errors.append(f"{file_name}: {str(e)}")

        assert not encoding_errors, f"Encoding errors found: {encoding_errors}"

    def test_files_have_proper_line_endings(self):
        """Test that files use consistent line endings"""
        line_ending_issues = []

        for file_name, file_path in self.expected_files.items():
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    content = f.read()

                # Check for mixed line endings
                has_crlf = b'\r\n' in content
                has_lf = b'\n' in content and b'\r\n' not in content
                has_cr = b'\r' in content and b'\r\n' not in content

                endings_count = sum([has_crlf, has_lf, has_cr])
                if endings_count > 1:
                    line_ending_issues.append(f"{file_name}: mixed line endings")

        assert not line_ending_issues, f"Line ending issues: {line_ending_issues}"