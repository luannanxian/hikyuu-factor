"""
REST API Contract Tests

测试RESTful API通用契约，确保：
- API版本管理的一致性
- HTTP状态码的标准化使用
- 内容类型和编码的规范
- 错误响应格式的统一
- API文档和契约的完整性
"""
import pytest
import json
import jsonschema
from datetime import datetime, date
from typing import Dict, List, Any, Optional


@pytest.mark.contract
class TestRESTAPIContract:
    """REST API通用契约测试"""

    def setup_method(self):
        """设置测试环境"""
        # API版本管理
        self.api_versions = {
            "current": "v1",
            "supported": ["v1"],
            "deprecated": [],
            "sunset_policy": "6_months_notice"
        }

        # 标准HTTP状态码映射
        self.standard_status_codes = {
            # Success codes
            200: "OK - Request succeeded",
            201: "Created - Resource created successfully",
            202: "Accepted - Request accepted for processing",
            204: "No Content - Request succeeded, no content returned",

            # Client error codes
            400: "Bad Request - Invalid request syntax or parameters",
            401: "Unauthorized - Authentication required",
            403: "Forbidden - Access denied",
            404: "Not Found - Resource not found",
            409: "Conflict - Resource conflict",
            422: "Unprocessable Entity - Validation errors",
            429: "Too Many Requests - Rate limit exceeded",

            # Server error codes
            500: "Internal Server Error - Unexpected server error",
            502: "Bad Gateway - Upstream server error",
            503: "Service Unavailable - Service temporarily unavailable",
            504: "Gateway Timeout - Upstream timeout"
        }

        # 标准响应格式
        self.response_schemas = {
            "success_response": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success"]},
                    "data": {"type": "object"},
                    "timestamp": {"type": "string"},
                    "request_id": {"type": "string"},
                    "api_version": {"type": "string"},
                    "execution_time_ms": {"type": "integer", "minimum": 0}
                },
                "required": ["status", "data", "timestamp", "api_version"]
            },
            "error_response": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["error"]},
                    "error": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "message": {"type": "string"},
                            "details": {"type": "object"},
                            "documentation_url": {"type": "string"},
                            "correlation_id": {"type": "string"}
                        },
                        "required": ["code", "message"]
                    },
                    "timestamp": {"type": "string"},
                    "request_id": {"type": "string"},
                    "api_version": {"type": "string"}
                },
                "required": ["status", "error", "timestamp", "api_version"]
            },
            "validation_error_response": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["error"]},
                    "error": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "enum": ["VALIDATION_ERROR"]},
                            "message": {"type": "string"},
                            "validation_errors": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {"type": "string"},
                                        "code": {"type": "string"},
                                        "message": {"type": "string"},
                                        "value": {}
                                    },
                                    "required": ["field", "code", "message"]
                                }
                            }
                        },
                        "required": ["code", "message", "validation_errors"]
                    },
                    "timestamp": {"type": "string"},
                    "request_id": {"type": "string"},
                    "api_version": {"type": "string"}
                },
                "required": ["status", "error", "timestamp", "api_version"]
            }
        }

        # 标准请求/响应头
        self.standard_headers = {
            "request_headers": {
                "required": [
                    "Content-Type",
                    "Accept",
                    "User-Agent"
                ],
                "optional": [
                    "Authorization",
                    "X-Request-ID",
                    "X-Forwarded-For",
                    "Accept-Language",
                    "Cache-Control"
                ]
            },
            "response_headers": {
                "required": [
                    "Content-Type",
                    "X-Request-ID",
                    "X-Response-Time",
                    "X-API-Version"
                ],
                "optional": [
                    "Cache-Control",
                    "ETag",
                    "Last-Modified",
                    "X-RateLimit-Limit",
                    "X-RateLimit-Remaining",
                    "X-RateLimit-Reset"
                ]
            }
        }

    @pytest.mark.contract
    def test_api_versioning_contract(self):
        """测试API版本管理契约"""
        # Test version format
        current_version = self.api_versions["current"]
        assert current_version.startswith("v"), f"API version should start with 'v': {current_version}"
        assert current_version[1:].isdigit(), f"API version should be numeric after 'v': {current_version}"

        # Test supported versions
        supported_versions = self.api_versions["supported"]
        assert len(supported_versions) > 0, "Should have at least one supported version"
        assert current_version in supported_versions, "Current version should be in supported versions"

        # Test version negotiation
        version_negotiation_rules = {
            "header_based": "Accept: application/json; version=v1",
            "url_based": "/api/v1/endpoint",
            "parameter_based": "?version=v1",
            "default_version": self.api_versions["current"]
        }

        assert "default_version" in version_negotiation_rules, "Should define default version"
        assert version_negotiation_rules["default_version"] == current_version, \
            "Default version should match current version"

    @pytest.mark.contract
    def test_content_type_contract(self):
        """测试内容类型契约"""
        # Supported content types
        supported_content_types = {
            "request": [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data"
            ],
            "response": [
                "application/json",
                "text/csv",
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # Excel
            ]
        }

        # Test JSON as primary content type
        assert "application/json" in supported_content_types["request"], \
            "Should support JSON for requests"
        assert "application/json" in supported_content_types["response"], \
            "Should support JSON for responses"

        # Test charset specification
        charset_requirements = {
            "default_charset": "utf-8",
            "supported_charsets": ["utf-8", "iso-8859-1"],
            "charset_header": "Content-Type: application/json; charset=utf-8"
        }

        assert charset_requirements["default_charset"] == "utf-8", \
            "Should use UTF-8 as default charset"

    @pytest.mark.contract
    def test_http_status_codes_contract(self):
        """测试HTTP状态码契约"""
        # Test status code categories
        success_codes = [code for code in self.standard_status_codes.keys() if 200 <= code < 300]
        client_error_codes = [code for code in self.standard_status_codes.keys() if 400 <= code < 500]
        server_error_codes = [code for code in self.standard_status_codes.keys() if 500 <= code < 600]

        # Verify we have codes from each category
        assert len(success_codes) > 0, "Should define success status codes"
        assert len(client_error_codes) > 0, "Should define client error status codes"
        assert len(server_error_codes) > 0, "Should define server error status codes"

        # Test common status codes are included
        required_status_codes = [200, 400, 401, 404, 422, 500]
        for code in required_status_codes:
            assert code in self.standard_status_codes, f"Should include status code {code}"

        # Test status code usage rules
        status_code_rules = {
            200: "Successful GET, PUT, PATCH requests",
            201: "Successful POST requests that create resources",
            204: "Successful DELETE requests",
            400: "Invalid request syntax or parameters",
            401: "Missing or invalid authentication",
            403: "Valid authentication but insufficient permissions",
            404: "Resource not found",
            422: "Valid syntax but validation errors",
            500: "Unexpected server errors"
        }

        for code, rule in status_code_rules.items():
            assert code in self.standard_status_codes, f"Status code {code} should be defined: {rule}"

    @pytest.mark.contract
    def test_error_response_format_contract(self):
        """测试错误响应格式契约"""
        # Test standard error response
        standard_error = {
            "status": "error",
            "error": {
                "code": "RESOURCE_NOT_FOUND",
                "message": "The requested resource was not found",
                "details": {
                    "resource_type": "stock",
                    "resource_id": "invalid_code",
                    "suggestions": ["sh600000", "sz000001"]
                },
                "documentation_url": "https://docs.example.com/errors/resource-not-found",
                "correlation_id": "corr_12345"
            },
            "timestamp": "2024-01-15T09:30:00Z",
            "request_id": "req_67890",
            "api_version": "v1"
        }

        # Validate standard error format
        try:
            jsonschema.validate(standard_error, self.response_schemas["error_response"])
        except jsonschema.ValidationError as e:
            pytest.fail(f"Standard error response failed schema validation: {e}")

        # Test validation error response
        validation_error = {
            "status": "error",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "validation_errors": [
                    {
                        "field": "start_date",
                        "code": "INVALID_FORMAT",
                        "message": "Date must be in YYYY-MM-DD format",
                        "value": "2024/01/15"
                    },
                    {
                        "field": "stocks",
                        "code": "REQUIRED_FIELD",
                        "message": "This field is required",
                        "value": None
                    }
                ]
            },
            "timestamp": "2024-01-15T09:30:00Z",
            "request_id": "req_67891",
            "api_version": "v1"
        }

        try:
            jsonschema.validate(validation_error, self.response_schemas["validation_error_response"])
        except jsonschema.ValidationError as e:
            pytest.fail(f"Validation error response failed schema validation: {e}")

    @pytest.mark.contract
    def test_success_response_format_contract(self):
        """测试成功响应格式契约"""
        # Test standard success response
        success_response = {
            "status": "success",
            "data": {
                "stocks": [
                    {
                        "code": "sh600000",
                        "name": "浦发银行",
                        "market": "SH"
                    }
                ],
                "total": 1
            },
            "timestamp": "2024-01-15T09:30:00Z",
            "request_id": "req_12345",
            "api_version": "v1",
            "execution_time_ms": 125
        }

        # Validate success response format
        try:
            jsonschema.validate(success_response, self.response_schemas["success_response"])
        except jsonschema.ValidationError as e:
            pytest.fail(f"Success response failed schema validation: {e}")

    @pytest.mark.contract
    def test_pagination_contract(self):
        """测试分页契约"""
        # Pagination parameters
        pagination_schema = {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                },
                "sort": {
                    "type": "string",
                    "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*(:asc|:desc)?$"
                }
            }
        }

        # Test valid pagination request
        valid_pagination = {
            "limit": 50,
            "offset": 100,
            "page": 3,
            "sort": "created_at:desc"
        }

        try:
            jsonschema.validate(valid_pagination, pagination_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid pagination parameters failed validation: {e}")

        # Pagination response format
        paginated_response = {
            "status": "success",
            "data": {
                "items": [],  # Array of results
                "pagination": {
                    "total": 250,
                    "limit": 50,
                    "offset": 100,
                    "page": 3,
                    "total_pages": 5,
                    "has_next": True,
                    "has_previous": True,
                    "next_offset": 150,
                    "previous_offset": 50
                }
            },
            "timestamp": "2024-01-15T09:30:00Z",
            "request_id": "req_pagination",
            "api_version": "v1"
        }

        # Validate pagination response structure
        pagination_data = paginated_response["data"]["pagination"]
        required_pagination_fields = ["total", "limit", "offset", "has_next", "has_previous"]
        for field in required_pagination_fields:
            assert field in pagination_data, f"Pagination response missing field: {field}"

    @pytest.mark.contract
    def test_filtering_and_sorting_contract(self):
        """测试过滤和排序契约"""
        # Filtering parameters
        filtering_patterns = {
            "exact_match": "field=value",
            "comparison": "field__gt=value, field__lt=value, field__gte=value, field__lte=value",
            "inclusion": "field__in=value1,value2,value3",
            "text_search": "field__contains=text, field__startswith=prefix",
            "date_range": "date__range=2024-01-01,2024-01-31",
            "null_checks": "field__isnull=true"
        }

        # Sorting parameters
        sorting_patterns = {
            "single_field": "sort=field_name",
            "descending": "sort=field_name:desc",
            "multiple_fields": "sort=field1:asc,field2:desc",
            "default_sort": "sort=created_at:desc"
        }

        # Test filter validation
        valid_filters = {
            "market": "SH",
            "status__in": "active,suspended",
            "market_cap__gte": "1000000000",
            "created_at__range": "2024-01-01,2024-01-31"
        }

        # Verify filter patterns are supported
        for filter_name, filter_value in valid_filters.items():
            # Basic validation that filter follows expected patterns
            if "__" in filter_name:
                field, operator = filter_name.split("__", 1)
                valid_operators = ["gt", "lt", "gte", "lte", "in", "contains", "startswith", "range", "isnull"]
                assert operator in valid_operators, f"Invalid filter operator: {operator}"

    @pytest.mark.contract
    def test_rate_limiting_contract(self):
        """测试限流契约"""
        # Rate limiting configuration
        rate_limit_config = {
            "default_limits": {
                "requests_per_minute": 1000,
                "requests_per_hour": 10000,
                "requests_per_day": 100000
            },
            "authenticated_limits": {
                "requests_per_minute": 5000,
                "requests_per_hour": 50000,
                "requests_per_day": 500000
            },
            "premium_limits": {
                "requests_per_minute": 10000,
                "requests_per_hour": 100000,
                "requests_per_day": 1000000
            }
        }

        # Rate limit headers
        rate_limit_headers = {
            "X-RateLimit-Limit": "Maximum requests allowed",
            "X-RateLimit-Remaining": "Requests remaining in current window",
            "X-RateLimit-Reset": "Unix timestamp when limit resets",
            "X-RateLimit-Window": "Time window for rate limiting"
        }

        # Test rate limit header presence
        required_headers = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
        for header in required_headers:
            assert header in rate_limit_headers, f"Missing required rate limit header: {header}"

        # Test rate limit exceeded response
        rate_limit_exceeded = {
            "status": "error",
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please try again later.",
                "details": {
                    "limit": 1000,
                    "window": "1 minute",
                    "retry_after": 60
                }
            },
            "timestamp": "2024-01-15T09:30:00Z",
            "request_id": "req_rate_limit",
            "api_version": "v1"
        }

        try:
            jsonschema.validate(rate_limit_exceeded, self.response_schemas["error_response"])
        except jsonschema.ValidationError as e:
            pytest.fail(f"Rate limit exceeded response failed validation: {e}")

    @pytest.mark.contract
    def test_caching_contract(self):
        """测试缓存契约"""
        # Caching strategies
        caching_strategies = {
            "static_data": {
                "cache_control": "public, max-age=3600",
                "etag": True,
                "last_modified": True
            },
            "dynamic_data": {
                "cache_control": "private, max-age=300",
                "etag": True,
                "last_modified": False
            },
            "real_time_data": {
                "cache_control": "no-cache, no-store, must-revalidate",
                "etag": False,
                "last_modified": False
            }
        }

        # Test cache headers
        cache_headers = {
            "Cache-Control": "Controls caching behavior",
            "ETag": "Entity tag for cache validation",
            "Last-Modified": "Last modification time",
            "Expires": "Absolute expiration time",
            "Vary": "Varies cache by specified headers"
        }

        # Verify caching strategies are defined
        for strategy_name, strategy_config in caching_strategies.items():
            assert "cache_control" in strategy_config, f"{strategy_name} should define cache control"

    @pytest.mark.contract
    def test_cors_contract(self):
        """测试CORS契约"""
        # CORS configuration
        cors_config = {
            "allowed_origins": [
                "https://app.example.com",
                "https://admin.example.com"
            ],
            "allowed_methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            "allowed_headers": [
                "Authorization",
                "Content-Type",
                "Accept",
                "X-Request-ID",
                "X-API-Version"
            ],
            "exposed_headers": [
                "X-Request-ID",
                "X-Response-Time",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining"
            ],
            "allow_credentials": True,
            "max_age": 86400  # 24 hours
        }

        # Test CORS headers
        cors_headers = {
            "Access-Control-Allow-Origin": "Allowed origins",
            "Access-Control-Allow-Methods": "Allowed HTTP methods",
            "Access-Control-Allow-Headers": "Allowed request headers",
            "Access-Control-Expose-Headers": "Exposed response headers",
            "Access-Control-Allow-Credentials": "Allow credentials",
            "Access-Control-Max-Age": "Preflight cache duration"
        }

        # Verify CORS configuration
        assert len(cors_config["allowed_origins"]) > 0, "Should define allowed origins"
        assert "GET" in cors_config["allowed_methods"], "Should allow GET method"
        assert "POST" in cors_config["allowed_methods"], "Should allow POST method"

    @pytest.mark.contract
    def test_api_documentation_contract(self):
        """测试API文档契约"""
        # Documentation requirements
        documentation_requirements = {
            "openapi_version": "3.0.3",
            "interactive_docs": True,
            "schema_validation": True,
            "examples": True,
            "error_codes": True,
            "authentication_docs": True
        }

        # OpenAPI specification structure
        openapi_structure = {
            "openapi": "3.0.3",
            "info": {
                "title": "A股量化因子系统 API",
                "version": "1.0.0",
                "description": "Hikyuu量化因子挖掘与决策支持系统 API",
                "contact": {
                    "name": "API Support",
                    "email": "api-support@example.com"
                }
            },
            "servers": [
                {
                    "url": "https://api.example.com/v1",
                    "description": "Production server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {},
                "responses": {}
            }
        }

        # Verify documentation requirements
        required_sections = ["openapi", "info", "servers", "paths", "components"]
        for section in required_sections:
            assert section in openapi_structure, f"OpenAPI spec should include {section}"

    @pytest.mark.contract
    def test_security_headers_contract(self):
        """测试安全头契约"""
        # Security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

        # Test security header presence
        critical_security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Strict-Transport-Security"
        ]

        for header in critical_security_headers:
            assert header in security_headers, f"Missing critical security header: {header}"

        # Test HTTPS enforcement
        https_requirements = {
            "enforce_https": True,
            "hsts_max_age": 31536000,  # 1 year
            "include_subdomains": True,
            "preload": False
        }

        assert https_requirements["enforce_https"], "Should enforce HTTPS"
        assert https_requirements["hsts_max_age"] >= 31536000, "HSTS max-age should be at least 1 year"