"""Pytest configuration for deep_field_metadetect tests.

This configuration handles JAX float32/float64 mode validation for ALL tests
in the package. Single file applies to all test subdirectories via pytest's
conftest.py discovery mechanism.
"""

import jax
import pytest


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--float32-only",
        action="store_true",
        default=False,
        help="Run only tests marked for pure float32 mode (jax_enable_x64=False).",
    )


def pytest_configure(config):
    """Validate JAX settings match test requirements."""
    # Validate JAX configuration matches test requirements
    # Do NOT set config here - it must be set via environment variable
    expected_x64 = not config.getoption("--float32-only")
    actual_x64 = jax.config.jax_enable_x64

    if actual_x64 != expected_x64:
        mode = "float32-only" if config.getoption("--float32-only") else "regular"
        expected_env = "false" if config.getoption("--float32-only") else "true"
        raise RuntimeError(
            f"JAX x64 configuration mismatch for {mode} tests!\n"
            f"  Expected jax_enable_x64={expected_x64}\n"
            f"  Got jax_enable_x64={actual_x64}\n"
            f"  Example: JAX_ENABLE_X64={expected_env} pytest "
            f"{'--float32-only ' if config.getoption('--float32-only') else ''}..."
        )


def pytest_runtest_setup(item):
    """Skip logic based on marker and command-line flag."""
    has_float32_marker = "float32_only" in item.keywords
    using_float32_flag = item.config.getoption("--float32-only")

    if using_float32_flag and not has_float32_marker:
        pytest.skip("Skipped: test not marked for float32-only mode")

    if has_float32_marker and not using_float32_flag:
        pytest.skip("Skipped: float32-only test requires --float32-only flag")


def pytest_report_header(config):
    """Show JAX x64 status in test output."""
    return [f"JAX float64 enabled: {jax.config.read('jax_enable_x64')}"]
