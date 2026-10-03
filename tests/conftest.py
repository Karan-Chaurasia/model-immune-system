import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: full end-to-end pipeline tests")
