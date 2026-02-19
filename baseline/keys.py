"""Utilities for loading API keys and creating provider clients.

This module only reads provider keys from environment variables. It exposes
helpers to retrieve each provider key, validate key availability, and construct
provider SDK clients on demand.
"""

from __future__ import annotations

import os
from typing import Dict, List


_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _get_required_env_var(var_name: str, provider: str) -> str:
    """Return a required environment variable.

    Args:
        var_name: Environment variable name to look up.
        provider: Provider label used in raised error messages.

    Returns:
        The environment variable value.

    Raises:
        RuntimeError: If the environment variable is missing or empty.
    """
    value = os.getenv(var_name)
    if value:
        return value
    raise RuntimeError(
        f"Missing API key for provider '{provider}'. "
        f"Set environment variable {var_name}."
    )


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key from ``ANTHROPIC_API_KEY``.

    Raises:
        RuntimeError: If ``ANTHROPIC_API_KEY`` is missing.
    """
    return _get_required_env_var("ANTHROPIC_API_KEY", "anthropic")


def get_openai_api_key() -> str:
    """Return the OpenAI API key from ``OPENAI_API_KEY``.

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is missing.
    """
    return _get_required_env_var("OPENAI_API_KEY", "openai")


def get_google_api_key() -> str:
    """Return the Google API key from ``GOOGLE_API_KEY``.

    Raises:
        RuntimeError: If ``GOOGLE_API_KEY`` is missing.
    """
    return _get_required_env_var("GOOGLE_API_KEY", "google")


def get_groq_api_key() -> str:
    """Return the Groq API key from ``GROQ_API_KEY``.

    Raises:
        RuntimeError: If ``GROQ_API_KEY`` is missing.
    """
    return _get_required_env_var("GROQ_API_KEY", "groq")


def validate_all() -> Dict[str, List[str]]:
    """Check whether all provider API key environment variables are available.

    Returns:
        A dictionary with two keys:
            - ``available``: providers with a non-empty configured key.
            - ``missing``: providers with a missing or empty key.
    """
    available: List[str] = []
    missing: List[str] = []

    for provider, env_var in _PROVIDER_ENV_VARS.items():
        if os.getenv(env_var):
            available.append(provider)
        else:
            missing.append(provider)

    return {"available": available, "missing": missing}


def get_client(provider: str):
    """Return an initialized API client for a supported provider.

    Args:
        provider: Provider name. Supported values are ``anthropic``, ``openai``,
            ``google``, and ``groq``.

    Returns:
        Initialized provider SDK client object. For Google, this returns the
        configured ``google.generativeai`` module.

    Raises:
        ValueError: If provider is not supported.
        RuntimeError: If the provider API key is missing.
        ImportError: If the provider SDK is not installed.
    """
    normalized = provider.lower().strip()

    if normalized == "anthropic":
        from anthropic import Anthropic

        return Anthropic(api_key=get_anthropic_api_key())

    if normalized == "openai":
        from openai import OpenAI

        return OpenAI(api_key=get_openai_api_key())

    if normalized == "google":
        import google.generativeai as genai

        genai.configure(api_key=get_google_api_key())
        return genai

    if normalized == "groq":
        from groq import Groq

        return Groq(api_key=get_groq_api_key())

    supported = ", ".join(sorted(_PROVIDER_ENV_VARS.keys()))
    raise ValueError(
        f"Unsupported provider '{provider}'. "
        f"Supported providers: {supported}."
    )
