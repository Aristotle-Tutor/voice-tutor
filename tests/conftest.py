from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

pytest.register_assert_rewrite("providers.testing.contracts")

# Fill keys from .env unless the environment already has a value. Unlike
# load_dotenv(), an empty variable (some shells export ANTHROPIC_API_KEY="")
# does not hide the .env value.
for _name, _value in dotenv_values(Path(__file__).parents[1] / ".env").items():
    if _value and not os.environ.get(_name):
        os.environ[_name] = _value
