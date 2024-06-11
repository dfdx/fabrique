#!/bin/bash
set -e

mypy . --install-types --non-interactive --exclude build/ --exclude tests/manual
isort .
black .