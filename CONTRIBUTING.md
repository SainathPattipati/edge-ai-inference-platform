# Contributing

See LICENSE and development setup in README.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=src
black src/
ruff check src/
```
