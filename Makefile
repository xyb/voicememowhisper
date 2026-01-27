.PHONY: test test-cov

test:
	python -m pytest

test-cov:
	python -m pytest --cov=voicememowhisper --cov-report=term-missing

