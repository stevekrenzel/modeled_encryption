test:
	PYTHONPATH='src':'tests' python -m unittest discover -s . -p '*_tests.py'
