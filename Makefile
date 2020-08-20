all: test

test:
	coverage run --source=. -m pytest
	coverage report -m
