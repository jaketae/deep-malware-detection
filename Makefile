.PHONY: quality style

check_dirs := .

quality: # Check that source code meets quality standards
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs) --max-line-length 119

style: # Format source code automatically
	black $(check_dirs)
	isort $(check_dirs)
