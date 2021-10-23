#!/usr/bin/env bash

# Discovers and runs all the tests in the project
# All tests should be contained in files
# which match the patter *_test.py.
# Also files for testing must not contain
# anything save the testing code.

python -m unittest discover . -p "*_test.py"
