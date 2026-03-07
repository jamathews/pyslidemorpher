#!/usr/bin/env python3
"""
Entry point for running pyslidemorpher as a module.
This allows the package to be executed with: python -m pyslidemorpher
"""

from .cli import main

if __name__ == "__main__":
    main()