#! /usr/bin/env python
"""Switch puffins to the git branch corresponding to a particular project."""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Checkout a particular projects' puffins branch")
    parser.add_argument(
        "branch_name",
        action="store",
        nargs="?",
        default="master",
        type=str,
        help="Name of the puffins git branch to be checked out.",
    )
    branch_name = parser.parse_args().branch_name
    setup_puffins(branch_name)
