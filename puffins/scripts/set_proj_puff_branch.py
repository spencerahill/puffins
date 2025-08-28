#! /usr/bin/env python
"""Switch puffins to the git branch corresponding to a particular project."""
import argparse
from puffins.nb_utils import setup_puffins

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Checkout a particular project's puffins branch"
    )
    parser.add_argument(
        "branch_name",
        nargs="?",
        default="master",
        help="Name of the puffins git branch to be checked out.",
    )
    args = parser.parse_args(argv)
    setup_puffins(args.branch_name)

if __name__ == "__main__":
    main()
