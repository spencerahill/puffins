#! /usr/bin/env python
"""Switch puffins to the git branch corresponding to a particular project."""
import argparse
import importlib
import os.path

import git


def package_rootdir(name):
    initfile = importlib.util.find_spec(name).origin
    return os.path.split(os.path.split(initfile)[0])[0]


def checkout_if_clean(repo, branch_name):
    clean_and_tracked = not (repo.is_dirty() or repo.untracked_files)
    if clean_and_tracked:
        repo.git.checkout(branch_name)
    else:
        raise Exception(f"The repo in the directory '{repo.working_dir}' "
                        "has an untracked file or uncommitted changes.  These "
                        "must be handled before puffins gets switched to "
                        "this project's branch.")


def setup_puffins(branch_name="master"):
    puffins_repo = git.Repo(package_rootdir("puffins"))
    checkout_if_clean(puffins_repo, branch_name)


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
