"""
    __init__.py
    Created by pierre
    4/26/22 - 10:08 AM
    Description:
    # Enter file description
 """
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser()
    args = parser.parse_args(argv)
    pass


if __name__ == "__main__":
    raise SystemExit(main())
