# -*- coding: utf-8 -*-
"""RL4EQ Continual PPO 新路线入口。"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")


if __name__ == "__main__":
    main()
