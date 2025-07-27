#!/usr/bin/env python
# -*- coding:utf8 -*-
import sys
from argparse import ArgumentParser
from ui import *


def inputArg():
    ap = ArgumentParser()
    ap.add_argument("-s", "--scene", type=int, default=0, help="scene index (default: 0)")
    ap.add_argument("-a", "--accumulate", type=int, default=1, help="accumulate flag (default: 1)")
    args = ap.parse_args()
    return args.scene, args.accumulate


if __name__ == "__main__":
    inputPath = "data/TCAR_DATA"
    scene_idx, accumulate = inputArg()
    w = genWindow(inputPath, scene_idx=scene_idx)



