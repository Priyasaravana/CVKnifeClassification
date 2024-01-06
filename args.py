# Copyright (c) EEEM071, University of Surrey

import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument("--n_classes", type=int, default=192, help="Number of classes")
    parser.add_argument("--img_weight", type=int, default=224, help="width of an image")
    parser.add_argument("--img_height", type=int, default=224, help="height of an image")
    parser.add_argument("--learning_rate", default=0.00005, type=float, help="initial learning rate")
    parser.add_argument("--epochs", default=20, type=int, help="maximum epochs to run")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")

    return parser
