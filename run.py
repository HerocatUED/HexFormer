# script to run HexFormerSeg for Segmentaition Task

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, required=True)
parser.add_argument("--alias", type=str, required=True)
parser.add_argument("--gpu", type=str, required=False, default="0,1,2,3")
parser.add_argument("--port", type=str, required=False, default="10008")
parser.add_argument("--ckpt", type=str, required=False, default="")
args = parser.parse_args()


def execute_command(cmds):
    cmd = " ".join(cmds)
    print("Execute: \n" + cmd + "\n")
    os.system(cmd)


def check_and_init():
    if args.alias in ["kitti_SemSeg", "hoi4d_SemSeg"]:
        cmd = "python SemanticSegmentation.py"
    elif args.alias in ["hoi4d_ActSeg"]:
        cmd = "python ActionSegmentation.py"
    else: raise NotImplementedError
    return cmd

def train():
    cmd = check_and_init()
    print(f"Training on dataset {args.alias}")
    cmds = [
        cmd,
        f"--config configs/{args.alias}/{args.alias}.yaml",
        "SOLVER.gpu  {},".format(args.gpu),
        "SOLVER.alias  {}".format(args.alias),
        "SOLVER.dist_url tcp://localhost:{}".format(args.port),
    ]
    execute_command(cmds)


def test():
    cmd = check_and_init()
    # get the predicted probabilities for each point
    print(f"Testing on dataset {args.alias}")
    print(f"Using checkpoint {args.ckpt}")
    ckpt = args.ckpt  # use args.ckpt if provided
    cmds = [
        cmd,
        f"--config configs/{args.alias}/{args.alias}.yaml",
        "LOSS.mask -255",  # to keep all points
        "SOLVER.gpu  {},".format(args.gpu),
        "SOLVER.run evaluate",
        "SOLVER.eval_epoch 1",  # can't voting with more than 1 predictions, out of memory
        "SOLVER.alias test_{}".format(args.alias),
        "SOLVER.ckpt {}".format(ckpt),
    ]
    execute_command(cmds)


if __name__ == "__main__":
    eval("%s()" % args.run)
