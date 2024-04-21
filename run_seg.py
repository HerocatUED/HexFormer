# script to run HexFormerSeg for Segmentaition Task

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, required=False, default="train")
parser.add_argument("--alias", type=str, required=False, default="hoi4d")
parser.add_argument("--gpu", type=str, required=False, default="5")
parser.add_argument("--port", type=str, required=False, default="10008")
parser.add_argument("--ckpt", type=str, required=False, default="")
args = parser.parse_args()


def execute_command(cmds):
    cmd = " ".join(cmds)
    print("Execute: \n" + cmd + "\n")
    os.system(cmd)


def train():
    print(f"Training on dataset {args.alias}")
    cmds = [
        "python segmentation.py",
        f"--config config/{args.alias}/seg_{args.alias}.yaml",
        "SOLVER.gpu  {},".format(args.gpu),
        "SOLVER.alias  {}".format(args.alias),
        "SOLVER.dist_url tcp://localhost:{}".format(args.port),
    ]
    execute_command(cmds)


def test():
    # get the predicted probabilities for each point
    print(f"Testing on dataset {args.alias}")
    print(f"Using checkpoint {args.ckpt}")
    ckpt = args.ckpt  # use args.ckpt if provided
    cmds = [
        "python segmentation.py",
        f"--config config/{args.alias}/seg_{args.alias}.yaml",
        "LOSS.mask -255",  # to keep all points
        "SOLVER.gpu  {},".format(args.gpu),
        "SOLVER.run evaluate",
        "SOLVER.eval_epoch 1",  # can't voting with more than 1 predictions, out of memory
        "SOLVER.alias test_{}".format(args.alias),
        "SOLVER.ckpt {}".format(ckpt),
    ]
    execute_command(cmds)


# def validate():
#   # get the predicted probabilities for each point
#   ckpt = ('logs/scannet/octformer_{}/best_model.pth'.format(args.alias)
#           if args.ckpt == '\'\'' else args.ckpt)   # use args.ckpt if provided
#   cmds = [
#       'python segmentation.py',
#       '--config configs/seg_scannet.yaml',
#       'LOSS.mask -255',       # to keep all points
#       'SOLVER.gpu  {},'.format(args.gpu),
#       'SOLVER.run evaluate',
#       'SOLVER.eval_epoch 120',  # voting with 120 predictions
#       'SOLVER.alias val_{}'.format(args.alias),
#       'SOLVER.ckpt {}'.format(ckpt),
#       'DATA.test.batch_size 1',
#       'DATA.test.distort True',]
#   execute_command(cmds)


if __name__ == "__main__":
    eval("%s()" % args.run)
