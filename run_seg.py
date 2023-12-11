# script to run HexFormerSeg for Segmentaition Task

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=False, default='train')
parser.add_argument('--alias', type=str, required=False, default='hoi4d')
parser.add_argument('--gpu', type=str, required=False, default='5')
parser.add_argument('--port', type=str, required=False, default='10008')
parser.add_argument('--ckpt', type=str, required=False, default='\'\'')
args = parser.parse_args()


def execute_command(cmds):
  cmd = ' '.join(cmds)
  print('Execute: \n' + cmd + '\n')
  os.system(cmd)


def train():
  print(f"using dataset {args.alias}")
  cmds = [
      'python segmentation.py',
      f'--config data_utils/config/{args.alias}/seg_{args.alias}.yaml',
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.alias  {}'.format(args.alias),
      'SOLVER.dist_url tcp://localhost:{}'.format(args.port),]
  execute_command(cmds)


def test():
  # get the predicted probabilities for each point
  ckpt = (args.ckpt)   # use args.ckpt if provided
  cmds = [
      'python segmentation.py',
      '--config data/configs/seg_hoi4d.yaml',
      'LOSS.mask -255',       # to keep all points
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.run evaluate',
      'SOLVER.eval_epoch 72',  # voting with 72 predictions
      'SOLVER.alias test_{}'.format(args.alias),
      'SOLVER.ckpt {}'.format(ckpt),]
  execute_command(cmds)

#   # map the probabilities to labels
#   cmds = [
#       'python tools/seg_scannet.py',
#       '--run generate_output_seg',
#       '--path_pred logs/scannet/octformer_test_{}'.format(args.alias),
#       '--path_out logs/scannet/octformer_test_seg_{}'.format(args.alias),
#       '--filelist  data/scannet.npz/scannetv2_test.txt', ]
#   execute_command(cmds)


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

#   # map the probabilities to labels
#   cmds = [
#       'python tools/seg_scannet.py',
#       '--run generate_output_seg',
#       '--path_pred logs/scannet/octformer_val_{}'.format(args.alias),
#       '--path_out  logs/scannet/octformer_val_seg_{}'.format(args.alias),
#       '--filelist  data/scannet.npz/scannetv2_val_npz.txt', ]
#   execute_command(cmds)

#   # calculate the mIoU
#   cmds = [
#       'python tools/seg_scannet.py',
#       '--run calc_iou',
#       '--path_in data/scannet.npz/train',
#       '--path_pred logs/scannet/octformer_val_seg_{}'.format(args.alias), ]
#   execute_command(cmds)


if __name__ == '__main__':
  eval('%s()' % args.run)
