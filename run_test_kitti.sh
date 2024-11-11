python -u SemanticSegmentation.py --config configs/kitti_SemSeg/kitti_SemSeg.yaml SOLVER.gpu 0,1,2,3,4,5,6,7, SOLVER.run evaluate SOLVER.alias kitti_SemSeg  SOLVER.ckpt /data/wangx/workspace/HexFormer/logs/log_3Dconv_4Dattention_CPE_RPE_huge_kitti_semseg/best_model.pth SOLVER.dist_url tcp://localhost:12332 > /dev/null

