python -u SemanticSegmentation.py --config configs/hoi4d_SemSeg/hoi4d_SemSeg.yaml SOLVER.gpu 4,5,6,7, SOLVER.run evaluate SOLVER.alias hoi4d_SemSeg SOLVER.ckpt /data/wangx/workspace/HexFormer/logs_hoi4d/log_3Dconv_4Dattention_CPE_RPE_huge_d10_hoi4d_semseg/best_model.pth SOLVER.dist_url tcp://localhost:12332 > /dev/null

