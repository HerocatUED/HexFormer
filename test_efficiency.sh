python -u SemanticSegmentation.py --config configs/hoi4d_SemSeg/hoi4d_SemSeg.yaml LOSS.mask -255 SOLVER.gpu  5, SOLVER.run evaluate SOLVER.eval_epoch 1 SOLVER.alias test_hoi4d_SemSeg SOLVER.ckpt ./logs_hoi4d/small/00050.model.pth | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done > output-small-test.log

python -u SemanticSegmentation.py --config configs/hoi4d_SemSeg/hoi4d_SemSeg.yaml LOSS.mask -255 SOLVER.gpu  5, SOLVER.run train SOLVER.eval_epoch 1 SOLVER.alias train_hoi4d_SemSeg | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done > output-small-train.log

python -u SemanticSegmentation.py --config configs/hoi4d_SemSeg/hoi4d_SemSeg.yaml LOSS.mask -255 SOLVER.gpu  6, SOLVER.run evaluate SOLVER.eval_epoch 1 SOLVER.alias test_hoi4d_SemSeg SOLVER.ckpt ./logs_hoi4d/large/00001.model.pth | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done > output-large-test.log

python -u SemanticSegmentation.py --config configs/hoi4d_SemSeg/hoi4d_SemSeg.yaml LOSS.mask -255 SOLVER.gpu  5, SOLVER.run train SOLVER.eval_epoch 1 SOLVER.alias train_hoi4d_SemSeg | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done > output-large-train.log

python -u SemanticSegmentation.py --config configs/hoi4d_SemSeg/hoi4d_SemSeg.yaml LOSS.mask -255 SOLVER.gpu  6, SOLVER.run evaluate SOLVER.eval_epoch 1 SOLVER.alias test_hoi4d_SemSeg SOLVER.ckpt ./logs_hoi4d/base/00001.model.pth | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done > output-base-test.log

python -u SemanticSegmentation.py --config configs/hoi4d_SemSeg/hoi4d_SemSeg.yaml LOSS.mask -255 SOLVER.gpu  5, SOLVER.run train SOLVER.eval_epoch 1 SOLVER.alias train_hoi4d_SemSeg | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done > output-base-train.log