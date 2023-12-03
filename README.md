# HexFormer

Step 1. Hextree

Step 2. HexFormer

VIT/Swin  - rpe(relative positional encoding/embedding)
~~ PVT/CoAt/Twins - conv / cpe(conditional positional encoding/embedding) ~~

TODO: **patch embedding**
RPE: +B(done), MLP*
loss design(lower weight for classes that have higher acc)
thsolver.solver: LOSS.mask?

Potential bugs: hextree.utils, hextree_conv,
Unfinished tests: conv

batch_id & key after pooling mask
**Note** 

torch version: function next() used in thsolver.solver;

octree_linear_upsample is not implemented;

Step 3. Expirements

TODO： 
data augmentations

tidy up scripts and utils

update to pytorch speedup 

