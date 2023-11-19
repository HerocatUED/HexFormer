# HexFormer

Step 1. Hextree

Step 2. HexFormer

VIT/Swin  - rpe(relative positional encoding/embedding)
~~ PVT/CoAt/Twins - conv / cpe(conditional positional encoding/embedding) ~~

TODO: **patch embedding**
RPE: +B(done), MLP*
loss design(lower weight for classes that have higher acc)

Potential bugs: hextree.utils, hextree_conv,
Unfinished tests: conv, interp, 
Pooling, Interp, didn't operate on dimention *t*

**Note** 

torch version: function next() used in thsolver.solver;

octree_linear_upsample is not implemented;

Step 3. Expirements

TODO： 
data augmentations



