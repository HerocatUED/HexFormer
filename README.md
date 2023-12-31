# HexFormer
VIT/Swin  - rpe(relative positional encoding/embedding)
~~ PVT/CoAt/Twins - conv / cpe(conditional positional encoding/embedding) ~~

Potential bugs: hextree.utils, hextree_conv

Unfinished tests: conv

**Note** 
torch version: function next() used in thsolver.solver;

Step 3. Expirements

TODO： 
- update to pytorch 2.0, speedup with torch.compile()
- loss design(lower weight for classes that have higher acc)
- settings, e.g. drop out rate?

