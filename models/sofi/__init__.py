# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from .sofi import build
def build_model(args):
    return build(args)

from .sofi import SOFI
from .backbone import Backbone, Joiner
from .deformable_transformer import DeformableTransformer  
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
