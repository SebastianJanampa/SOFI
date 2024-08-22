from .ctrlc import build as build1
from .ctrlcplus import build as build2

def build_ctrlc(args):
    return build1(args)
    
def build_ctrlcplus(args):
    return build2(args)
