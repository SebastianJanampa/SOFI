from datasets.gsv_dataset import build_gsv
from datasets.hlw_dataset import build_hlw
from datasets.image_dataset import build_image
from datasets.holicity_dataset import build_holicity

def build_gsv_dataset(image_set, args):
    return build_gsv(image_set, args)

def build_yud_dataset(image_set, args):
    return build_yud(image_set, args)

def build_hlw_dataset(image_set, args):
    return build_hlw(image_set, args)

def build_image_dataset(image_set, args):
    return build_image(image_set, args)
    
def build_holicity_dataset(image_set, args):
    return build_holicity(image_set, args)
