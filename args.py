import argparse
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    
    # Solver
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--no_mixed_queries', dest='mixed_queries', action='store_false')
    parser.add_argument('--resolution', default='high', type=str)
    parser.add_argument('--dense_fusion', action='store_true')


    # Model parameters
    parser.add_argument('--model', type=str, default='sofi', help='Model name')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--no_svd', dest='use_svd', action='store_false',
                        help="Use SVD decomposition to estimate zvp")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=3, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--coarse_frozen',action='store_true', help="Only require for MSCC")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Loss coefficients
    parser.add_argument('--zvp_loss_coef', default=5, type=float)
    parser.add_argument('--fovy_loss_coef', default=5, type=float)
    parser.add_argument('--hl_loss_coef', default=5, type=float)
    parser.add_argument('--line_score_loss_coef', default=1, type=float)
    parser.add_argument('--line_class_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # * Line angles
    parser.add_argument('--line_pos_angle', default=88.0, type=float)
    parser.add_argument('--line_neg_angle', default=85.0, type=float)

    # dataset parameters
    parser.add_argument('--dataset', default='gsv')
    parser.add_argument('--masks', default=False, type=bool)

    parser.add_argument('--input_width', default=512, type=int)
    parser.add_argument('--input_height', default=512, type=int)
    parser.add_argument('--min_line_length', default=10, type=int)
    parser.add_argument('--num_input_lines', default=512, type=int)
    parser.add_argument('--num_input_vert_lines', default=128, type=int)
    parser.add_argument('--vert_line_angle', default=22.5, type=float)
    parser.add_argument('--return_vert_lines', default=False, type=bool)

    parser.add_argument('--output_dir', default='logs/gsv',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--append_word', default=None, type=str, help="Name of the convolutional backbone to use")

    parser.add_argument('--print_freq', default=200, type=int)


    # Benchmark FPS
    parser.add_argument('--num_iters', type=int, default=300, help='total iters to benchmark speed')
    parser.add_argument('--warm_iters', type=int, default=5, help='ignore first several iters that are very slow')

    return parser

