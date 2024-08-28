from models.sofi import (SOFI, Backbone, Joiner, 
	PositionEmbeddingSine, DeformableTransformer)

hidden_dim = 256

sofi = SOFI(
        backbone=Joiner(
        	backbone=Backbone(
        		name= 'resnet50',
                train_backbone= True,
                return_interm_indices= [1, 2],
                dilation= False
                ),
        	position_embedding=PositionEmbeddingSine(
        		num_pos_feats=hidden_dim//2,
        		temperature=10000,
        		normalize=True
        		),
        	),
        transformer=DeformableTransformer(
        	d_model=hidden_dim,
	        nhead=8,
	        num_encoder_layers=6,
	        num_decoder_layers=6,
	        dim_feedforward=2048,
	        dropout=0.1,
	        activation="relu",
	        return_intermediate_dec=True,
	        num_feature_levels=2,
	        dec_n_points=8,
	        enc_n_points=32,
        	),
        num_classes=3,
        num_queries=3,
        num_feature_levels=2,
        mixed_query = True,
        aux_loss=True,
    )