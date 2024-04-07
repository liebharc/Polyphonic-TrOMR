import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.inception_resnet_v2 import InceptionResnetV2
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from einops import repeat

from configs import Config

class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size, patch_size=16, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(img_size=img_size, patch_size=patch_size, *args, **kwargs)
        self.height, self.width = img_size
        self.patch_size = patch_size

    def forward_features(self, x):
        B, c, h, w = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = h//self.patch_size, w//self.patch_size
        pos_emb_ind = repeat(torch.arange(h)*(self.width//self.patch_size-w), 'h -> (h w)', w=w)+torch.arange(h*w)
        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind+1), dim=0).long()

        # Normalize ranges
        x = x[:, range(0, len(pos_emb_ind))]
        x += self.pos_embed[:, pos_emb_ind]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

def get_encoder(config: Config):
    backbone_layers = list(config.backbone_layers)
    #backbone = InceptionResnetV2(
    #    num_classes=0, global_pool='', in_chans=config.channels
    #)
    backbone = ResNetV2(
        num_classes=0, global_pool='', in_chans=config.channels,
        drop_rate=0.8, drop_path_rate=0.8,
        layers=backbone_layers, preact=True, stem_type='same', conv_layer=StdConv2dSame
    )
    min_patch_size = 2**(len(backbone_layers)+1)

    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, 'patch_size needs to be multiple of %i with current backbone configuration' % min_patch_size
        return HybridEmbed(**x, patch_size=ps//min_patch_size, backbone=backbone)

    # TODO: A shorter run showed that a VisionTransformer gives the same results as a CustomVisionTransformer
    encoder = VisionTransformer(img_size=(config.max_height, config.max_width),
                                      patch_size=config.patch_size,
                                      in_chans=config.channels,
                                      num_classes=0,
                                      embed_dim=config.encoder_dim,
                                      depth=config.encoder_depth,
                                      num_heads=config.encoder_heads,
                                      embed_layer=embed_layer,
                                      global_pool=""
                                      )
    return encoder