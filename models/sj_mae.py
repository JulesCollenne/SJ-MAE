import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class SJ_MAEViT(nn.Module):
    """ Siamese Jigsaw Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=8, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 w_mae=1., w_jigsaw=0.05, w_siam=1., mask_ratio_mae=0.75, mask_ratio_jigsaw=0.,
                 output_jigsaw=False, pred_dim=512):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.w_mae = w_mae
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(
            torch.rand(1, int(round(mask_ratio_mae * self.patch_embed.num_patches)), decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch TODO
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Jigsaw specifics
        self.mask_ratio_mae = mask_ratio_mae
        self.mask_ratio_jigsaw = mask_ratio_jigsaw
        self.jigsaw = torch.nn.Sequential(*[torch.nn.Linear(embed_dim, embed_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(embed_dim, embed_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(embed_dim, self.patch_embed.num_patches)])
        self.w_jigsaw = w_jigsaw
        self.output_jigsaw = output_jigsaw
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Siamese specifics
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-3)
        self.w_siam = w_siam

        self.siam_predictor = nn.Sequential(
            nn.Linear(embed_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, embed_dim)
        )

        self.temperature = 0.1

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        self.cosine_scores = []

    def forward(self, imgs):
        if isinstance(imgs, (list, tuple)):
            img1, img2 = imgs  # Siamese case
        else:
            img1 = imgs  # Standard case
            img2 = None

        loss = torch.tensor(0., device=imgs[0].device)
        losses = Munch()

        if self.w_mae > 0. or self.w_siam > 0.:
            latent, mask, ids_restore, ids_keep = self.forward_encoder(img1, True, self.mask_ratio_mae)

            if self.w_mae > 0.:
                # Reconstruction task
                pred_recon = self.forward_decoder(latent, ids_restore)
                loss = self.forward_loss(img1, pred_recon, mask) * self.w_mae
                losses.recon = loss.item()

        if self.w_jigsaw > 0. or self.w_siam > 0.:
            if img2 is None:
                latent2, mask2, ids_restore2, ids_keep2 = self.forward_encoder(img1, False, self.mask_ratio_jigsaw)
            else:
                latent2, mask2, ids_restore2, ids_keep2 = self.forward_encoder(img2, False, self.mask_ratio_jigsaw)

            # Jigsaw task
            if self.w_jigsaw > 0.:
                pred_jigsaw = self.forward_jigsaw(latent2)
                gt_jigsaw = ids_keep2.reshape(-1)
                loss_jigsaw = F.cross_entropy(pred_jigsaw, gt_jigsaw) * self.w_jigsaw
                loss += loss_jigsaw
                losses.jigsaw = loss_jigsaw.item()

            # Siamese task
            if self.w_siam > 0.:
                # Extract only CLS token (index 0)
                cls1 = latent[:, 0]
                cls2 = latent2[:, 0]
                p1 = self.siam_predictor(cls1)
                p2 = self.siam_predictor(cls2)

                z1 = F.normalize(p1, dim=1)
                z2 = F.normalize(p2, dim=1)

                # Compute pairwise dot products between z1 and z2
                logits = torch.mm(z1, z2.T) / self.temperature  # [B, B]
                labels = torch.arange(z1.size(0), device=z1.device)

                loss_siam = F.cross_entropy(logits, labels) * self.w_siam

                # SimSiam-style cosine similarity loss
                # loss_siam = (
                #                     (1 - self.cosine_similarity(cls1.detach(), p2).mean()) +
                #                     (1 - self.cosine_similarity(cls2.detach(), p1).mean())
                #             ) * 0.5 * self.w_siam

                loss += loss_siam
                losses.siam = loss_siam.item()

        outputs = (loss,)

        outputs += (pred_recon,) if self.w_mae > 0. else (None,)
        outputs += (mask,) if self.w_mae > 0. or self.w_siam > 0. else (None,)
        outputs += (losses,)

        if self.w_jigsaw > 0. and self.output_jigsaw:
            outputs += (pred_jigsaw,)
            outputs += (gt_jigsaw,)

        return outputs

    def forward_encoder(self, x, pos_embed, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        if pos_embed:
            x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        # MAE : append cls token
        if pos_embed:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        else:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_keep

    def forward_decoder(self, x, ids_restore=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], 1, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        if ids_restore is not None:
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_jigsaw(self, x):
        x = self.jigsaw(x[:, 1:])
        return x.reshape(-1, self.patch_embed.num_patches)

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # Make target [N, L, p*p*3]
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(round(L * (1 - mask_ratio)))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

