"""TA-DOTCNet with residual adaptation and optimal-transport regularization."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from cross_scene_adapter import Projector
from optimal_transport import sinkhorn_log_stabilized
EPS = 1e-12
PROTOCOL = 'core_train_test_v1'

class TADOTCNet(nn.Module):
    """Residual image adaptation with a parameter-free, two-domain DOTC head."""

    def __init__(self, num_classes=6, clip_model=None, tokenizer=None, feature_dim=512,
                 proj_dim=512, text_anchors=None):
        super(TADOTCNet, self).__init__()
        assert clip_model is not None and tokenizer is not None
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.dim = feature_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.projector = Projector(input_dim=self.dim, output_dim=self.proj_dim)
        # Preserve the validated seed sequence before classifier initialization.
        # This temporary layer is never registered, trained, or saved.
        nn.Linear(self.dim, self.proj_dim)
        self.classifier = FusionHead(input_dim=self.proj_dim * 2, num_classes=num_classes)
        self.register_buffer('fixed_text_anchors', torch.zeros(num_classes, proj_dim))
        if text_anchors is not None:
            self.set_text_anchors(text_anchors)
        self.sinkhorn_stats = {'attempts': 0, 'successes': 0, 'fallbacks': 0}
        self.sinkhorn_debug = False
        self.ot_epsilon = 0.1
        self.ot_max_iter = 200
        self.nu_smooth = 0.1
        self._last_kl = (0.0, 0.0, 0.0)
        self._last_losses = (0.0, 0.0)
        self.register_buffer('global_src_proto', torch.zeros(1, proj_dim))
        self.register_buffer('global_tgt_proto', torch.zeros(1, proj_dim))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.long))

    def _projector_param_ref(self):
        param = next(self.projector.parameters(), None)
        if param is not None:
            return param
        return None

    def _projector_device_dtype(self, fallback_tensor=None):
        param = self._projector_param_ref()
        if param is not None:
            return (param.device, param.dtype)
        if fallback_tensor is not None:
            return (fallback_tensor.device, fallback_tensor.dtype)
        return (torch.device('cpu'), torch.float32)

    def forward(self, source_img, target_img, s_label, text_src, text_tgt=None):
        """Return both domain logits and losses; evaluation also computes three KL terms."""
        src_img_feat = self.clip_model.encode_image(source_img)
        tgt_img_feat = self.clip_model.encode_image(target_img)
        src_img_feat = F.normalize(src_img_feat, dim=-1)
        tgt_img_feat = F.normalize(tgt_img_feat, dim=-1)
        (_, proj_dtype) = self._projector_device_dtype(src_img_feat)
        src_img_feat = src_img_feat.to(proj_dtype)
        tgt_img_feat = tgt_img_feat.to(proj_dtype)
        src_img_proto = F.normalize(self.projector(src_img_feat), dim=-1)
        tgt_img_proto = F.normalize(self.projector(tgt_img_feat), dim=-1)
        src_mean = src_img_proto.mean(dim=0, keepdim=True).detach()
        tgt_mean = tgt_img_proto.mean(dim=0, keepdim=True).detach()
        # Historical evaluation also updates these cumulative batch prototypes.
        self.num_updates += 1
        n = self.num_updates.item()
        if n == 1:
            self.global_src_proto.data.copy_(src_mean)
            self.global_tgt_proto.data.copy_(tgt_mean)
        else:
            alpha = 1.0 / n
            self.global_src_proto.data.mul_(1.0 - alpha).add_(src_mean, alpha=alpha)
            self.global_tgt_proto.data.mul_(1.0 - alpha).add_(tgt_mean, alpha=alpha)

        def encode_text_list_to_proto(text_list):
            toks = self.tokenizer(text_list)
            toks = toks.to(next(self.clip_model.parameters()).device)
            with torch.no_grad():
                feat = self.clip_model.encode_text(toks)
            return F.normalize(feat, dim=-1)
        if isinstance(text_src, torch.Tensor):
            txt_src_proto = self._ensure_text_in_proj_space(text_src)
        elif isinstance(text_src, list):
            txt_src_proto = self._ensure_text_in_proj_space(encode_text_list_to_proto(text_src))
        else:
            raise TypeError('Source text must be a feature tensor or a list of prompts')
        if text_tgt is None:
            txt_tgt_proto = None
        elif isinstance(text_tgt, torch.Tensor):
            txt_tgt_proto = self._ensure_text_in_proj_space(text_tgt)
        elif isinstance(text_tgt, list):
            txt_tgt_proto = self._ensure_text_in_proj_space(encode_text_list_to_proto(text_tgt))
        else:
            raise TypeError('Target text must be a feature tensor, a list of prompts, or None')

        def build_text_counterpart(img_proto, global_proto, txt_proto, temperature=0.07):
            if txt_proto is None:
                return None
            combined_img = F.normalize(0.5 * img_proto + 0.5 * global_proto, dim=-1)
            sim = combined_img @ txt_proto.t()
            weight = F.softmax(sim / temperature, dim=1)
            txt_cp = weight @ txt_proto
            return txt_cp
        src_txt_cp = build_text_counterpart(src_img_proto, self.global_src_proto, txt_src_proto)
        tgt_txt_cp = build_text_counterpart(tgt_img_proto, self.global_tgt_proto, txt_tgt_proto)
        feat_src = torch.cat([src_img_proto, src_txt_cp], dim=1)
        feat_tgt = torch.cat([tgt_img_proto, tgt_txt_cp if tgt_txt_cp is not None else tgt_img_proto], dim=1)
        src_out = self.classifier(feat_src)
        ce_src = F.cross_entropy(src_out, s_label)
        tgt_out = self.classifier(feat_tgt)
        with torch.no_grad():
            tgt_pseudo = tgt_out.argmax(dim=1)
        loss_tgt_ce = F.cross_entropy(tgt_out, tgt_pseudo)
        source_text, target_text, union_image = self.compute_kl_components(src_img_proto, tgt_img_proto)
        dec_loss = source_text + target_text + union_image
        self._last_kl = tuple(float(value.detach()) for value in (source_text, target_text, union_image))
        self._last_losses = (float(ce_src.detach()), float(dec_loss.detach()))
        return {'src_logits': src_out, 'tgt_logits': tgt_out, 'loss_src_ce': ce_src, 'loss_tgt_ce': loss_tgt_ce, 'loss_dec': dec_loss}

    def _ensure_text_in_proj_space(self, text_tensor):
        if text_tensor.ndim != 2 or text_tensor.shape[1] != self.proj_dim:
            raise ValueError('Text features must match the projected image dimensions')
        device, dtype = self._projector_device_dtype(text_tensor)
        return text_tensor.to(device=device, dtype=dtype)

    @torch.no_grad()
    def set_text_anchors(self, anchors):
        if anchors.shape != self.fixed_text_anchors.shape:
            raise ValueError('Fixed text anchors must have one row per class')
        if not torch.isfinite(anchors).all() or (anchors.norm(dim=1) == 0).any():
            raise ValueError('Fixed text anchors must be finite and nonzero')
        self.fixed_text_anchors.copy_(anchors.detach().to(self.fixed_text_anchors))

    def compute_kl_components(self, src_img_proj, tgt_img_proj):
        # Classification text arguments can use first prompts at evaluation;
        # DOTC always uses the saved mean-of-five training semantic pool.
        if src_img_proj is None or tgt_img_proj is None or not len(src_img_proj) or not len(tgt_img_proj):
            raise ValueError('DOTC requires both source and target image features')
        pool = torch.cat([src_img_proj, tgt_img_proj], dim=0)
        text_anchors = self.fixed_text_anchors.detach().to(pool)
        if not torch.isfinite(text_anchors).all() or (text_anchors.norm(dim=1) == 0).any():
            raise ValueError('Initialize the fixed text semantic pool before computing DOTC')
        scores = F.normalize(pool, dim=-1) @ F.normalize(text_anchors, dim=-1).t()
        indices = scores.argmax(dim=0)
        # Keep gradients through the text-selected image features.
        image_anchors = pool[indices]
        self._last_anchor_selection = dict(indices=indices.detach().cpu().tolist(),
            pool_size=len(pool), source_samples=len(src_img_proj), target_samples=len(tgt_img_proj),
            selection_scope='current source-target batch', anchor_requires_grad=image_anchors.requires_grad)
        kwargs = dict(sinkhorn_eps=self.ot_epsilon, sinkhorn_max_iter=self.ot_max_iter,
                      sinkhorn_tol=1e-6)
        return (self._kl_between(src_img_proj, text_anchors, **kwargs),
                self._kl_between(tgt_img_proj, text_anchors, **kwargs),
                self._kl_between(pool, image_anchors, **kwargs))

    def _kl_between(self, feat, centers, sinkhorn_eps=0.1, sinkhorn_max_iter=200, sinkhorn_tol=1e-6):
        feat = feat.to(device=centers.device, dtype=centers.dtype)
        if feat.ndim != 2 or centers.ndim != 2 or not len(feat) or not len(centers):
            raise ValueError('KL requires nonempty feature and anchor matrices')
        if not torch.isfinite(feat).all() or not torch.isfinite(centers).all():
            raise ValueError('KL features or anchors contain NaN/Inf')
        sim = F.normalize(feat, dim=-1) @ F.normalize(centers, dim=-1).t()
        # The historical Student-t kernel only determines the OT column prior.
        # It does not replace the cosine-softmax assignment in the KL objective.
        cost = 1.0 - sim
        cost = cost / (cost.mean().detach() + EPS)
        cost = torch.clamp(cost, min=EPS)
        alpha = 1.0
        prior_q = 1.0 / (1.0 + cost / alpha)
        prior_q = prior_q ** ((alpha + 1.0) / 2.0)
        prior_q = prior_q / (prior_q.sum(dim=1, keepdim=True) + EPS)
        prior_q = torch.clamp(prior_q, min=EPS)
        batch, classes = sim.shape
        mu = torch.full((batch,), 1.0 / max(batch, 1), device=centers.device, dtype=centers.dtype)
        nu_raw = torch.sum(prior_q, dim=0).to(device=centers.device, dtype=centers.dtype)
        nu = nu_raw / (nu_raw.sum() + EPS)
        uniform = torch.full((classes,), 1.0 / classes, device=centers.device, dtype=centers.dtype)
        nu = (1.0 - self.nu_smooth) * nu + self.nu_smooth * uniform
        nu = torch.clamp(nu, min=EPS)
        q = F.softmax(sim, dim=1)
        raw_cost = 1.0 - sim
        self.sinkhorn_stats['attempts'] += 1
        transport = sinkhorn_log_stabilized(raw_cost, mu, nu, epsilon=sinkhorn_eps,
            max_iter=sinkhorn_max_iter, tol=sinkhorn_tol, debug=self.sinkhorn_debug)
        if transport is None or not torch.isfinite(transport).all():
            raise FloatingPointError('OT returned a missing or non-finite plan')
        transport = transport.to(device=centers.device, dtype=centers.dtype)
        transport = torch.clamp(transport, min=0.0)
        if transport.shape != raw_cost.shape or (transport.sum(dim=1) <= 0).any():
            raise FloatingPointError('OT returned a wrong-shape or empty-row plan')
        self.sinkhorn_stats['successes'] += 1
        target = transport / (transport.sum(dim=1, keepdim=True) + EPS)
        target = torch.clamp(target, min=EPS).detach()
        kl = F.kl_div(torch.log(q + EPS), target, reduction='batchmean')
        if not torch.isfinite(kl):
            raise FloatingPointError('Non-finite OT KL')
        return kl

    def encode_projected_features(self, image):
        """Return unnormalized projector features for target-domain BN calibration."""
        feat = self.clip_model.encode_image(image)
        feat = F.normalize(feat, dim=-1)
        (_, proj_dtype) = self._projector_device_dtype(feat)
        if feat.dtype != proj_dtype:
            feat = feat.to(proj_dtype)
        feat = self.projector(feat)
        return feat

class FusionHead(nn.Module):

    def __init__(self, input_dim=512, num_classes=6):
        super(FusionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def model_from_checkpoint(clip_model, tokenizer, checkpoint):
    """Construct the core model using its saved fixed text anchors."""
    protocol = checkpoint.get('config', {}).get('protocol')
    if protocol != PROTOCOL:
        raise ValueError(f'Checkpoint protocol {protocol!r} does not match {PROTOCOL!r}')
    state = checkpoint['model_state_dict']
    text_anchors = state['fixed_text_anchors']
    return TADOTCNet(clip_model=clip_model, tokenizer=tokenizer,
        feature_dim=state['projector.projection.0.weight'].shape[1],
        proj_dim=text_anchors.shape[1], num_classes=text_anchors.shape[0],
        text_anchors=text_anchors)
