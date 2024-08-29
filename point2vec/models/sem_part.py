from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy

from point2vec.modules.feature_upsampling import PointNetFeatureUpsampling
from point2vec.modules.pointnet import PointcloudTokenizer
from point2vec.modules.transformer import TransformerEncoder, TransformerEncoderOutput
from point2vec.utils import transforms
from point2vec.utils.checkpoint import extract_model_checkpoint


import numpy as np
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


class Point2VecSemPart(pl.LightningModule):
    def __init__(
            self,
            tokenizer_num_groups: int = 128,
            tokenizer_group_size: int = 32,
            tokenizer_group_radius: float | None = None,
            encoder_dim: int = 384,
            encoder_depth: int = 12,
            encoder_heads: int = 6,
            encoder_dropout: float = 0,
            encoder_attention_dropout: float = 0,
            encoder_drop_path_rate: float = 0.2,
            encoder_add_pos_at_every_layer: bool = True,
            encoder_unfreeze_epoch: int = 0,
            seg_head_fetch_layers: List[int] = [3, 7, 11],
            seg_head_dim: int = 512,
            seg_head_dropout: float = 0.5,
            learning_rate: float = 0.001,
            optimizer_adamw_weight_decay: float = 0.05,
            lr_scheduler_linear_warmup_epochs: int = 10,
            lr_scheduler_linear_warmup_start_lr: float = 1e-6,
            lr_scheduler_cosine_eta_min: float = 1e-6,
            pretrained_ckpt_path: str | None = None,
            train_transformations: List[str] = [
                "center",
                "unit_sphere",
            ],  # scale, center, unit_sphere, rotate, translate, height_norm
            val_transformations: List[str] = ["center", "unit_sphere"],
            transformation_scale_min: float = 0.8,
            transformation_scale_max: float = 1.2,
            transformation_scale_symmetries: Tuple[int, int, int] = (1, 0, 1),
            transformation_rotate_dims: List[int] = [1],
            transformation_rotate_degs: Optional[int] = None,
            transformation_translate: float = 0.2,
            transformation_height_normalize_dim: int = 1,
            momentum: float = 0.9,
            lr_decay: float = 0.5,
            dropout: float = 0.5,
            bn_decay: float = 0.5,
            emb_dims: int = 1024,
            k: int = 20,
            step_size: int = 40,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        def build_transformation(name: str) -> transforms.Transform:
            if name == "scale":
                return transforms.PointcloudScaling(
                    min=transformation_scale_min, max=transformation_scale_max
                )
            elif name == "center":
                return transforms.PointcloudCentering()
            elif name == "unit_sphere":
                return transforms.PointcloudUnitSphere()
            elif name == "rotate":
                return transforms.PointcloudRotation(
                    dims=transformation_rotate_dims, deg=transformation_rotate_degs
                )
            elif name == "translate":
                return transforms.PointcloudTranslation(transformation_translate)
            elif name == "height_norm":
                return transforms.PointcloudHeightNormalization(
                    transformation_height_normalize_dim
                )
            else:
                raise RuntimeError(f"No such transformation: {name}")

        self.train_transformations = transforms.Compose(
            [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [build_transformation(name) for name in val_transformations]
        )

        self.tokenizer = PointcloudTokenizer(
            num_groups=tokenizer_num_groups,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_group_radius,
            token_dim=encoder_dim,
        )

        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )

        dpr = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        # self.encoder = TransformerEncoder(
        #     embed_dim=encoder_dim,
        #     depth=encoder_depth,
        #     num_heads=encoder_heads,
        #     qkv_bias=True,
        #     drop_rate=encoder_dropout,
        #     attn_drop_rate=encoder_attention_dropout,
        #     drop_path_rate=dpr,
        #     add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        # )
        from point2vec.modules.point_mamba import MixerModel
        self.encoder = MixerModel(d_model=encoder_dim,
                                  n_layer=encoder_depth,
                                  rms_norm=False)

        point_dim = 3
        upsampling_dim = 384
        self.upsampling = PointNetFeatureUpsampling(in_channel=self.hparams.encoder_dim + point_dim,
                                                    mlp=[upsampling_dim, upsampling_dim])
        self.seg_head = nn.Sequential(
            nn.Conv1d(
                2 * self.hparams.encoder_dim + upsampling_dim,  # type: ignore
                self.hparams.seg_head_dim,  # type: ignore
                1,
                bias=False,
            ),
            nn.BatchNorm1d(self.hparams.seg_head_dim),  # type: ignore
            nn.ReLU(),
            nn.Dropout(self.hparams.seg_head_dropout),  # type: ignore
            nn.Conv1d(self.hparams.seg_head_dim, self.hparams.seg_head_dim // 2, 1, bias=False),  # type: ignore
            nn.BatchNorm1d(self.hparams.seg_head_dim // 2),  # type: ignore
            nn.ReLU(),
            # nn.Dropout(self.hparams.seg_head_dropout),
            nn.Conv1d(self.hparams.seg_head_dim // 2, 13, 1),  # type: ignore
        )

        # self.loss_func = nn.NLLLoss()

    def cal_loss(self, pred, gold, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

    def setup(self, stage: Optional[str] = None) -> None:
        self.num_classes: int = 13  # type: ignore

        label_embedding_dim = 64
        self.label_embedding = nn.Sequential(
            nn.Linear(self.num_classes, label_embedding_dim, bias=False),
            nn.BatchNorm1d(label_embedding_dim),
            nn.LeakyReLU(0.2),
        )

        self.val_macc = Accuracy(num_classes=self.num_classes, average="macro")

        if self.hparams.pretrained_ckpt_path is not None:  # type: ignore
            self.load_pretrained_checkpoint(self.hparams.pretrained_ckpt_path)  # type: ignore

        self.encoder.requires_grad_(False)  # will unfreeze later

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(self)
                logger.experiment.define_metric("val_acc", summary="last,max")
                logger.experiment.define_metric("val_macc", summary="last,max")
                # logger.experiment.define_metric("val_ins_miou", summary="last,max")
                # logger.experiment.define_metric("val_cat_miou", summary="last,max")

        self.train_true_cls = []
        self.train_pred_cls = []
        self.train_true_seg = []
        self.train_pred_seg = []
        self.test_true_cls = []
        self.test_pred_cls = []
        self.test_true_seg = []
        self.test_pred_seg = []

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        B, N, C = points.shape

        tokens: torch.Tensor
        centers: torch.Tensor
        tokens, centers = self.tokenizer(points)  # (B, T, C), (B, T, 3)
        pos_embeddings = self.positional_encoding(centers)
        # output: TransformerEncoderOutput = self.encoder(
        #     tokens, pos_embeddings, return_hidden_states=True
        # )

        output, all_output = self.encoder(
            tokens, pos_embeddings
        )
        # hidden_states = [F.layer_norm((output.hidden_states[i]), output.hidden_states[i].shape[-1:]) for i in self.hparams.seg_head_fetch_layers]  # type: ignore [(B, T, C)]
        hidden_states = [F.layer_norm((all_output[i]), all_output[i].shape[-1:]) for i in
                         self.hparams.seg_head_fetch_layers]
        token_features = torch.stack(hidden_states, dim=0).mean(0)  # (B, T, C)
        token_features_max = token_features.max(dim=1).values  # (B, C)
        token_features_mean = token_features.mean(dim=1)  # (B, C)

        global_feature = torch.cat(
            [token_features_max, token_features_mean], dim=-1
        )  # (B, 2*C')
        x = self.upsampling(points, centers, points, token_features)  # (B, N, C)
        x = torch.cat(
            [x, global_feature.unsqueeze(-1).expand(-1, -1, N).transpose(1, 2)], dim=-1
        )  # (B, N, C'); C' = 3*C + L
        x = self.seg_head(x.transpose(1, 2)).transpose(1, 2)  # (B, N, cls)
        return x

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # points: (B, N, 9)
        # label: (B,)

        points, seg = batch
        points = points[:, :, :3]

        points = self.train_transformations(points)
        seg_pred = self.forward(points)
        seg_pred = seg_pred.contiguous()

        loss = self.cal_loss(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
        self.log("train_loss", loss, on_epoch=True)

        pred = torch.max(seg_pred, dim=-1)[1]
        pred = pred.detach().cpu().numpy()
        seg = seg.cpu().numpy()

        self.train_true_cls.append(seg.reshape(-1))
        self.train_pred_cls.append(pred.reshape(-1))
        self.train_true_seg.append(seg)
        self.train_pred_seg.append(pred)

        # train_true_cls = np.concatenate(self.train_true_cls)
        # train_pred_cls = np.concatenate(self.train_pred_cls)
        # train_true_seg = np.concatenate(self.train_true_seg, axis=0)
        # train_pred_seg = np.concatenate(self.train_pred_seg, axis=0)

        # import sklearn.metrics as metrics
        # train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        # # self.train_acc(pred, seg)
        # self.log("train_acc", train_acc, on_epoch=True)

        return loss

    def training_epoch_end(
            self, outputs: List[Dict[str, List[torch.Tensor]]]
    ) -> torch.Tensor:
        # points: (B, N, 9)
        # label: (B,)
        train_true_cls = np.concatenate(self.train_true_cls)
        train_pred_cls = np.concatenate(self.train_pred_cls)
        train_true_seg = np.concatenate(self.train_true_seg, axis=0)
        train_pred_seg = np.concatenate(self.train_pred_seg, axis=0)

        import sklearn.metrics as metrics
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_pre_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_ious = self.calculate_sem_IoU(train_pred_seg, train_true_seg)
        self.log("train_acc", train_acc)
        self.log("avg_pre_class_acc", avg_pre_class_acc)
        self.log("train_ious", np.mean(train_ious))

    def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, List[torch.Tensor]]:
        # points: (B, N, 9)
        # seg: (B, N)


        points, seg = batch
        points = points[:, :, :3]

        points = self.train_transformations(points)
        seg_pred = self.forward(points)
        seg_pred = seg_pred.contiguous()

        # loss = self.loss_func(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
        loss = self.cal_loss(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
        self.log("val_loss", loss)

        pred = torch.max(seg_pred, dim=-1)[1]
        pred = pred.detach().cpu().numpy()
        seg = seg.cpu().numpy()

        self.test_true_cls.append(seg.reshape(-1))
        self.test_pred_cls.append(pred.reshape(-1))
        self.test_true_seg.append(seg)
        self.test_pred_seg.append(pred)

        # test_true_cls = np.concatenate(test_true_cls)
        # test_pred_cls = np.concatenate(test_pred_cls)
        # test_true_seg = np.concatenate(test_true_seg, axis=0)
        # test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        #
        # test_ious = self.calculate_sem_IoU(test_pred_seg, test_true_seg)
        # self.log("val_ious", test_ious)
        #
        # import sklearn.metrics as metrics
        # # self.val_acc(test_true_cls, test_pred_cls)
        # test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        # self.log("val_acc", test_acc)
        # avg_test_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        # # self.val_macc(test_true_cls, test_pred_cls)
        # self.log("val_macc", avg_test_acc)

        return loss

    def validation_epoch_end(
            self, outputs: List[Dict[str, List[torch.Tensor]]]
    ) -> None:
        test_true_cls = np.concatenate(self.test_true_cls)
        test_pred_cls = np.concatenate(self.test_pred_cls)
        test_true_seg = np.concatenate(self.test_true_seg, axis=0)
        test_pred_seg = np.concatenate(self.test_pred_seg, axis=0)

        test_ious = self.calculate_sem_IoU(test_pred_seg, test_true_seg)
        self.log("test_ious", np.mean(test_ious))

        import sklearn.metrics as metrics
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        self.log("test_acc", test_acc)
        avg_test_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        self.log("avg_test_acc", avg_test_acc)

    def calculate_sem_IoU(self, pred_np, seg_np):
        I_all = np.zeros(13)
        U_all = np.zeros(13)
        for sem_idx in range(seg_np.shape[0]):
            for sem in range(13):
                I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
                U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
                I_all[sem] += I
                U_all[sem] += U
        return I_all / U_all
        # tp = np.sum(np.logical_and(pred_np, seg_np))
        # tn = np.sum(np.logical_and(np.logical_not(pred_np), np.logical_not(seg_np)))
        # fp = np.sum(np.logical_and(pred_np, np.logical_not(seg_np)))
        # fn = np.sum(np.logical_and(np.logical_not(pred_np), seg_np))
        # iou = tp / (tp + fn + fp + 1e-7)
        # return iou


    def configure_optimizers(self):
        assert self.trainer is not None

        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.optimizer_adamw_weight_decay,  # type: ignore
        )
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=self.hparams.lr_scheduler_linear_warmup_epochs,  # type: ignore
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=self.hparams.lr_scheduler_linear_warmup_start_lr,  # type: ignore
            eta_min=self.hparams.lr_scheduler_cosine_eta_min,  # type: ignore
        )

        return [opt], [sched]

    def load_pretrained_checkpoint(self, path: str) -> None:
        print(f"Loading pretrained checkpoint from '{path}'.")

        checkpoint = extract_model_checkpoint(path)

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)  # type: ignore
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    def on_train_epoch_start(self) -> None:
        if self.trainer.current_epoch == self.hparams.encoder_unfreeze_epoch:  # type: ignore
            self.encoder.requires_grad_(True)
            print("Unfreeze encoder")

