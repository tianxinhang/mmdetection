import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms

from ..builder import HEADS
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin
import torch.nn.functional as F

@HEADS.register_module()
class RPNHead(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(
            1, in_channels, background_label=0, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    
    
    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        return dets[:cfg.nms_post]
    
    
    def _get_bboxes_single_2(self,
                             cls_scores,
                             bbox_preds,
                             mlvl_anchors,
                             img_shape,
                             scale_factor,
                             semantic_num_classes,
                             semantic_pred_single=None,  # shape(1,H,W) >> 实际shape(H,W)，因为1会被退化掉
                             cfg = None,
                             rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
#             H = rpn_cls_score.size()[-2]
#             W = rpn_cls_score.size()[-1]

#             mask_1 = torch.ne(semantic_pred_single, 0)
#             mask_2 = torch.lt(semantic_pred_single, 92)
#             mask_3 = torch.eq(mask_1, mask_2).type(torch.float32)
#             mask_3 = mask_3.sigmoid()
#             mask_3 = mask_3.unsqueeze(0)
#             mask_3 = mask_3.unsqueeze(0)
#             mask_3 = F.interpolate(mask_3, size=(H,W), mode='nearest')
#             mask_3 = mask_3.squeeze()

            if self.use_sigmoid_cls:
                  # 3*w*h
#                 scores = scores + mask_3
                scores = scores.permute(1, 2, 0)
                scores = scores.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(  # 见delta2bbox，返回的shape N*4,不同的scale被凭借在同一维度
            anchors, rpn_bbox_pred, max_shape=img_shape)
        centers = centers.type(torch.long)
        ids = torch.cat(level_ids)# 用来记录每个proposal所在的层数

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]



        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        return dets[:cfg.nms_post]

    def _get_bboxes_single_5(self,                          #v5.0
                             cls_scores,
                             bbox_preds,
                             mlvl_anchors,
                             img_shape,
                             scale_factor,
                             gt_semantic_seg_single=None,  # shape(1,H,W) >> 实际shape(H,W)，因为1会被退化掉
                             cfg = None,
                             rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []

        # for i in range(cls_score.size()[0]):                       #单层
        #     H = cls_score[i].size()[-2]
        #     W = cls_score[i].size()[-1]
        #     mask_0 = gt_semantic_segs[i].squeeze()
        #     mask_1 = torch.ne(mask_0, 0)
        #     mask_2 = torch.lt(mask_0, 92)
        #     mask_3 = torch.eq(mask_1, mask_2).type(torch.float32)
        #     mask_3 = mask_3.unsqueeze(0)
        #     mask_3 = mask_3.unsqueeze(0)
        #     mask_3 = F.interpolate(mask_3, size=(H, W), mode='nearest')
        #     mask_3 = mask_3.squeeze()
        #     # label = labels[i]
        #     mask_3 = mask_3.view(-1)
        #     # print("mask_3: ",mask_3.size())
        #     # print("label: ",label.size())
        #     mask_3= mask_3.type(torch.long)
        #     mask_3 = mask_3.view(mask_3.size()[0], -1)
        #     mask_3 = mask_3.expand(mask_3.size()[0], 3)
        #     mask_3 = mask_3.reshape(-1)
        #     labels[i] = labels[i]* mask_3

        for idx in range(len(cls_scores)):              #单图
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
#             H = rpn_cls_score.size()[-2]
#             W = rpn_cls_score.size()[-1]

#             mask_0 = gt_semantic_seg_single.squeeze()
#             mask_1 = torch.ne(mask_0, 0)
#             mask_2 = torch.lt(mask_0, 92)
#             mask_3 = torch.eq(mask_1, mask_2).type(torch.float32)
#             mask_3 = mask_3.unsqueeze(0)
#             mask_3 = mask_3.unsqueeze(0)
#             mask_3 = F.interpolate(mask_3, size=(H, W), mode='nearest')
#             mask_3 = mask_3.squeeze()
#             mask_3 = mask_3.sigmoid()


            # mask_1 = torch.ne(semantic_pred_single, 0)
            # mask_2 = torch.lt(semantic_pred_single, 92)
            # mask_3 = torch.eq(mask_1, mask_2).type(torch.float32)
            # mask_3 = mask_3.sigmoid()
            # mask_3 = mask_3.unsqueeze(0)
            # mask_3 = mask_3.unsqueeze(0)
            # mask_3 = F.interpolate(mask_3, size=(H,W), mode='nearest')
            # mask_3 = mask_3.squeeze()

            if self.use_sigmoid_cls:
                  # 3*w*h
#                 scores = scores + mask_3
                scores = scores.permute(1, 2, 0)
                scores = scores.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(  # 见delta2bbox，返回的shape N*4,不同的scale被凭借在同一维度
            anchors, rpn_bbox_pred, max_shape=img_shape)
        centers = centers.type(torch.long)
        ids = torch.cat(level_ids)# 用来记录每个proposal所在的层数

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]



        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        return dets[:cfg.nms_post]
    
#     def _get_bboxes_single_2(self,
#                              cls_scores,
#                              bbox_preds,
#                              mlvl_anchors,
#                              img_shape,
#                              scale_factor,
#                              semantic_num_classes,
#                              semantic_pred_single=None,  # shape(1,H,W) >> 实际shape(H,W)，因为1会被退化掉
#                              cfg = None,
#                              rescale=False):
#         """Transform outputs for a single batch item into bbox predictions.

#         Args:
#             cls_scores (list[Tensor]): Box scores for each scale level
#                 Has shape (num_anchors * num_classes, H, W).
#             bbox_preds (list[Tensor]): Box energies / deltas for each scale
#                 level with shape (num_anchors * 4, H, W).
#             mlvl_anchors (list[Tensor]): Box reference for each scale level
#                 with shape (num_total_anchors, 4).
#             img_shape (tuple[int]): Shape of the input image,
#                 (height, width, 3).
#             scale_factor (ndarray): Scale factor of the image arange as
#                 (w_scale, h_scale, w_scale, h_scale).
#             cfg (mmcv.Config): Test / postprocessing configuration,
#                 if None, test_cfg would be used.
#             rescale (bool): If True, return boxes in original image space.

#         Returns:
#             Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
#                 are bounding box positions (tl_x, tl_y, br_x, br_y) and the
#                 5-th column is a score between 0 and 1.
#         """
#         cfg = self.test_cfg if cfg is None else cfg
#         # bboxes from different level should be independent during NMS,
#         # level_ids are used as labels for batched NMS to separate them
#         level_ids = []
#         mlvl_scores = []
#         mlvl_bbox_preds = []
#         mlvl_valid_anchors = []
#         for idx in range(len(cls_scores)):
#             rpn_cls_score = cls_scores[idx]
#             rpn_bbox_pred = bbox_preds[idx]
#             assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
#             H = rpn_cls_score.size()[-2]
#             W = rpn_cls_score.size()[-1]

#             mask_1 = torch.ne(semantic_pred_single, 0)
#             mask_2 = torch.lt(semantic_pred_single, 92)
#             mask_3 = torch.eq(mask_1, mask_2).type(torch.float32)
# #             mask_3 = mask_3.sigmoid()
#             mask_3 = mask_3.unsqueeze(0)
#             mask_3 = mask_3.unsqueeze(0)
#             mask_3 = F.interpolate(mask_3, size=(H,W), mode='nearest')
#             mask_3 = mask_3.squeeze()
#             mask_3 = mask_3.sigmoid()

#             if self.use_sigmoid_cls:
#                 scores = rpn_cls_score.sigmoid()  # 3*w*h
#                 scores = scores + mask_3
#                 scores = scores.permute(1, 2, 0)
#                 scores = scores.reshape(-1)
#             else:
#                 rpn_cls_score = rpn_cls_score.reshape(-1, 2)
#                 # we set FG labels to [0, num_class-1] and BG label to
#                 # num_class in other heads since mmdet v2.0, However we
#                 # keep BG label as 0 and FG label as 1 in rpn head
#                 scores = rpn_cls_score.softmax(dim=1)[:, 1]
#             rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
#             anchors = mlvl_anchors[idx]
#             if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
#                 # sort is faster than topk
#                 # _, topk_inds = scores.topk(cfg.nms_pre)
#                 ranked_scores, rank_inds = scores.sort(descending=True)
#                 topk_inds = rank_inds[:cfg.nms_pre]
#                 scores = ranked_scores[:cfg.nms_pre]
#                 rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
#                 anchors = anchors[topk_inds, :]
#             mlvl_scores.append(scores)
#             mlvl_bbox_preds.append(rpn_bbox_pred)
#             mlvl_valid_anchors.append(anchors)
#             level_ids.append(
#                 scores.new_full((scores.size(0), ), idx, dtype=torch.long))

#         scores = torch.cat(mlvl_scores)
#         anchors = torch.cat(mlvl_valid_anchors)
#         rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
#         centers, proposals = self.bbox_coder.decode_2(  # 见delta2bbox，返回的shape N*4,不同的scale被凭借在同一维度
#             anchors, rpn_bbox_pred, max_shape=img_shape)
#         centers = centers.type(torch.long)
#         ids = torch.cat(level_ids)# 用来记录每个proposal所在的层数

#         if cfg.min_bbox_size > 0:
#             w = proposals[:, 2] - proposals[:, 0]
#             h = proposals[:, 3] - proposals[:, 1]
#             valid_inds = torch.nonzero(
#                 (w >= cfg.min_bbox_size)
#                 & (h >= cfg.min_bbox_size),
#                 as_tuple=False).squeeze()
#             if valid_inds.sum().item() != len(proposals):
#                 proposals = proposals[valid_inds, :]
#                 scores = scores[valid_inds]
#                 ids = ids[valid_inds]



#         # TODO: remove the hard coded nms type
#         nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
#         dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
#         return dets[:cfg.nms_post]

#     def _get_bboxes_single_5(self,                          #v5.0
#                              cls_scores,
#                              bbox_preds,
#                              mlvl_anchors,
#                              img_shape,
#                              scale_factor,
#                              gt_semantic_seg_single=None,  # shape(1,H,W) >> 实际shape(H,W)，因为1会被退化掉
#                              cfg = None,
#                              rescale=False):
#         """Transform outputs for a single batch item into bbox predictions.

#         Args:
#             cls_scores (list[Tensor]): Box scores for each scale level
#                 Has shape (num_anchors * num_classes, H, W).
#             bbox_preds (list[Tensor]): Box energies / deltas for each scale
#                 level with shape (num_anchors * 4, H, W).
#             mlvl_anchors (list[Tensor]): Box reference for each scale level
#                 with shape (num_total_anchors, 4).
#             img_shape (tuple[int]): Shape of the input image,
#                 (height, width, 3).
#             scale_factor (ndarray): Scale factor of the image arange as
#                 (w_scale, h_scale, w_scale, h_scale).
#             cfg (mmcv.Config): Test / postprocessing configuration,
#                 if None, test_cfg would be used.
#             rescale (bool): If True, return boxes in original image space.

#         Returns:
#             Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
#                 are bounding box positions (tl_x, tl_y, br_x, br_y) and the
#                 5-th column is a score between 0 and 1.
#         """
#         cfg = self.test_cfg if cfg is None else cfg
#         # bboxes from different level should be independent during NMS,
#         # level_ids are used as labels for batched NMS to separate them
#         level_ids = []
#         mlvl_scores = []
#         mlvl_bbox_preds = []
#         mlvl_valid_anchors = []

      

#         for idx in range(len(cls_scores)):              #单图
#             rpn_cls_score = cls_scores[idx]
#             rpn_bbox_pred = bbox_preds[idx]
#             assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
#             H = rpn_cls_score.size()[-2]
#             W = rpn_cls_score.size()[-1]

#             mask_0 = gt_semantic_seg_single.squeeze()
#             mask_1 = torch.ne(mask_0, 0)
#             mask_2 = torch.lt(mask_0, 92)
#             mask_3 = torch.eq(mask_1, mask_2).type(torch.float32)
#             mask_3 = mask_3.unsqueeze(0)
#             mask_3 = mask_3.unsqueeze(0)
#             mask_3 = F.interpolate(mask_3, size=(H, W), mode='nearest')
#             mask_3 = mask_3.squeeze()
#             mask_3 = mask_3.sigmoid()


#             # mask_1 = torch.ne(semantic_pred_single, 0)
#             # mask_2 = torch.lt(semantic_pred_single, 92)
#             # mask_3 = torch.eq(mask_1, mask_2).type(torch.float32)
#             # mask_3 = mask_3.sigmoid()
#             # mask_3 = mask_3.unsqueeze(0)
#             # mask_3 = mask_3.unsqueeze(0)
#             # mask_3 = F.interpolate(mask_3, size=(H,W), mode='nearest')
#             # mask_3 = mask_3.squeeze()

#             if self.use_sigmoid_cls:
#                 scores = rpn_cls_score.sigmoid()  # 3*w*h
#                 scores = scores + mask_3
#                 scores = scores.permute(1, 2, 0)
#                 scores = scores.reshape(-1)
#             else:
#                 rpn_cls_score = rpn_cls_score.reshape(-1, 2)
#                 # we set FG labels to [0, num_class-1] and BG label to
#                 # num_class in other heads since mmdet v2.0, However we
#                 # keep BG label as 0 and FG label as 1 in rpn head
#                 scores = rpn_cls_score.softmax(dim=1)[:, 1]
#             rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
#             anchors = mlvl_anchors[idx]
#             if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
#                 # sort is faster than topk
#                 # _, topk_inds = scores.topk(cfg.nms_pre)
#                 ranked_scores, rank_inds = scores.sort(descending=True)
#                 topk_inds = rank_inds[:cfg.nms_pre]
#                 scores = ranked_scores[:cfg.nms_pre]
#                 rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
#                 anchors = anchors[topk_inds, :]
#             mlvl_scores.append(scores)
#             mlvl_bbox_preds.append(rpn_bbox_pred)
#             mlvl_valid_anchors.append(anchors)
#             level_ids.append(
#                 scores.new_full((scores.size(0), ), idx, dtype=torch.long))

#         scores = torch.cat(mlvl_scores)
#         anchors = torch.cat(mlvl_valid_anchors)
#         rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
#         centers, proposals = self.bbox_coder.decode_2(  # 见delta2bbox，返回的shape N*4,不同的scale被凭借在同一维度
#             anchors, rpn_bbox_pred, max_shape=img_shape)
#         centers = centers.type(torch.long)
#         ids = torch.cat(level_ids)# 用来记录每个proposal所在的层数

#         if cfg.min_bbox_size > 0:
#             w = proposals[:, 2] - proposals[:, 0]
#             h = proposals[:, 3] - proposals[:, 1]
#             valid_inds = torch.nonzero(
#                 (w >= cfg.min_bbox_size)
#                 & (h >= cfg.min_bbox_size),
#                 as_tuple=False).squeeze()
#             if valid_inds.sum().item() != len(proposals):
#                 proposals = proposals[valid_inds, :]
#                 scores = scores[valid_inds]
#                 ids = ids[valid_inds]



#         # TODO: remove the hard coded nms type
#         nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
#         dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
#         return dets[:cfg.nms_post]

    
    
    
#     def _get_bboxes_single(self,
#                            cls_scores,
#                            bbox_preds,
#                            mlvl_anchors,
#                            img_shape,
#                            scale_factor,
#                            cfg,
#                            rescale=False):
#         """Transform outputs for a single batch item into bbox predictions.

#         Args:
#             cls_scores (list[Tensor]): Box scores for each scale level
#                 Has shape (num_anchors * num_classes, H, W).
#             bbox_preds (list[Tensor]): Box energies / deltas for each scale
#                 level with shape (num_anchors * 4, H, W).
#             mlvl_anchors (list[Tensor]): Box reference for each scale level
#                 with shape (num_total_anchors, 4).
#             img_shape (tuple[int]): Shape of the input image,
#                 (height, width, 3).
#             scale_factor (ndarray): Scale factor of the image arange as
#                 (w_scale, h_scale, w_scale, h_scale).
#             cfg (mmcv.Config): Test / postprocessing configuration,
#                 if None, test_cfg would be used.
#             rescale (bool): If True, return boxes in original image space.

#         Returns:
#             Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
#                 are bounding box positions (tl_x, tl_y, br_x, br_y) and the
#                 5-th column is a score between 0 and 1.
#         """
#         cfg = self.test_cfg if cfg is None else cfg
#         # bboxes from different level should be independent during NMS,
#         # level_ids are used as labels for batched NMS to separate them
#         level_ids = []
#         mlvl_scores = []
#         mlvl_bbox_preds = []
#         mlvl_valid_anchors = []
#         for idx in range(len(cls_scores)):
#             rpn_cls_score = cls_scores[idx]
#             rpn_bbox_pred = bbox_preds[idx]
#             assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
#             rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
#             if self.use_sigmoid_cls:
#                 rpn_cls_score = rpn_cls_score.reshape(-1)
#                 scores = rpn_cls_score.sigmoid()
#             else:
#                 rpn_cls_score = rpn_cls_score.reshape(-1, 2)
#                 # we set FG labels to [0, num_class-1] and BG label to
#                 # num_class in other heads since mmdet v2.0, However we
#                 # keep BG label as 0 and FG label as 1 in rpn head
#                 scores = rpn_cls_score.softmax(dim=1)[:, 1]
#             rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
#             anchors = mlvl_anchors[idx]
#             if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
#                 # sort is faster than topk
#                 # _, topk_inds = scores.topk(cfg.nms_pre)
#                 ranked_scores, rank_inds = scores.sort(descending=True)
#                 topk_inds = rank_inds[:cfg.nms_pre]
#                 scores = ranked_scores[:cfg.nms_pre]
#                 rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
#                 anchors = anchors[topk_inds, :]
#             mlvl_scores.append(scores)
#             mlvl_bbox_preds.append(rpn_bbox_pred)
#             mlvl_valid_anchors.append(anchors)
#             level_ids.append(
#                 scores.new_full((scores.size(0), ), idx, dtype=torch.long))

#         scores = torch.cat(mlvl_scores)
#         anchors = torch.cat(mlvl_valid_anchors)
#         rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
#         proposals = self.bbox_coder.decode(
#             anchors, rpn_bbox_pred, max_shape=img_shape)
#         ids = torch.cat(level_ids)

#         if cfg.min_bbox_size > 0:
#             w = proposals[:, 2] - proposals[:, 0]
#             h = proposals[:, 3] - proposals[:, 1]
#             valid_inds = torch.nonzero(
#                 (w >= cfg.min_bbox_size)
#                 & (h >= cfg.min_bbox_size),
#                 as_tuple=False).squeeze()
#             if valid_inds.sum().item() != len(proposals):
#                 proposals = proposals[valid_inds, :]
#                 scores = scores[valid_inds]
#                 ids = ids[valid_inds]

#         # TODO: remove the hard coded nms type
#         nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
#         dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
#         return dets[:cfg.nms_post]
