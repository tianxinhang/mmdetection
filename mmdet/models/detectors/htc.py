from ..builder import DETECTORS
from .cascade_rcnn import CascadeRCNN


@DETECTORS.register_module()
class HybridTaskCascade(CascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs):
        super(HybridTaskCascade, self).__init__(**kwargs)

    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic

    def simple_test_2(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)
        proposal_list = None
        if proposals is not None:   # 我的方法就在这了。htc在预测阶段没有用semantic_pred。而我需要把这个利用起来。
            proposal_list = proposals
        tmp_rpn_head = self.rpn_head #惊险
        return self.roi_head.simple_test_2(x, proposal_list, img_metas, tmp_rpn_head,rescale=rescale)
