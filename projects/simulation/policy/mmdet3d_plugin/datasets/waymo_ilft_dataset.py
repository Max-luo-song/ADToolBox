# Modified by ADToolbox contributors, 2026: standalone simulation and scene conversion support.

from mmdet.datasets import DATASETS

from mmdet3d_plugin.datasets.navsim_openscene_nuplan import (
    NavSimOpenSceneE2E,
)
from mmdet3d_plugin.datasets.navsim_openscene_finetuning import (
    NavSimOpenSceneE2EFineTune,
)


class _SkipPDM:
    def load_pdm_infos(self):
        self.metric_cache_dict = {}
        self.pdm_score_dict = {}
        self.pdm_cache_dict = {}
        self.pdm_dict = {}

    def get_pdm_score_info(self, input_dict, index, info=None):
        return input_dict


@DATASETS.register_module()
class WaymoOpenSceneE2EFineTune(
    _SkipPDM,
    NavSimOpenSceneE2EFineTune,
):
    pass


@DATASETS.register_module()
class WaymoOpenSceneE2E(
    _SkipPDM,
    NavSimOpenSceneE2E,
):
    pass
