"""节点构建器模块。

提供不同类型的节点配置构建器:
- BackgroundNodeBuilder: 背景节点
- RigidNodeBuilder: 刚体节点(车辆、路障等)
- SMPLNodeBuilder: 人体模型节点(SMPL)
"""

from .base import NodeBuilder
from .background import BackgroundNodeBuilder
from .rigid import RigidNodeBuilder
from .smpl import SMPLNodeBuilder

__all__ = [
    "NodeBuilder",
    "BackgroundNodeBuilder",
    "RigidNodeBuilder",
    "SMPLNodeBuilder",
]
