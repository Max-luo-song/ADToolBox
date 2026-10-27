"""支持 `python -m scene_editing` 启动 MCP 服务器。"""

from .server import main_sync

if __name__ == "__main__":
    main_sync()
