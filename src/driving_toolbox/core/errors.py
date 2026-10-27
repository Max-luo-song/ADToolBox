class ToolError(Exception):
    code = "tool_error"


class InputError(ToolError):
    code = "invalid_input"


class DependencyError(ToolError):
    code = "missing_dependency"


class BackendError(ToolError):
    code = "backend_failed"
