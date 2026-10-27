"""Dependency-free validation for the JSON Schema subset used by this package."""

import math

from .errors import InputError


def validate(value, schema, path="input"):
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
    }
    kind = schema.get("type")
    if kind and (
        not isinstance(value, types[kind]) or (kind in {"integer", "number"} and isinstance(value, bool))
    ):
        raise InputError(f"{path} must be {kind}")
    if isinstance(value, float) and not math.isfinite(value):
        raise InputError(f"{path} must be finite")
    if "enum" in schema and value not in schema["enum"]:
        raise InputError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise InputError(f"{path}.{key} is required")
        for key, item in value.items():
            if key in schema.get("properties", {}):
                validate(item, schema["properties"][key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise InputError(f"{path}.{key} is not supported")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise InputError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise InputError(f"{path} has too many items")
        for i, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{i}]")
    if isinstance(value, str) and len(value.strip()) < schema.get("minLength", 0):
        raise InputError(f"{path} cannot be empty")
    if type(value) in (int, float):
        if "minimum" in schema and value < schema["minimum"]:
            raise InputError(f"{path} must be >= {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise InputError(f"{path} must be > {schema['exclusiveMinimum']}")
