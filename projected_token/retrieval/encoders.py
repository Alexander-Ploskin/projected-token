from __future__ import annotations

from typing import Any


def build_encoder(config: dict[str, Any]):
    name = config.get("name") or config.get("type") or config.get("encoder")
    kwargs = dict(config.get("kwargs", {}))
    if "class" in config:
        from projected_token.config import instantiate
        return instantiate(config, kind="encoder")
    if name == "oscar":
        from projected_token.encoders import OscarEncoder
        return OscarEncoder(**kwargs)
    if name == "salesforce":
        from projected_token.encoders import SalesforceEncoder
        return SalesforceEncoder(**kwargs)
    if name in {"bge", "bge_base_en_v15"}:
        from projected_token.encoders import BGEEncoder
        if "model_name_or_path" not in kwargs:
            kwargs["model_name_or_path"] = "BAAI/bge-base-en-v1.5"
        return BGEEncoder(**kwargs)
    if name == "oscar_projector":
        from projected_token.encoders import OscarProjectorEncoder
        return OscarProjectorEncoder(**kwargs)
    raise ValueError(f"Unknown encoder: {name}")
