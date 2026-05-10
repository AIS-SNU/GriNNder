"""Training engine with SSO orchestration."""


def __getattr__(name):
    if name == "Trainer":
        from grinnder.engine.trainer import Trainer
        return Trainer
    if name == "StreamManager":
        from grinnder.engine.streams import StreamManager
        return StreamManager
    raise AttributeError(f"module 'grinnder.engine' has no attribute {name!r}")


__all__ = ["Trainer", "StreamManager"]
