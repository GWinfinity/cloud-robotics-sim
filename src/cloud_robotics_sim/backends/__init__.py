"""Concrete backend implementations for cloud_robotics_sim."""

__all__ = ["GenesisBackend", "MTLambdaBackend"]


def __getattr__(name: str):
    if name == "GenesisBackend":
        from cloud_robotics_sim.backends.genesis_backend import GenesisBackend

        return GenesisBackend
    if name == "MTLambdaBackend":
        from cloud_robotics_sim.backends.mt_lambda_backend import MTLambdaBackend

        return MTLambdaBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
