"""graph — MDP deterministico DoorKey con cache indicizzata."""
import importlib as _il

__all__ = ["build_mdp", "load_or_build", "get_paths", "render_map", "load_or_train_qstates", "get_qstates_paths",
           "load_or_extract_states", "get_states_paths"]

def __getattr__(name):
    if name in ("build_mdp", "load_or_build", "get_paths", "render_map"):
        mod = _il.import_module(".mdp_graph", __name__)
        return getattr(mod, name)
    if name in ("load_or_train_qstates", "get_qstates_paths"):
        mod = _il.import_module(".qlearning_states", __name__)
        # qlearning_states espone load_or_train / get_qstates_paths
        if name == "load_or_train_qstates":
            return getattr(mod, "load_or_train")
        return getattr(mod, name)
    if name in ("load_or_extract_states", "get_states_paths"):
        mod = _il.import_module(".doorkey_states", __name__)
        # doorkey_states espone load_or_extract / get_states_paths
        if name == "load_or_extract_states":
            return getattr(mod, "load_or_extract")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
