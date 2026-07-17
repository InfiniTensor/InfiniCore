from infinicore.lib import _infinicore


class Graph:
    """
    Python wrapper around a InfiniCore Graph instance.
    """

    def __init__(self, graph: _infinicore.Graph):
        if not isinstance(graph, _infinicore.Graph):
            raise TypeError("Expected _infinicore.Graph")
        self._graph = graph

    def run(self):
        return self._graph.run()

    def has_device_exec(self) -> bool:
        return self._graph.has_device_exec()

    def device_segment_count(self) -> int:
        return int(self._graph.device_segment_count())

    def device_graph_log(self) -> str:
        return self._graph.device_graph_log()

    def last_replay_used_device(self) -> bool:
        return self._graph.last_replay_used_device()

    def replay_device_ok(self) -> int:
        return self._graph.replay_device_ok()

    def replay_op_list_fallback(self) -> int:
        return self._graph.replay_op_list_fallback()

    def __repr__(self):
        return f"<Graph wrapper of {self._graph!r}>"
