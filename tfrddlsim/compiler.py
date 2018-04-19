class Compiler(object):

    def __init__(self, rddl, graph):
        self._rddl = rddl
        self._graph = graph

    def _build_object_table(self):
        types = self._rddl.domain.types
        objects = dict(self._rddl.non_fluents.objects)
        self._object_table = dict()
        for name, value in self._rddl.domain.types:
            if value == 'object':
                objs = objects[name]
                idx = { obj: i for i, obj in enumerate(objs) }
                self._object_table[name] = { 'size': len(objs), 'idx': idx }
