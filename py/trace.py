from collections import deque, defaultdict
from .defs.samplers import SAMPLERS

class Trace:
    @classmethod
    def trace(cls, start_node_id, prompt):
        node = prompt.get(start_node_id)
        class_type = node["class_type"] if node else None
        Q = deque([(start_node_id, 0)])
        visited = {start_node_id}
        trace_tree = {start_node_id: (0, class_type)}

        while Q:
            current_node_id, distance = Q.popleft()
            node = prompt.get(current_node_id)
            if not node:
                continue

            input_fields = node.get("inputs", {})
            for value in input_fields.values():
                if isinstance(value, list) and value:
                    nid = value[0]
                    if nid not in visited:
                        node = prompt.get(nid)
                        if node:
                            class_type = node["class_type"]
                            trace_tree[nid] = (distance + 1, class_type)
                            Q.append((nid, distance + 1))
                            visited.add(nid)

        return trace_tree

    @classmethod
    def find_node_by_class_types(cls, trace_tree, class_type_set, node_id=None):
        if node_id:
            node = trace_tree.get(str(node_id))
            if node and node[1] in class_type_set:
                return node_id
        else:
            for nid, (_, class_type) in trace_tree.items():
                if class_type in class_type_set:
                    return nid
        return None

    @classmethod
    def find_sampler_node_id(cls, trace_tree):
        sampler = cls.find_node_by_class_types(
            trace_tree,
            set(SAMPLERS.keys()),
        )

        if sampler:
            return sampler

        raise ValueError("Could not find a sampler node in the trace tree.")

    @classmethod
    def filter_inputs_by_trace_tree(cls, inputs, trace_tree):
        filtered_inputs = defaultdict(list)
        for meta, input_list in inputs.items():
            for node_id, input_value in input_list:
                trace = trace_tree.get(node_id)
                if trace:
                    filtered_inputs[meta].append((node_id, input_value, trace[0]))

        for key in filtered_inputs:
            filtered_inputs[key].sort(key=lambda x: x[2])  # Sort by distance

        return filtered_inputs
