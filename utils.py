import graphviz

from micrograd.engine import Value


def trace(root) -> tuple[set[Value, set[Value]]]:
    nodes, edges = set(), set()
    def build(n: Value):
        nodes.add(n)
        for child in n._prev:
            edges.add((child, n))
            build(child)
    build(root)
    return nodes, edges
            
def print_tree(root):
    nodes, edges = trace(root)
    print("========= NODES ===========")
    for ind, val in enumerate(nodes):
        print(f"{ind} - {val}")
    print(" ========== EDGES =========")
    for ind, (t, h) in enumerate(edges):
        print(f"{ind} - ({t.name}, {h.name})")
      
def draw_computation_graph(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph('computation_graph', comment='Graph of Computations', graph_attr={'rankdir': 'LR'}, format='SVG')
    nodes, edges = trace(root)
    for n in nodes:
        dot.node(name=str(id(n)), label=f"{{{n.name} | data={n.data:.4f} | grad={n.grad:.4f} }}", shape='record')    
        if n._op:
            dot.node(name=str(id(n))+n._op, label=n._op)
            dot.edge(tail_name=str(id(n))+n._op, head_name=str(id(n)))
    for tail, head in edges:
        dot.edge(tail_name=str(id(tail)), head_name=str(id(head)) + head._op)
    return dot