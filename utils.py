import graphviz

from engine import Value


def trace(root) -> tuple[set[Value, set[Value]]]:
    nodes, edges = set(), set()
    def build(n: Value):
        nodes.add(n)
        for child in n._prev:
            edges.add((child.name, n.name))
            build(child)
    build(root)
    return nodes, edges
        
            
def print_tree(root):
    nodes, edges = trace(root)
    print("========= NODES ===========")
    for ind, val in enumerate(nodes):
        print(f"{ind} - {val}")
    print(" ========== EDGES =========")
    for ind, val in enumerate(edges):
        print(f"{ind} - {val}")
      
def draw_computation_graph(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph('computation_graph', comment='Graph of Computations', graph_attr={'rankdir': 'LR'}, format='SVG')
    nodes, edges = trace(root)
    for n in nodes:
        dot.node(name=n.name, label=f"{{{n.name} | data={n.data:.4f} | tag={n.tag} }}", shape='record')    
        if n._op:
            dot.node(name=n.name+"_op", label=n._op, )
            dot.edge(tail_name=n.name+"_op", head_name=n.name)
    for tail, head in edges:
        dot.edge(tail_name=tail, head_name=head+"_op")
    return dot
        
