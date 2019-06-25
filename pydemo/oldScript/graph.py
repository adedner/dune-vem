import metis
def partition(grid,nparts,**opts):
    graph = []
    indexSet = grid.indexSet
    for e in grid.elements:
        edges = (indexSet.index(i.outside) for i in grid.intersections(e) if i.outside)
        graph += [ tuple(edges) ]
    (edgecuts, parts) = metis.part_graph(graph, nparts=nparts,recursive=False,**opts)
    return parts
