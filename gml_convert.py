import networkx as nx

def gml_convert(filename):
    g=nx.read_gml(filename)
    nx.write_edgelist(g,f'{filename}_edgelist.txt')

if __name__=='__main__':
    fn=input("please enter filename: ")
    gml_convert(fn)