from abc import ABC, abstractmethod

#import networkx as nx
import plotly.express as px

class AbstractVisTest(ABC):

    def __init__(self, embeddings, has_feature, location):
        self.embeddings = embeddings
        self.has_feature = has_feature
        self.X = self.embeddings.iloc[:, :-has_feature]
        self.location = location
        self.projections = None
    
    @abstractmethod
    def getProjection(self):
        """
        :param self:
        :return: None
        
        Set self.projections to projections data.
        """
        pass

    def getScatterPlot(self):
        self.getProjection()
        print("Projections done.")
        if self.has_feature:
            fig = px.scatter(self.projections, x=0, y=1, color=self.embeddings.feature)
        else:
            fig = px.scatter(self.projections, x=0, y=1)
        print("Scatter plot created.")
        return fig
    
    def savePlot(self, edgeset):
        fig = self.getScatterPlot()
        self.knn(edgeset, fig)
        if self.has_feature:
            self.detectDensity(fig)
        print("Saving plot at [ {} ]".format(self.location), end="... ")
        fig.write_image(self.location)
        print("Done.")
    
    # 这是一段吉祥物代码，因为不知道写可视化效果测试的Romee同志需不需要用到。
    # 要用这段代码的话 记得把上面 import networkx as nx 取消注释
    #
    #def readGraph(self):
    #    print("Reading graph data", end="... ")
    #    G = nx.read_edgelist(self.edgeset, create_using=nx.Graph(), nodetype=None, data=[("weight", int)])
    #    print("Done.")
    #    return G
    
    def knn(self, edgeset, fig):
        score = 0
        print("KNN accuracy: {}".format(score))

    def detectDensity(self, fig):
        score = 0
        print("Density accuracy: {}".format(score))