from abc import ABC, abstractmethod
from visualizing_tests.utils import *

import plotly.express as px

class AbstractVisTest(ABC):

    def __init__(self, embeddings, has_feature, location):
        self.embeddings = embeddings
        self.has_feature = has_feature
        if has_feature:
            self.X = self.embeddings.iloc[:, :-1]
        else:
            self.X = self.embeddings.iloc[:, :]
        self.location = location
        self.projections = None
        self.duration = None
    
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
    
    def savePlot(self):
        fig = self.getScatterPlot()
        print("Saving scatter plot", end="... ")
        fig.write_image(self.location)
        print("Done.")
        print("Location: {}".format(self.location))