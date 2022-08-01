from abc import ABC, abstractmethod

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

    def savePlot(self):
        self.getProjection()
        print("Projections done.")
        if self.has_feature:
            fig = px.scatter(self.projections, x=0, y=1, color=self.embeddings.feature)
        else:
            fig = px.scatter(self.projections, x=0, y=1)
        print("Plot figure created, saving", end="... ")
        fig.write_image(self.location)
        print("Done.")