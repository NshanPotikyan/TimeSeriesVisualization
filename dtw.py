import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class DTW:
    
    def __init__(self, s1, s2):
        # converting the series into numpy arrays
        if not isinstance(s1, np.ndarray):
            s1, s2 = np.array(s1), np.array(s2)

        self.s1 = s1
        self.s2 = s2
        self.cost_matrix = self.get_cost_matrix(self.s1,
                                                self.s2)

    @staticmethod
    def get_cost_matrix(s1, s2):
        """
        Calculates the cost matrix using dynamic time warping
        for the given series
        :param s1: series 1
        :param s2: series 2
        :return: cost matrix
        """
        n = len(s1)
        m = len(s2)

        dtw_out = np.zeros(shape=(n+1, m+1))

        dtw_out[0, 1:] = np.inf
        dtw_out[1:, 0] = np.inf

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_out[i, j] = cost + min(dtw_out[i-1, j],
                                           dtw_out[i, j-1],
                                           dtw_out[i-1, j-1])
        return dtw_out[1:, 1:]

    def get_path(self):
        """
        Reconstructs the path from the cost matrix
        :return: list of indices for the closest points of the two series
        """
        cost_matrix = self.cost_matrix
        x_len = cost_matrix.shape[0]
        y_len = cost_matrix.shape[1] 

        path = [[x_len-1, y_len-1]]
        i = x_len - 1
        j = y_len - 1
        while i > 0 and j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                if cost_matrix[i-1, j] == min(cost_matrix[i-1, j-1],
                                              cost_matrix[i-1, j],
                                              cost_matrix[i, j-1]):
                    i -= 1
                elif cost_matrix[i, j-1] == min(cost_matrix[i-1, j-1],
                                                cost_matrix[i-1, j],
                                                cost_matrix[i, j-1]):
                    j -= 1
                else:
                    i -= 1
                    j -= 1
            path.append([i, j])
        path.append([0, 0])

        return path
        
    def get_distance(self):
        """
        Returns the lower right element of the cost matrix
        which is the distance between the 2 series
        :return:
        """
        return self.cost_matrix[-1, -1]
    
    def plot(self):
        """
        Plots the two time series and marks their alignment
        obtained with DTW
        :return:
        """
        path = self.get_path()
        s1 = self.s1
        s2 = self.s2
        n1 = len(s1)
        n2 = len(s2)
        title = 'DTW for TSC'
        s1_name = 'series1'
        s2_name = 'series2'

        if n2 > n1:
            n = n2
            s1, s2 = s2, s1  # s1 is the longest
            n1, n2 = n2, n1
            s1_name, s2_name = s2_name, s1_name
            i, j = 1, 0
        else:
            n = n1
            i, j = 0, 1

        x_shift = 6
        y_shift = 2

        plt.plot(np.arange(n)[:n1] + x_shift, s1 + y_shift, label=s1_name)
        plt.plot(np.arange(n)[:n2], s2, label=s2_name)

        for step in range(1, len(path) + 1):
            x1_x2 = [path[-step][i] + x_shift, path[-step][j]]
            y1_y2 = [s1[path[-step][i]] + y_shift, s2[x1_x2[1]]]
            # drawing a line from (x1, y1) to (x2, y2)
            plt.plot(x1_x2, y1_y2, c='k', linestyle=':')
        
        plt.legend()
        plt.xlabel('Index')
        plt.ylabel('Data')
        plt.title(title)
        plt.show()

    def animation_plot(self):
        """
        Plots the two time series and marks their alignment
        obtained with DTW as an animation with play/stop button
        :return:
        """
        path = self.get_path()
        path.reverse()
        path_len = len(path)
        s1 = self.s1
        s2 = self.s2
        n1 = len(s1)
        n2 = len(s2)
        title = 'Time Series Alignment with DTW'
        s1_name = 'Series 1'
        s2_name = 'Series 2'

        if n2 > n1:
            n = n2
            s1, s2 = s2, s1  # s1 is the longest
            n1, n2 = n2, n1
            s1_name, s2_name = s2_name, s1_name
            i, j = 1, 0
        else:
            n = n1
            i, j = 0, 1

        x_shift = 6
        y_shift = 2

        fig = go.Figure(
            data=[go.Scatter(x=np.arange(n)[:n1] + x_shift,
                             y=s1 + y_shift,
                             showlegend=False),
                  go.Scatter(x=np.arange(n)[:n2],
                             y=s2,
                             showlegend=False),
                  go.Scatter(x=np.arange(n)[:n1] + x_shift,       # adding the same graphs again for
                             y=s1 + y_shift,                       # animation purposes
                             name=s1_name,
                             line=dict(color="red"),
                             customdata=s1,
                             hovertemplate='<i>Value</i>: %{customdata:.2f}'),
                  go.Scatter(x=np.arange(n)[:n2],
                             y=s2,
                             name=s2_name,
                             line=dict(color="blue"),
                             hovertemplate='<i>Value</i>: %{y:.2f}')],
            layout=go.Layout(xaxis=dict(range=[0, 100], autorange=False, zeroline=False),
                             yaxis=dict(range=[min(s2), max(s1)+y_shift], autorange=False, zeroline=False),
                             updatemenus=[dict(type="buttons",
                                               buttons=[dict(label="Play",
                                                             method="animate",
                                                             args=[None,
                                                                   {"frame": {"duration": 100, "redraw": False},
                                                                    "transition": {"duration": 300,
                                                                                   "easing":"quadratic-in-out"}}]),
                                                        dict(label="Stop",
                                                             method="animate",
                                                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                            "mode": "immediate",
                                                                            "transition": {"duration": 0}}])])]),
            frames=[go.Frame(
                data=[go.Scatter(x=[path[step][i] + x_shift, path[step][j]],
                                 y=[s1[path[step][i]] + y_shift, s2[path[step][j]]],
                                 mode='lines',
                                 line=dict(color="black"))]) for step in range(path_len)])

        fig.update_layout(title_text=title,
                          xaxis_rangeslider_visible=True)

        fig.update_xaxes(title_text='Time Index')
        fig.update_yaxes(title_text='Data', showticklabels=False)

        fig.show()
