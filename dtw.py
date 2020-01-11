import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import librosa
import json


class DTW:
    
    def __init__(self, s1, s2, audio_files=False):

        if audio_files:
            # s1, s2 are the audio file names .wav, .mp3 etc.

            s1, s2 = self.load_audio(file_name=s1), self.load_audio(file_name=s2)

            # decreasing the dimensionality of the signal
            # by moving average smoothing
            if len(s1) > 1000:
                s1 = self.moving_average(input_series=s1)

            if len(s2) > 1000:
                s2 = self.moving_average(input_series=s2)

        # converting the series into numpy arrays
        if not isinstance(s1, np.ndarray):
            s1, s2 = np.array(s1), np.array(s2)

        self.s1 = s1
        self.s2 = s2
        self.cost_matrix = self.get_cost_matrix(self.s1,
                                                self.s2)
        self.plot_params = None

    def load_audio(self, file_name, sr=100):

        try:
            output_series, _ = librosa.load(file_name, sr=sr)
        except ZeroDivisionError:
            sr += 100
            output_series = self.load_audio(file_name, sr=sr)

        return output_series

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
        path.reverse()

        return path
        
    def get_distance(self):
        """
        Returns the lower right element of the cost matrix
        which is the distance between the 2 series
        :return:
        """
        return self.cost_matrix[-1, -1]
    
    def plot(self, standard_graph=True, x_shift=None, y_shift=None):
        """
        Plots the two time series and marks their alignment
        obtained with DTW

        :param standard_graph: boolean
            - if False, plots an interactive graph

        :param x_shift: numeric (optional)
            - specifies the shifting margin for the longest
            series on the x axis
        :param y_shift: numeric (optional)
            - specifies the shifting margin for the longest
            series on the y axis
        :return:
        """
        s1 = self.s1
        s2 = self.s2
        n1 = len(s1)
        n2 = len(s2)

        if n2 > n1:
            n = n2           # length of the longest series
            s1, s2 = s2, s1  # s1 is the longest
            i, j = 1, 0      # index w.r.t input series
        else:
            n = n1
            i, j = 0, 1

        #  shifting one of the series
        #  (the longest, if the series differ in size)
        #  for visual purposes
        if x_shift is None:
            x_shift = 6
        if y_shift is None:
            y_shift = max(s1)

        self.plot_params = {'s1': s1,
                            's2': s2,
                            'n': n,
                            'i': i,
                            'j': j,
                            'x_shift': x_shift,
                            'y_shift': y_shift,
                            'title': 'Time Series Alignment with DTW',
                            's1_name': 'Series 1',
                            's2_name': 'Series 2',
                            'x_label': 'Time Index',
                            'y_label': 'Data'
                            }

        if standard_graph:
            self.standard_plot()
        else:
            return self.interactive_plot()

    def standard_plot(self):
        plot_params = self.plot_params
        s1, s2 = plot_params['s1'], plot_params['s2']
        n1, n2 = len(s1), len(s2)
        i, j = plot_params['i'], plot_params['j']
        x_shift, y_shift = plot_params['x_shift'], plot_params['y_shift']
        path = self.get_path()
        n = plot_params['n']

        fig = go.Figure(
            data=[go.Scatter(x=np.arange(n)[:n1] + x_shift,
                             y=s1 + y_shift,
                             name=plot_params['s1_name'],
                             line=dict(color="red"),
                             customdata=s1,                 # using the default values (without shifting) for hovering
                             hovertemplate='<i>Value</i>: %{customdata:.4f}'),
                  go.Scatter(x=np.arange(n)[:n2],
                             y=s2,
                             name=plot_params['s2_name'],
                             line=dict(color="blue"),
                             hovertemplate='<i>Value</i>: %{y:.4f}')]
        )

        for k in range(len(path)):
            fig.add_trace(
                go.Scatter(x=[path[k][i] + x_shift, path[k][j]],
                           y=[s1[path[k][i]] + y_shift, s2[path[k][j]]],
                           mode='lines',
                           line=dict(color="black", dash='dot'),
                           showlegend=False, hoverinfo="y",
                           hovertemplate='<i>Value</i>: %{y:.4f}')
                            )

        fig.update_layout(title_text=plot_params['title'],
                          xaxis_rangeslider_visible=True)

        fig.update_xaxes(title_text=plot_params['x_label'])
        fig.update_yaxes(title_text=plot_params['y_label'], showticklabels=False)

        fig.show()

    def interactive_plot(self):
        """
        Plots an interactive graph with play/stop button
        :return:
        """
        plot_params = self.plot_params
        s1, s2 = plot_params['s1'], plot_params['s2']
        n1, n2 = len(s1), len(s2)
        i, j = plot_params['i'], plot_params['j']
        x_shift, y_shift = plot_params['x_shift'], plot_params['y_shift']
        path = self.get_path()
        n = plot_params['n']

        fig = go.Figure(
            data=[go.Scatter(x=np.arange(n)[:n1] + x_shift,
                             y=s1 + y_shift,
                             showlegend=False),
                  go.Scatter(x=np.arange(n)[:n2],
                             y=s2,
                             showlegend=False),
                  go.Scatter(x=np.arange(n)[:n1] + x_shift,       # adding the same graphs again for
                             y=s1 + y_shift,                      # animation purposes
                             name=plot_params['s1_name'],
                             line=dict(color="red"),
                             customdata=s1,                 # using the default values (without shifting) for hovering
                             hovertemplate='<i>Value</i>: %{customdata:.4f}'),
                  go.Scatter(x=np.arange(n)[:n2],
                             y=s2,
                             name=plot_params['s2_name'],
                             line=dict(color="blue"),
                             hovertemplate='<i>Value</i>: %{y:.4f}')],
            layout=go.Layout(xaxis=dict(range=[0, n + x_shift], autorange=False, zeroline=False),
                             yaxis=dict(range=[min(s2), max(s1)+y_shift], autorange=False, zeroline=False),
                             updatemenus=[dict(type="buttons",
                                               buttons=[dict(label="Play",
                                                             method="animate",
                                                             args=[None,
                                                                   {"frame": {"duration": 100, "redraw": False},
                                                                    "transition": {"duration": 300,
                                                                                   "easing": "quadratic-in-out"}}]),
                                                        dict(label="Stop",
                                                             method="animate",
                                                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                            "mode": "immediate",
                                                                            "transition": {"duration": 0}}])])]),
            frames=[go.Frame(
                data=[go.Scatter(x=[path[step][i] + x_shift, path[step][j]],
                                 y=[s1[path[step][i]] + y_shift, s2[path[step][j]]],
                                 mode='lines',
                                 line=dict(color="black", dash='dot'))]) for step in range(len(path))]
        )

        fig.update_layout(title_text=plot_params['title'],
                          xaxis_rangeslider_visible=True)

        fig.update_xaxes(title_text=plot_params['x_label'])
        fig.update_yaxes(title_text=plot_params['y_label'], showticklabels=False)
        #print(fig)
        return fig.to_json()
        #fig.show()
        #graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # fig.show()

    def moving_average(self, input_series, window_size=11, stride=5):
        """
        Performs moving average smoothing
        on the given time series
        :param input_series: numpy array
            - time series
        :param window_size: int
            - the sliding window size
        :param stride: int
            - step size of the window
        :return:
        """
        # print(input_series)
        len_y = len(input_series)
        assert len_y >= window_size
        nr_filters = np.floor((len_y - window_size + 1 * stride) / stride)
        denoised = []
        for i in range(int(nr_filters)):
            denoised.append(input_series[i*stride:(window_size + i*stride)].mean())
        return np.array(denoised)

        #return graphJSON
        #return fig


    # def standard_plot(self):
    #     """
    #     Plot a standard plot with matplotlib
    #     :return:
    #     """
    #     plot_params = self.plot_params
    #     s1, s2 = plot_params['s1'], plot_params['s2']
    #     i, j = plot_params['i'], plot_params['j']
    #     x_shift, y_shift = plot_params['x_shift'], plot_params['y_shift']
    #     path = self.get_path()
    #     n = plot_params['n']
    #
    #     plt.figure(figsize=(10, 8))
    #     plt.plot(np.arange(n)[:len(s1)] + x_shift, s1 + y_shift, label=plot_params['s1_name'])
    #     plt.plot(np.arange(n)[:len(s2)], s2, label=plot_params['s2_name'])
    #
    #     for step in range(len(path)):
    #         x1_x2 = [path[step][i] + x_shift, path[step][j]]
    #         y1_y2 = [s1[path[step][i]] + y_shift, s2[x1_x2[j]]]
    #         # drawing a line from (x1, y1) to (x2, y2)
    #         plt.plot(x1_x2, y1_y2, c='k', linestyle=':')
    #
    #     plt.legend()
    #     plt.xlabel(plot_params['x_label'])
    #     plt.ylabel(plot_params['y_label'])
    #     plt.title(plot_params['title'])
    #     # we don't need to show the y axis values
    #     # since one of the series is shifted
    #     # for visual purposes
    #     plt.yticks(ticks=[])
    #
    #     plt.show()
