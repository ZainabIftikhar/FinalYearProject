import pickle
import threading

import numpy as np
from cv2 import cv2
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, QtCore

from ModelGeneration.neuralnetworkmodel import NeuralNetworkModel
from ui_mainwindow import Ui_MainWindow

cameraOpen = False
cameraThread = None
graphOpen = False

fig = None
ax = None


class MainWindowClass(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.start_cam.clicked.connect(self.start_camera)

        self.show()

    def update_graph(self, df):
        global fig, ax

        # Setting the positions and width for the bars
        pos = list(range(len(df['avg'])))
        width = 0.25

        # Plotting the bars
        fig, ax = plt.subplots(figsize=(10, 5))

        # Create a bar with pre_score data,
        # in position pos,
        plt.bar(pos,
                # using df['pre_score'] data,
                df['avg'],
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='#EE3224',
                # with label the first value in first_name
                label=df['trait_name'][0])

        # Create a bar with mid_score data,
        # in position pos + some width buffer,
        plt.bar([p + width for p in pos],
                # using df['mid_score'] data,
                df['min'],
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='#F78F1E',
                # with label the second value in first_name
                label=df['trait_name'][1])

        # Create a bar with post_score data,
        # in position pos + some width buffer,
        plt.bar([p + width * 2 for p in pos],
                # using df['post_score'] data,
                df['max'],
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='#FFC222',
                # with label the third value in first_name
                label=df['trait_name'][2])

        # Set the y axis label
        ax.set_ylabel('Trait Value')

        # Set the chart's title
        ax.set_title('Values of traits')

        # Set the position of the x ticks
        ax.set_xticks([p + 1.5 * width for p in pos])

        # Set the labels for the x ticks
        ax.set_xticklabels(df['trait_name'])

        # Setting the x-axis and y-axis limits
        plt.xlim(min(pos) - width, max(pos) + width * 4)
        plt.ylim([0, max(df['avg'] + df['min'] + df['max'])])

        # Adding the legend and showing the plot
        plt.legend(['Average', 'Minimum', 'Maximum'], loc='upper left')
        plt.grid()
        plt.show()

    def camera_loop(self):
        global fig, ax
        global cameraOpen

        neuralModel = NeuralNetworkModel()

        neuralModelPickle = open("ModelStorage/nnlivetest.pickle", "rb")

        neuralModelDict = pickle.load(neuralModelPickle)

        video_capture = cv2.VideoCapture(0)

        frame_count = 0
        openness_average = []
        openness_min = 1
        openness_max = 0

        extraversion_average = []
        extraversion_min = 1
        extraversion_max = 0

        neuroticism_average = []
        neuroticism_min = 1
        neuroticism_max = 0

        agreeableness_average = []
        agreeableness_min = 1
        agreeableness_max = 0

        conscientiousness_average = []
        conscientiousness_min = 1
        conscientiousness_max = 0

        while cameraOpen:
            ret, frame = video_capture.read()

            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)

            valueUpdated = False

            if frame_count % 15 is 0:
                temp_openness, temp_extraversion, temp_neuroticism, temp_agreeableness, temp_conscientiousness = neuralModel.predict_single_frame(
                    frame, neuralModelDict)

                if temp_openness["average"] is not -1:
                    openness_average.append(temp_openness["average"])
                    if temp_openness["min"] < openness_min:
                        openness_min = temp_openness["min"]
                    if temp_openness["max"] > openness_max:
                        openness_max = temp_openness["max"]

                    valueUpdated = True

                if temp_extraversion["average"] is not -1:
                    extraversion_average.append(temp_extraversion["average"])
                    if temp_extraversion["min"] < extraversion_min:
                        extraversion_min = temp_extraversion["min"]
                    if temp_extraversion["max"] > extraversion_max:
                        extraversion_max = temp_extraversion["max"]

                    valueUpdated = True

                if temp_agreeableness["average"] is not -1:
                    agreeableness_average.append(temp_agreeableness["average"])
                    if temp_agreeableness["min"] < agreeableness_min:
                        agreeableness_min = temp_agreeableness["min"]
                    if temp_agreeableness["max"] > agreeableness_max:
                        agreeableness_max = temp_agreeableness["max"]

                    valueUpdated = True

                if temp_conscientiousness["average"] is not -1:
                    conscientiousness_average.append(temp_conscientiousness["average"])
                    if temp_conscientiousness["min"] < conscientiousness_min:
                        conscientiousness_min = temp_conscientiousness["min"]
                    if temp_conscientiousness["max"] > conscientiousness_max:
                        conscientiousness_max = temp_conscientiousness["max"]

                    valueUpdated = True

                if temp_neuroticism["average"] is not -1:
                    neuroticism_average.append(temp_neuroticism["average"])
                    if temp_neuroticism["min"] < neuroticism_min:
                        neuroticism_min = temp_neuroticism["min"]
                    if temp_neuroticism["max"] > neuroticism_max:
                        neuroticism_max = temp_neuroticism["max"]

                    valueUpdated = True

                font = cv2.FONT_HERSHEY_SIMPLEX

            if valueUpdated:
                raw_data = {
                    'trait_name': ['Openness', 'Extraversion', 'Agreeableness', 'Neuroticism', 'Conscientiousness'],
                    'avg': [np.average(openness_average), np.average(extraversion_average),
                            np.average(agreeableness_average), np.average(neuroticism_average),
                            np.average(conscientiousness_average)],
                    'min': [openness_min, extraversion_min, agreeableness_min, neuroticism_min, conscientiousness_min],
                    'max': [openness_max, extraversion_max, agreeableness_max, neuroticism_max, conscientiousness_max]}
                df = pd.DataFrame(raw_data, columns=['trait_name', 'avg', 'min', 'max'])
                graphThread = threading.Thread(target=self.update_graph, args=(df,))
                graphThread.start()

                self.Otable.setItem(0, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(np.average(openness_average))))
                self.Otable.setItem(1, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(openness_min)))
                self.Otable.setItem(2, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(openness_max)))


                self.Atable.setItem(0, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(np.average(agreeableness_average))))
                self.Atable.setItem(1, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(agreeableness_min)))
                self.Atable.setItem(2, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(agreeableness_max)))


                self.Etable.setItem(0, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(np.average(extraversion_average))))
                self.Etable.setItem(1, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(extraversion_min)))
                self.Etable.setItem(2, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(extraversion_max)))


                self.Ctable.setItem(0, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(np.average(conscientiousness_average))))
                self.Ctable.setItem(1, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(conscientiousness_min)))
                self.Ctable.setItem(2, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(conscientiousness_max)))

                self.Ntable.setItem(0, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(np.average(neuroticism_average))))
                self.Ntable.setItem(1, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(neuroticism_min)))
                self.Ntable.setItem(2, 0, QtWidgets.QTableWidgetItem("{0:.4f}".format(neuroticism_max)))


            cv2.putText(frame, "Openness (Avg) = {0:.4f}".format(np.average(openness_average)), (10, 20), font, 0.45,
                        color)
            cv2.putText(frame, "Extraversion (Avg) = {0:.4f}".format(np.average(extraversion_average)), (10, 40), font,
                        0.45, color)
            cv2.putText(frame, "Neuroticism (Avg) = {0:.4f}".format(np.average(neuroticism_average)), (10, 60), font,
                        0.45, color)
            cv2.putText(frame, "Agreeableness (Avg) = {0:.4f}".format(np.average(agreeableness_average)), (10, 80),
                        font, 0.45, color)
            cv2.putText(frame, "Consentiousness (Avg)= {0:.4f}".format(np.average(conscientiousness_average)),
                        (10, 100), font, 0.45, color)

            cv2.putText(frame, "Openness (Min) = {0:.4f}".format(openness_min), (10, 140), font, 0.45, color)
            cv2.putText(frame, "Extraversion (Min) = {0:.4f}".format(extraversion_min), (10, 160), font, 0.45, color)
            cv2.putText(frame, "Neuroticism (Min) = {0:.4f}".format(neuroticism_min), (10, 180), font, 0.45, color)
            cv2.putText(frame, "Agreeableness (Min) = {0:.4f}".format(agreeableness_min), (10, 200), font, 0.45, color)
            cv2.putText(frame, "Consentiousness (Min)= {0:.4f}".format(conscientiousness_min), (10, 220), font, 0.45,
                        color)

            cv2.putText(frame, "Openness (Max) = {0:.4f}".format(openness_max), (10, 260), font, 0.45, color)
            cv2.putText(frame, "Extraversion (Max) = {0:.4f}".format(extraversion_max), (10, 280), font, 0.45, color)
            cv2.putText(frame, "Neuroticism (Max) = {0:.4f}".format(neuroticism_max), (10, 300), font, 0.45, color)
            cv2.putText(frame, "Agreeableness (Max) = {0:.4f}".format(agreeableness_max), (10, 320), font, 0.45, color)
            cv2.putText(frame, "Consentiousness (Max)= {0:.4f}".format(conscientiousness_max), (10, 340), font, 0.45,
                        color)

            frame_count += 1
            cv2.imshow("output", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        plt.close()

    def start_camera(self):
        global cameraOpen
        global cameraThread

        if not cameraOpen:
            self.start_cam.setText("Stop")
            cameraOpen = True
            cameraThread = threading.Thread(target=self.camera_loop)
            cameraThread.start()

            if self.Atable.rowCount() is 0:


                self.Atable.setEnabled(True)
                self.Atable.insertColumn(0)
                self.Atable.setHorizontalHeaderLabels(["Value"])

                self.Atable.insertRow(self.Atable.rowCount())
                self.Atable.setItem(self.Atable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Atable.insertRow(self.Atable.rowCount())
                self.Atable.setItem(self.Atable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Atable.insertRow(self.Atable.rowCount())
                self.Atable.setItem(self.Atable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))


                self.Atable.setVerticalHeaderLabels(["Average", "Minimum", "Maximum"])

                self.Ctable.setEnabled(True)

                self.Ctable.insertColumn(0)
                self.Ctable.setHorizontalHeaderLabels(["Value"])

                self.Ctable.insertRow(self.Ctable.rowCount())
                self.Ctable.setItem(self.Ctable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Ctable.insertRow(self.Ctable.rowCount())
                self.Ctable.setItem(self.Ctable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Ctable.insertRow(self.Ctable.rowCount())
                self.Ctable.setItem(self.Ctable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))


                self.Ctable.setVerticalHeaderLabels(["Average", "Minimum", "Maximum"])



                self.Ntable.setEnabled(True)
                self.Ntable.insertColumn(0)
                self.Ntable.setHorizontalHeaderLabels(["Value"])

                self.Ntable.insertRow(self.Ntable.rowCount())
                self.Ntable.setItem(self.Ntable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Ntable.insertRow(self.Ntable.rowCount())
                self.Ntable.setItem(self.Ntable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Ntable.insertRow(self.Ntable.rowCount())
                self.Ntable.setItem(self.Ntable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))


                self.Ntable.setVerticalHeaderLabels(["Average", "Minimum", "Maximum"])


                self.Etable.setEnabled(True)
                self.Etable.insertColumn(0)
                self.Etable.setHorizontalHeaderLabels(["Value"])

                self.Etable.insertRow(self.Etable.rowCount())
                self.Etable.setItem(self.Etable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Etable.insertRow(self.Etable.rowCount())
                self.Etable.setItem(self.Etable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Etable.insertRow(self.Etable.rowCount())
                self.Etable.setItem(self.Etable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))


                self.Etable.setVerticalHeaderLabels(["Average", "Minimum", "Maximum"])


                self.Otable.setEnabled(True)
                self.Otable.insertColumn(0)
                self.Otable.setHorizontalHeaderLabels(["Value"])

                self.Otable.insertRow(self.Otable.rowCount())
                self.Otable.setItem(self.Otable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Otable.insertRow(self.Otable.rowCount())
                self.Otable.setItem(self.Otable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))

                self.Otable.insertRow(self.Otable.rowCount())
                self.Otable.setItem(self.Otable.rowCount() - 1, 0, QtWidgets.QTableWidgetItem("0"))


                self.Otable.setVerticalHeaderLabels(["Average", "Minimum", "Maximum"])


        else:
            self.start_cam.setText("Start")
            cameraOpen = False

