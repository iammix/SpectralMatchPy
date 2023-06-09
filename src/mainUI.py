import sys

import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QDialog, QMessageBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import reqpy
import utilities


def remove_widget_from_layout(layout):
    count = layout.count()
    if count == 0:
        return layout
    else:
        item = layout.itemAt(count - 1)
        layout.removeItem(item)
    return layout


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=600, fig=None):
        if fig is None:
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
        else:
            self.fig = fig
            self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

    def reset(self):
        self.fig.clf()


class MainUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        uic.loadUi('../ui/main.ui', self)

        self.actionAbout.triggered.connect(self._gotoabout)
        self.actionExit.triggered.connect(self.exit_program)
        self.pushButton.clicked.connect(self.loadEqFile_and_plot)
        self.pushButton_2.clicked.connect(self.plot_ec8)
        self.pushButton_3.clicked.connect(self.fit)
        self.pushButton_4.clicked.connect(self.save_results_tab1)
        self.comboBox.currentTextChanged.connect(self._enable_dt)
        self.lineEdit_4.setEnabled(False)

    def _enable_dt(self):
        if self.comboBox.currentText() == 'PEER NGA' or self.comboBox.currentText() == 'Two Columns':
            self.lineEdit_4.setEnabled(False)
        else:
            self.lineEdit_4.setEnabled(True)

    def save_results_tab1(self):
        with open('vel.txt', 'w') as vel_file:
            for index, item in enumerate(self.cvel):
                vel_file.write(f'{self.time[index]} {item}\n')
        vel_file.close()

        with open('disp.txt', 'w') as disp_file:
            for index, item in enumerate(self.cdespl):
                disp_file.write(f'{self.time[index]} {item}\n')
        disp_file.close()

        with open('accel.txt', 'w') as accel_file:
            for index, item in enumerate(self.PSAccs):
                accel_file.write(f'{self.time[index]} {item}\n')
        accel_file.close()
        message_box = QMessageBox()
        message_box.setWindowTitle("Info")
        message_box.setText("Data Saved")
        message_box.setIcon(QMessageBox.Icon.Information)
        message_box.setStandardButtons(QMessageBox.StandardButton.Ok)

        # Show the message box and wait for the user's response
        response = message_box.exec()

    def exit_program(self):
        sys.exit()

    def _gotoabout(self):
        self.gotoabout = AboutPage()
        self.gotoabout.show()

    def fit(self):
        # TODO Export fitted data
        # labels: enhancement
        # assignees: iammix

        self.progressBar.setValue(0)
        fs = 1 / (self.time[1] - self.time[0])
        ccs, rms, misfit, self.cvel, self.cdespl, self.PSAccs, PSAs, T, sf, fig1, fig2 = reqpy.REQPY_single(
            np.array(self.accel), fs,
            self.ds_pga, self.ds_periods,
            T1=0, T2=10,
            zi=float(self.lineEdit_3.text()),
            nit=30, NS=100,
            baseline=True, plots=True, progress_bar_object=self.progressBar)
        plot_layout1 = self.verticalLayout_3
        plot_layout1 = remove_widget_from_layout(plot_layout1)
        canvas1 = FigureCanvasQTAgg(fig1)
        plot_layout1.addWidget(canvas1)
        canvas1.show()

        plot_layout2 = self.verticalLayout_4
        plot_layout2 = remove_widget_from_layout(plot_layout2)
        canvas2 = FigureCanvasQTAgg(fig2)
        plot_layout2.addWidget(canvas2)
        canvas2.show()

    def plot_ec8(self):
        self.ds_periods, self.ds_pga = utilities.ec8_rs(float(self.lineEdit_2.text()), self.comboBox_2.currentText(),
                                                        int(self.comboBox_3.currentText()),
                                                        importance_class=int(self.comboBox_4.currentText()),
                                                        damping=float(self.lineEdit_3.text()))

        plot_layout = self.verticalLayout_2
        sc = MplCanvas(self, width=10, height=4, dpi=80)
        plot_layout = remove_widget_from_layout(plot_layout)
        sc.axes.plot(self.ds_periods, self.ds_pga, linewidth=1.0)
        sc.axes.set_title('EC8 Design Spectrum')
        sc.axes.set_xlabel('Periods (sec)')
        sc.axes.set_ylabel('PGA (g)')
        plot_layout.addWidget(sc)
        sc.show()

    def loadEqFile_and_plot(self):
        # TODO Read different file formats
        # labels: enhancement
        # assignees: iammix

        eq_loader = QtWidgets.QFileDialog()
        self.eq_filePath = eq_loader.getOpenFileNames(self, 'Load File')
        if len(self.eq_filePath[0]) != 0:
            plot_layout = self.verticalLayout
            eq_line_edit = self.lineEdit
            sc = MplCanvas(self, width=10, height=4, dpi=60)
            plot_layout = remove_widget_from_layout(plot_layout)
            if self.comboBox.currentText() == 'PEER NGA':
                self.time, self.accel, self.dt = utilities.processNGAfile(self.eq_filePath[0][0])
            elif self.comboBox.currentText() == 'One Column':
                self.time, self.accel, self.dt = utilities.processOneCfile(self.eq_filePath[0][0])
            elif self.comboBox.currentText() == 'Two Columns':
                self.time, self.accel = utilities.processTwoCfile(self.eq_filePath[0][0])
            eq_line_edit.setText(self.eq_filePath[0][0])
            sc.axes.plot(self.time, self.accel, linewidth=0.5)
            sc.axes.set_title('Earthquake')
            sc.axes.set_xlabel('Time (sec)')
            sc.axes.set_ylabel('Acceleration (g)')
            plot_layout.addWidget(sc)
            sc.show()


class AboutPage(QDialog):
    def __init__(self):
        super(AboutPage, self).__init__()
        uic.loadUi('../ui/about.ui', self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainUI()
    window.show()
    app.exec()
