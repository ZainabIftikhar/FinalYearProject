import sys
from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindowClass

app = QApplication(sys.argv)

window = MainWindowClass()

sys.exit(app.exec_())