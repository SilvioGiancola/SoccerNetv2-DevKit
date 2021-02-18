import sys
from PyQt5.QtWidgets import QApplication

from interface.main_window import MainWindow

if __name__ == "__main__":

	application = QApplication(sys.argv)
	window = MainWindow()
	sys.exit(application.exec_())