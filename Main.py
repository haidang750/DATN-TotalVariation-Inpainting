import sys
import myGUI
from PyQt5 import QtWidgets
# from PyQt5.QtGui import QIcon, QPixmap

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Image Inpainting With Total Variation")
    ex = myGUI.App()
    sys.exit(app.exec_())