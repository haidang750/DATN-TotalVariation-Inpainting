from PyQt5 import QtCore, QtGui
from PyQt5 import QtGui
from PyQt5 import QtWidgets

class Ui_Dialog(QtWidgets.QDialog):
    def setupUi(self):
        self.setWindowTitle("About")
        self.setObjectName("Dialog")
        self.resize(450, 250)
        self.gridLayout_2 = QtWidgets.QGridLayout(self)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        font = QtGui.QFont()
        font.setPixelSize(18)

        self.label = QtWidgets.QLabel(self)
        self.label.setText("Ứng dụng phương pháp biến phân để xây dựng chương trình khôi phục ảnh số bằng kỹ thuật Inpainting")
        self.label.setAlignment(QtCore.Qt.AlignHCenter)
        self.label.setFont(font)
        self.gridLayout.addWidget(self.label, 1, 0)

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setText("Sinh viên thực hiện: Nguyễn Hải Đăng")
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignHCenter)
        self.gridLayout.addWidget(self.label_2, 2, 0)

        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setText("Mã số sinh viên: 102160034")
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignHCenter)
        self.gridLayout.addWidget(self.label_3, 3, 0)

        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setText("Lớp: 16T1")
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignHCenter)
        self.gridLayout.addWidget(self.label_4, 4, 0)

        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setText("Giáo viên hương dẫn: TS. Phạm Công Thắng")
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignHCenter)
        self.gridLayout.addWidget(self.label_5, 5, 0)

        font2 = QtGui.QFont()
        font2.setPixelSize(14)

        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setText("Đà Nẵng, 12/2020")
        self.label_6.setFont(font2)
        self.label_6.setAlignment(QtCore.Qt.AlignHCenter)
        self.gridLayout.addWidget(self.label_6, 6, 0)

        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.show()