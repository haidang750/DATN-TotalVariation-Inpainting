from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
import numpy as np
import ImageProcessing as ImgPro
import CheckQualityImage as quality
import inpainting as ip
from datetime import datetime
import About
from cv2 import cv2
from numpy import clip, empty,  Inf, mod, sum, vstack, zeros
from numpy.linalg import norm

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Image Inpainting with Total Variation'
        self.originalImagePath = None
        self.originalImage = None
        self.textImage = None
        self.damagedImage = None
        self.recoveredImage = None
        self.differenceImage = None
        self.missingRate = 0
        self.timeRun = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        central = QtWidgets.QWidget()
        self.mainView = QtWidgets.QGridLayout()
        central.setLayout(self.mainView)
        self.setCentralWidget(central)

        self.addMenuBar()
        self.groupAddToImageButton()
        self.addGroupImage()
        self.addSaveImageButton()
        self.addResultRecoveredImage()
        self.addParameterView()

        imageView = QtWidgets.QGridLayout()
        imageView.addWidget(self.groupOriginalImage, 1, 0)
        imageView.addWidget(self.groupDamagedImage, 1, 1)
        imageView.addWidget(self.groupRecoveredImage, 1, 2)
        imageView.addWidget(self.groupDifferenceImage, 1 ,3)
        self.mainView.addLayout(imageView, 2, 0)

        buttonGroup = QtWidgets.QGridLayout()
        buttonGroup.addWidget(self.groupAddToImageButton, 1, 0)

        viewBottom = QtWidgets.QGridLayout()
        viewBottom.addLayout(buttonGroup, 1, 0)
        viewBottom.addWidget(self.groupParameter, 1, 1)
        viewBottom.addWidget(self.groupResultRecovered, 1, 2)

        self.mainView.addLayout(viewBottom, 3, 0)

        btnGroup = QtWidgets.QGridLayout()

        runMyIPButton = QtWidgets.QPushButton(text="Run MY INPAINTING")
        runMyIPButton.setStyleSheet(
            "font-weight: bold; color: black; font-size: 20; background-color: #44bcd8")
        runMyIPButton.pressed.connect(self.clickRunMyIPButton)
        runMyIPButton.setFixedSize(250, 40)

        runAvailableIPButton = QtWidgets.QPushButton(text="Run AVAILABLE INPAINTING")
        runAvailableIPButton.setStyleSheet(
            "font-weight: bold; color: black; font-size: 20; background-color: #FFFFFF")
        runAvailableIPButton.pressed.connect(self.clickRunAvailableIPButton)
        runAvailableIPButton.setFixedSize(250, 40)

        clearButton = QtWidgets.QPushButton(text="Clear")
        clearButton.setStyleSheet(
            "font-weight: bold; color: white; font-size: 20; background-color: red")
        clearButton.setFixedSize(250, 40)
        clearButton.pressed.connect(self.clearData)

        saveButton = QtWidgets.QPushButton(text="Save")
        saveButton.setStyleSheet(
            "font-weight: bold; color: white; font-size: 20; background-color: green")
        saveButton.pressed.connect(self.chooseFolderSaveImage)
        saveButton.setFixedSize(250, 40)

        btnGroup.addWidget(runMyIPButton, 1, 0)
        btnGroup.addWidget(runAvailableIPButton, 1, 1)
        btnGroup.addWidget(clearButton, 1, 2)
        btnGroup.addWidget(saveButton, 1, 3)
        self.mainView.addLayout(btnGroup, 4, 0)

        self.show()

    def addResultRecoveredImage(self):
        self.viewResult = QtWidgets.QGridLayout()
        self.groupResultRecovered = QtWidgets.QGroupBox("Result")

        self.psnrLabel = QtWidgets.QLabel()
        self.psnrLabel.setStyleSheet("font-weight: bold; color: black")
        self.psnrLabel.setText("PSNR: ")

        self.ssimLabel = QtWidgets.QLabel()
        self.ssimLabel.setStyleSheet("font-weight: bold; color: black")
        self.ssimLabel.setText("SSIM: ")

        self.timeRunLabel = QtWidgets.QLabel()
        self.timeRunLabel.setStyleSheet("font-weight: bold; color: black")
        self.timeRunLabel.setText("Time Run: ")

        self.viewResult.addWidget(self.psnrLabel, 3, 0)
        self.viewResult.addWidget(self.ssimLabel, 4, 0)
        self.viewResult.addWidget(self.timeRunLabel, 5, 0)

        self.groupResultRecovered.setLayout(self.viewResult)
        self.groupResultRecovered.setFixedWidth(350)

    def addParameterView(self):
        self.viewParameter = QtWidgets.QGridLayout()
        self.groupParameter = QtWidgets.QGroupBox("Parameters for MY INPAINTING")

        self.alphaLabel = QtWidgets.QLabel()
        self.alphaLabel.setStyleSheet("font-weight: bold; color: black")
        self.alphaLabel.setText("Alpha:")

        self.alphaTextField = QtWidgets.QLineEdit()
        self.alphaTextField.setText("400")

        self.betaLabel = QtWidgets.QLabel()
        self.betaLabel.setStyleSheet("font-weight: bold; color: black")
        self.betaLabel.setText("Beta:")

        self.betaTextField = QtWidgets.QLineEdit()
        self.betaTextField.setText("1")

        self.viewParameter.addWidget(self.alphaLabel, 1, 0)
        self.viewParameter.addWidget(self.alphaTextField, 2, 0)
        self.viewParameter.addWidget(self.betaLabel, 3, 0)
        self.viewParameter.addWidget(self.betaTextField, 4, 0)

        self.groupParameter.setLayout(self.viewParameter)
        self.groupParameter.setFixedHeight(150)
        self.groupParameter.setFixedWidth(200)

    def groupAddToImageButton(self):
        self.viewContainButton = QtWidgets.QGridLayout()

        self.groupAddToImageButton = QtWidgets.QGroupBox("Add To Image")

        self.addTextButton = QtWidgets.QPushButton(text="Text")
        self.addTextButton.pressed.connect(self.addText)
        self.addTextButton.setFixedWidth(160)

        self.addMissingPixelButton = QtWidgets.QPushButton(text="Missing Pixels")
        self.addMissingPixelButton.pressed.connect(self.makeMissingPixels)
        self.addMissingPixelButton.setFixedWidth(160)

        self.viewContainButton.addWidget(self.addTextButton, 1, 0)
        self.viewContainButton.addWidget(self.addMissingPixelButton, 2, 0)

        self.groupAddToImageButton.setLayout(self.viewContainButton)
        self.groupAddToImageButton.setFixedHeight(100)
        self.groupAddToImageButton.setFixedWidth(200)

    def addGroupImage(self):
        self.groupOriginalImage = QtWidgets.QGroupBox("Original Image")

        self.originalImageView = QtWidgets.QLabel()
        self.originalImageView.setFixedSize(255, 255)

        originalImageLayout = QtWidgets.QGridLayout()
        originalImageLayout.addWidget(self.originalImageView)

        self.groupOriginalImage.setLayout(originalImageLayout)
        self.groupOriginalImage.setFixedHeight(300)

        self.groupDamagedImage = QtWidgets.QGroupBox("Damaged Image")

        self.damagedImageView = QtWidgets.QLabel()
        self.damagedImageView.setFixedSize(255, 255)

        damagedImageLayout = QtWidgets.QGridLayout()
        damagedImageLayout.addWidget(self.damagedImageView)

        self.groupDamagedImage.setLayout(damagedImageLayout)
        self.groupDamagedImage.setFixedHeight(300)

        self.groupDifferenceImage = QtWidgets.QGroupBox("Difference")

        self.differenceImageView = QtWidgets.QLabel()
        self.differenceImageView.setFixedSize(255, 255)

        differenceImageLayout = QtWidgets.QGridLayout()
        differenceImageLayout.addWidget(self.differenceImageView)

        self.groupDifferenceImage.setLayout(differenceImageLayout)
        self.groupDifferenceImage.setFixedHeight(300)

        self.groupRecoveredImage = QtWidgets.QGroupBox("Recovered Image")

        self.recoveredImageView = QtWidgets.QLabel()
        self.recoveredImageView.setFixedSize(255, 255)

        recoveredImageLayout = QtWidgets.QGridLayout()
        recoveredImageLayout.addWidget(self.recoveredImageView)

        self.groupRecoveredImage.setLayout(recoveredImageLayout)
        self.groupRecoveredImage.setFixedHeight(300)

    def addSaveImageButton(self):
        clearButton = QtWidgets.QPushButton(text="Clear")
        clearButton.setFixedWidth(150)
        clearButton.pressed.connect(self.clearData)

        saveButton = QtWidgets.QPushButton(text="Save")
        saveButton.pressed.connect(self.chooseFolderSaveImage)
        saveButton.setFixedWidth(150)

        self.viewClearSaveButton = QtWidgets.QHBoxLayout()
        self.viewClearSaveButton.addStretch()
        self.viewClearSaveButton.addWidget(clearButton)
        self.viewClearSaveButton.addWidget(saveButton)

    def addMenuBar(self):
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        aboutMenu = mainMenu.addMenu('About')

        openButton = QtWidgets.QAction(
            QtGui.QIcon('exit24.png'), 'Open Image', self)
        openButton.setShortcut('Ctrl+O')
        openButton.setStatusTip('Open an image')
        openButton.triggered.connect(self.openOriginalImage)
        fileMenu.addAction(openButton)

        aboutButton = QtWidgets.QAction(QtGui.QIcon(), 'About Project', self)
        aboutButton.triggered.connect(self.openAboutDialog)
        aboutMenu.addAction(aboutButton)

    @QtCore.pyqtSlot()
    def openAboutDialog(self):
        ui = About.Ui_Dialog()
        ui.setupUi()
        ui.exec_()

    @QtCore.pyqtSlot()
    def openOriginalImage(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", "", "Image File (*.*)", options=options)
        if fileName:
            image = cv2.imread(fileName)
            image = ImgPro.resizeImage(image,256)
            resizePath = ImgPro.saveImage("resizeOriginalImage.jpg", image)
            pixmap = QtGui.QPixmap(resizePath)
            self.clearData()
            self.originalImage = image
            self.originalImageView.setPixmap(pixmap)
            self.originalImagePath = resizePath

    @QtCore.pyqtSlot()
    def chooseFolderSaveImage(self):
        if self.recoveredImage is not None:
            recovered_im = ImgPro.rescale255(self.recoveredImage)

            damaged_im = ImgPro.rescale255(self.damagedImage)

            options = QtWidgets.QFileDialog.Options()
            fileName_damaged, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Damaged image", "", "Image File (*.png *.jpg *.jpeg)", options=options)
            fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Recovered image", "", "Image File (*.png *.jpg *.jpeg)", options=options)
            if fileName_damaged:
                ImgPro.customSaveImage(fileName_damaged, damaged_im)
                QtWidgets.QMessageBox.information(
                    self, "Image Processing", "Lưu Damaged Image thành công.")
            if fileName:
                ImgPro.customSaveImage(fileName, recovered_im)
                QtWidgets.QMessageBox.information(
                    self, "Image Processing", "Lưu Recovered Image thành công.")
        else:
            self.showAlert("Bạn chưa chọn ảnh nào.")

    @QtCore.pyqtSlot()
    def addText(self):
        if self.originalImagePath is not None:
            options = QtWidgets.QFileDialog.Options()
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open image", "", "Image File (*.*)", options=options)
            if fileName:
                # reset current image
                self.recoveredImage = None
                self.recoveredImageView.setPixmap(QtGui.QPixmap())
                self.differenceImage = None
                self.differenceImageView.setPixmap(QtGui.QPixmap())

                image = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
                image = ImgPro.resizeImage(image, 256)
                merge_image = self.originalImage.copy()
                alpha_s = image[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    merge_image[:, :, c] = (alpha_s * image[:, :, c] + alpha_l * merge_image[:, :, c])

                resizePath = ImgPro.saveImage("text_damage.jpg", merge_image)
                pixmap = QtGui.QPixmap(resizePath)
                self.damagedImageView.setPixmap(pixmap)
                self.damagedImage = merge_image
                # show Total variation value of Damaged Image
                rows, cols, colors = merge_image.shape
                col_diff = merge_image[: -1, 1:] - merge_image[: -1, : -1]
                row_diff = merge_image[1:, : -1] - merge_image[: -1, : -1]
                diff_norms = norm(vstack((col_diff.T.flatten(), row_diff.T.flatten())).T, ord=2, axis=1)
                value = sum(diff_norms) / ((rows - 1) * (cols - 1))
                print("Total variation value of Damaged Image = ", value)
        else:
            self.showAlert("Bạn chưa chọn ảnh gốc.")
    def makeMissingPixels(self):
        if self.originalImagePath is not None:
            d, okPressed = QtWidgets.QInputDialog.getInt(
                self, "Enter value", "Rate(%):", 50, 0, 90, 5)
            if okPressed:
                # reset current image
                self.recoveredImage = None
                self.recoveredImageView.setPixmap(QtGui.QPixmap())
                self.differenceImage = None
                self.differenceImageView.setPixmap(QtGui.QPixmap())

                shape = self.originalImage.shape
                total = range(np.prod(shape))
                unknown = self.killed_pixels(shape, frac_kill=d/100)
                known = list(set(total) - set(unknown))
                image = np.zeros(shape)
                rows, cols, colors = np.unravel_index(known, shape)
                image[rows, cols, colors] = self.originalImage[rows, cols, colors]

                resizePath = ImgPro.saveImage("missing_pixels_damage.jpg", image)
                pixmap = QtGui.QPixmap(resizePath)
                self.damagedImageView.setPixmap(pixmap)
                self.damagedImage = image
                # show Total variation value of Damaged Image
                rows, cols, colors = image.shape
                col_diff = image[: -1, 1:] - image[: -1, : -1]
                row_diff = image[1:, : -1] - image[: -1, : -1]
                diff_norms = norm(vstack((col_diff.T.flatten(), row_diff.T.flatten())).T, ord=2, axis=1)
                value = sum(diff_norms) / ((rows - 1) * (cols - 1))
                print("Total variation value of Damaged Image = ", value)
        else:
            self.showAlert("Bạn chưa chọn ảnh gốc.")

    def killed_pixels(self, shape, frac_kill):
        npixels = np.prod(shape)
        num_kill = int(frac_kill * npixels)
        inds = np.random.choice(npixels, num_kill, replace=False)
        return inds

    def showAlert(self, message):
        QtWidgets.QMessageBox.warning(self, "Alert", message)

    @QtCore.pyqtSlot()
    def clickRunMyIPButton(self):
        if self.damagedImage is not None:
            print("Inpainting Image Process...")
            start = datetime.now()

            alpha = float(self.alphaTextField.text())
            beta = float(self.betaTextField.text())
            image = ip.getRecoveredImage(self.originalImage, self.damagedImage, alpha, beta)

            self.timeRun = datetime.now() - start
            print("Done")
            self.showAlert("Quá trình khôi phục ảnh hoàn tất.")

            diffImage = np.abs(image - self.originalImage)
            self.recoveredImage = image
            self.differenceImage = diffImage

            resizePath = ImgPro.saveImage("recoveredImage.jpg", self.recoveredImage)
            pixmap = QtGui.QPixmap(resizePath)
            self.recoveredImageView.setPixmap(pixmap)

            resizePath2 = ImgPro.saveImage("differenceImage.jpg", diffImage)
            pixmap2 = QtGui.QPixmap(resizePath2)
            self.differenceImageView.setPixmap(pixmap2)

            if self.originalImage is not None and self.recoveredImage is not None:
                self.evaluationImage(self.originalImage, self.recoveredImage)
        else:
            self.showAlert("Vui lòng chọn ảnh gốc và thêm Text hoặc Missing Pixels vào ảnh.")

    def clickRunAvailableIPButton(self):
        if self.damagedImage is not None:
            start = datetime.now()

            image = ip.availableInpainting(self.originalImage, self.damagedImage)
            diffImage = np.abs(image - self.originalImage)
            self.showAlert("Quá trình khôi phục ảnh hoàn tất.")

            self.timeRun = datetime.now() - start

            self.recoveredImage = image
            self.differenceImage = diffImage

            resizePath = ImgPro.saveImage("recoveredImage.jpg", self.recoveredImage)
            pixmap = QtGui.QPixmap(resizePath)
            self.recoveredImageView.setPixmap(pixmap)

            resizePath2 = ImgPro.saveImage("differenceImage.jpg", diffImage)
            pixmap2 = QtGui.QPixmap(resizePath2)
            self.differenceImageView.setPixmap(pixmap2)

            if self.originalImage is not None and self.recoveredImage is not None:
                self.evaluationImage(self.originalImage, self.recoveredImage)
        else:
            self.showAlert("Vui lòng chọn ảnh gốc và thêm Text hoặc Missing Pixels vào ảnh.")

    def clearData(self):
        self.recoveredImage = None
        self.originalImagePath = None
        self.textImage = None
        self.damagedImage = None
        self.differenceImage = None
        self.originalImageView.setPixmap(QtGui.QPixmap())
        self.damagedImageView.setPixmap(QtGui.QPixmap())
        self.recoveredImageView.setPixmap(QtGui.QPixmap())
        self.differenceImageView.setPixmap(QtGui.QPixmap())
        self.psnrLabel.setText("PSNR: ")
        self.ssimLabel.setText("SSIM: ")
        self.timeRunLabel.setText("Time Run: ")

    def evaluationImage(self, original, recovered):
        original = original.astype(np.float32)
        recovered = recovered.astype(np.float32)
        psnr = quality.PSNR(original, recovered)
        ssim = quality.SSIM(original, recovered)
        self.psnrLabel.setText("PSNR: " + str(round(psnr, 4)))
        self.ssimLabel.setText("SSIM: " + str(round(ssim, 4)))
        self.timeRunLabel.setText(
            "Time Run: " + str(round(self.timeRun.total_seconds(), 4)) + " (s)")
