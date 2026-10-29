#!/usr/bin/env python
# -*- coding:utf8 -*-
import os, sys
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QEvent  # 명시적으로 QEvent 추가
from PyQt5.QtWidgets import QApplication, QShortcut
from PyQt5.QtGui import QPen, QColor, QPainterPath, QPixmap, QBrush, QKeySequence


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path
import cv2
#from threading import Timer

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

from window_tools.viz import VizTools
from window_tools.event import EventTools
from utils.updater import DelayedUpdater
from utils.converter import load_calibration_params

# set initial 4 points
x1=800
y1=100

x2=500
y2=200

x3=300
y3=300

x4=200
y4=500

x5=100
y5=200

class Window(QWidget, VizTools, EventTools):
    imgIndex = -1
    saveFlag = True
    # Lane 색상 및 타입 매핑
    LANE_COLORS = {
        'Yellow': {
            'mpl_color': 'red',
            'vtk_color': (1.0, 0.0, 0.0),
            'category': 'yellow_lane'
        },
        'White': {
            'mpl_color': '#ff69b4',
            'vtk_color': (1.0, 0.412, 0.706),
            'category': 'white_lane'
        },
        'WhiteDash': {
            'mpl_color': 'limegreen',
            'vtk_color': (0.196, 0.804, 0.196),
            'category': 'white_dash_lane'
        },
        'Default': {
            'mpl_color': 'limegreen',
            'vtk_color': (0.196, 0.804, 0.196),
            'category': 'unknown'
        }
    }

    def __init__(self,path):
        super().__init__()
        self.img_path = os.getcwd() + os.sep + path  # path는 'data'
        self.img_dir = os.path.join(self.img_path, 'image')
        self.pcd_dir = os.path.join(self.img_path, 'pcd')

        # a figure instance to plot on
        self.figure = Figure(tight_layout=True, dpi=96)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas.updateGeometry()

        # To store the draggable polygon
        self.list_points = []
        self.list_points_type = []
        self.sampled_pts = []
        self.sampled_uv = []
        self.vtk_lanes = []  # VTK 라벨 여러 개 저장

        # calibration params
        self.t, self.r, self.k, self.distortion = load_calibration_params()

        # To store img path
        self.list_img_path = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        # PCD 파일명 리스트
        self.list_pcd_path = sorted([
            f for f in os.listdir(self.pcd_dir)
            if f.endswith('.pcd.bin')
        ])

        self.loadImg(self.img_path)


        # LiDAR widget
        # self.vtk_actor = vtk.vtkActor()
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.vtkRenderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.vtkRenderer)
        # self.addPointCloudToVTK()
        # self.addColoredPointCloudToVTK()
        
        # Connect VTK click event to handler
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        interactor.AddObserver("LeftButtonPressEvent", self.on_vtk_click)
        interactor.AddObserver("MouseMoveEvent", self.on_vtk_motion)
        interactor.AddObserver("LeftButtonReleaseEvent", self.on_vtk_release)
        self._drag = {'active': False, 'vtk_actor': None}


        addLaneLabel = QLabel()
        addLaneLabel.setText("Label Setting")
        
        # 모드 선택기 삭제 - 통합 관리로 변경

        addLaneListButton = QComboBox()
        addLaneListButton.addItems(["---Select line type---","White line", "White dash line", "Yellow line"])

        lineGroupBox = QGroupBox("Line Class", self)
        vtkGroupBox = QGroupBox("3D View", self)

        # 라디오 버튼 스타일 시트
        radio_style = """
            QRadioButton {
                spacing: 10px;
                font-weight: bold;
                min-height: 30px;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border-radius: 10px;
            }
            QRadioButton::indicator::unchecked {
                background-color: #cccccc;
                border: 2px solid #666666;
            }
            QRadioButton::indicator::checked {
                background-color: #c2185b;
                border: 2px solid #888888;
            }
            QRadioButton::indicator::unchecked:hover {
                background-color: #ff69b4;
            }
            QRadioButton::indicator::checked:hover {
                background-color: #ff69b4;
            }
        """

        
        # Yellow Line 라디오 버튼
        self.lineRadio1 = QRadioButton("Yellow Line (q)", self)
        self.lineRadio1.setStyleSheet(radio_style)
        self.lineRadio1.clicked.connect(self.radioButtonClicked)
        self.lineRadio2 = QRadioButton("White Line (w)", self)
        self.lineRadio2.setStyleSheet(radio_style)
        self.lineRadio2.clicked.connect(self.radioButtonClicked)
        self.lineRadio3 = QRadioButton("White Dash Line (e)", self)
        self.lineRadio3.setStyleSheet(radio_style)
        self.lineRadio3.clicked.connect(self.radioButtonClicked)

        self.lineButtonGroup = QButtonGroup(self)
        self.lineButtonGroup.addButton(self.lineRadio1)
        self.lineButtonGroup.addButton(self.lineRadio2)
        self.lineButtonGroup.addButton(self.lineRadio3)
        self.lineRadio1.setChecked(True)
        self.beforeRadioChecked = self.lineRadio1

        # VTK 컬러모드 라디오 버튼
        self.colorRadio = QRadioButton("RGB (r)", self)
        self.colorRadio.setStyleSheet(radio_style)
        self.colorRadio.clicked.connect(self.radioButtonClicked)
        self.intensityRadio = QRadioButton("Intensity (t)", self)
        self.intensityRadio.setStyleSheet(radio_style)
        self.intensityRadio.clicked.connect(self.radioButtonClicked)

        self.vtkButtonGroup = QButtonGroup(self)
        self.vtkButtonGroup.addButton(self.colorRadio)
        self.vtkButtonGroup.addButton(self.intensityRadio)
        self.colorRadio.setChecked(True)

        
           
        addLaneButton = QPushButton("Add Lane (a)")
        addLaneButton.clicked.connect(self.add_lane)

        delLaneButton = QPushButton("Delete Lane (s)")
        delLaneButton.clicked.connect(self.delete_last_lane)

        delPointButton = QPushButton("Delete Point (d)")
        delPointButton.clicked.connect(self.delete_point)

        curPosButton = QPushButton("Show current Labels (f)")
        curPosButton.clicked.connect(self.showPosition)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        nextImgButton = QPushButton("Next Image (c)")

        nextImgButton.clicked.connect(self.loadNextImage)

        preImgButton = QPushButton("Prev Image (z)")

        preImgButton.clicked.connect(self.loadPrevImage)

        saveButton = QPushButton("Save (x)")
        saveButton.clicked.connect(lambda: self.saveAll(self.img_path))

        self.editBox = QPlainTextEdit()
        self.editBox.setFixedWidth(160)
        self.editBox.setPlainText("")
        self.editBox.setReadOnly(True)
        self.editBox.setDisabled(True)


        # Prepare a group button

        preNextLayout = QHBoxLayout()
        preNextLayout.addWidget(preImgButton)
        preNextLayout.addWidget(nextImgButton)

        saveLayout = QVBoxLayout()
        saveLayout.addLayout(preNextLayout)
        saveLayout.addWidget(saveButton)

        lineGrouplayout = QVBoxLayout()
        lineGroupBox.setLayout(lineGrouplayout)
        lineGrouplayout.addWidget(self.lineRadio1)
        lineGrouplayout.addWidget(self.lineRadio2)
        lineGrouplayout.addWidget(self.lineRadio3)

        vtkGrouplayout = QVBoxLayout()
        vtkGroupBox.setLayout(vtkGrouplayout)
        vtkGrouplayout.addWidget(self.colorRadio)
        vtkGrouplayout.addWidget(self.intensityRadio)

        addDelLayout = QHBoxLayout()
        addDelLayout.addWidget(delPointButton)

        addLayout = QVBoxLayout()
        addLayout.addWidget(addLaneLabel)
        addLayout.addWidget(lineGroupBox)
        addLayout.addWidget(vtkGroupBox)
        
        addLayout.addWidget(addLaneButton)
        addLayout.addWidget(delLaneButton)
        addLayout.addLayout(addDelLayout)

        rightLayout = QVBoxLayout()
        rightLayout.addLayout(addLayout)
        rightLayout.addSpacing(20)
        rightLayout.addSpacing(20)
        rightLayout.addWidget(curPosButton)
        rightLayout.addWidget(self.editBox)
        rightLayout.addSpacing(20)
        rightLayout.addSpacerItem(verticalSpacer)
        rightLayout.addLayout(saveLayout)
        
        

        pcdWidget = QWidget()
        pcdWidget.setLayout(QVBoxLayout())
        pcdWidget.layout().addWidget(self.vtkWidget)
        
        layout = QHBoxLayout()
        layout.addWidget(self.canvas, 3)
        layout.addWidget(pcdWidget, 2)
        layout.addLayout(rightLayout)
        self.setLayout(layout)

        # Prevent "QWidget::repaint: Recursive repaint detected"
        self.delayer = DelayedUpdater(self.canvas)

        # for first label load bug
        #t = Timer(1.0, self.loadFirstLabel)
        #t.start()

        # 점 저장용 리스트
        self.lane_points = []
        # 곡선별 점 인덱스 범위 저장 ([(start_idx, end_idx), ...])
        self.lane_labels = []
        # 점/곡선 아티스트 저장
        self.lane_point_artists = []  # scatter 반환값들 (Add Label 전 점)
        self.lane_curve_artists = []  # plot 반환값들
        self.all_point_artists = []   # 전체 점 아티스트 (곡선 생성 후에도 유지)

        # __init__ 안에 추가
        self.canvas.mpl_connect('button_press_event', self.on_mpl_click)
        
        # 키보드 단축키 설정 - 전체 어플리케이션에 적용
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 키보드 이벤트를 전체 어플리케이션에 적용
        QApplication.instance().installEventFilter(self)
        

        self.plotBackGround(self.img_path,0,True)

    # 단축키 함수들
    def select_yellow_line(self):
        self.lineRadio1.setChecked(True)
        self.radioButtonClicked()
    
    def select_white_line(self):
        self.lineRadio2.setChecked(True)
        self.radioButtonClicked()
    
    def select_white_dash_line(self):
        self.lineRadio3.setChecked(True)
        self.radioButtonClicked()
    
    def select_color(self):
        self.colorRadio.setChecked(True)
        self.vtkViewClicked()
    
    def select_intensity(self):
        self.intensityRadio.setChecked(True)
        self.vtkViewClicked()
    
    def eventFilter(self, obj, event):
        """전체 어플리케이션에서 키보드 이벤트를 캡처
        q: Yellow line button
        w: white line button
        e: white dash line button
        a: add Lane button
        s: delete Lane button
        d: delete point button
        z: prev image button
        x: save button
        c: next image button
        """
        if event.type() == QEvent.KeyPress:
            key = event.key()
            text = event.text().lower()
                        
            # 단축키 처리
            if text == 'q' or text=='ㅂ':
                self.select_yellow_line()
                return True
            elif text == 'w' or text=='ㅈ':
                self.select_white_line()
                return True
            elif text == 'e' or text=='ㄷ':
                self.select_white_dash_line()
                return True
            elif text == 'a' or text=='ㅁ':
                self.add_lane()
                return True
            elif text == 's' or text=='ㄴ':
                self.delete_last_lane()
                return True
            elif text == 'd' or text=='ㅇ':
                self.delete_point()
                return True
            elif text == 'z' or text=='ㅋ':
                self.loadPrevImage()
                return True
            elif text == 'x' or text=='ㅌ':
                self.saveAll(self.img_path)
                return True
            elif text == 'c' or text=='ㅊ':
                self.loadNextImage()
                return True
            elif text == 'r' or text=='ㄱ':
                self.select_color()
                return True
            elif text == 't' or text=='ㅅ':
                self.select_intensity()
                return True
            elif text == 'f' or text=='ㄹ':
                self.showPosition()
                return True
                
        return super().eventFilter(obj, event)
    
    def keyPressEvent(self, event):
        """기존 keyPressEvent 메서드는 유지하지만 이벤트 필터가 우선 처리함"""
        # 이벤트 필터가 우선적으로 처리하도록 함
        super().keyPressEvent(event)


    def plotBackGround(self,img_path,action,isFirst=False):
        self.unified_lanes = []  # 이미지 바뀔 때 레인 정보 초기화
        ''' Plot background method '''
        isPlot = True
        isEdge = False

        """
        if not isFirst:
            # if not saved, popup message box
            if self.imgIndex == len(self.list_img_path):
                self.saveFlag = self.isPosNotChange(img_path,self.imgIndex-1)
            else:
                self.saveFlag = self.isPosNotChange(img_path,self.imgIndex)

            if not self.saveFlag:
                isPlot = self.msgBoxEvent()
        """


        if hasattr(self, 'lane_curve_artists'):
            # 안전하게 커브 아티스트 제거
            for curve in list(self.lane_curve_artists):
                try:
                    curve.remove()
                except ValueError:
                    # 이미 제거된 경우 무시
                    pass
            self.lane_curve_artists.clear()

        # 점 아티스트도 함께 제거 (선택 사항)
        # 점 아티스트도 함께 제거 (선택 사항)
        if hasattr(self, 'all_point_artists'):
            for pt in list(self.all_point_artists):
                try:
                    pt.remove()
                except ValueError:
                    # 이미 제거된 경우 무시
                    pass
            self.all_point_artists.clear()
                
        if isPlot:
            # increase img index
            if action == 0 and self.imgIndex < len(self.list_img_path):
                if self.imgIndex == -1 and isFirst == False :    # boundary scenario
                    self.imgIndex += 1
                self.imgIndex += 1
            elif action == 1 and self.imgIndex > -1:
                if self.imgIndex == len(self.list_img_path):
                    self.imgIndex -= 1
                self.imgIndex -= 1

            if self.imgIndex == len(self.list_img_path) or (isFirst == False and self.imgIndex == -1):
                isEdge = self.msgBoxReachEdgeEvent()

            if not isEdge:
                # for faster scene update
                self.canvas.setUpdatesEnabled(False)

                # clean up list points
                self.delAllLine()


                #while self.list_points:
                #    self.delLastLine()

                path = img_path + '/' + self.list_img_path[self.imgIndex].replace('\\', '/')

                # TODO: written for test
                img = cv2.imread(path)
                if img is None:
                    print(f"이미지 {path}를 불러올 수 없습니다.")
                    return

                # distortion 보정
                img_undistorted = cv2.undistort(img, self.k, self.distortion)

                # matplotlib 등에서 RGB로 쓰려면 변환
                img_undistorted = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
                self.img = img_undistorted
                height, width, channels = img_undistorted.shape

                if isFirst:
                    self.pyt = self.axes.imshow(img_undistorted)
                else:
                    self.pyt.set_data(img_undistorted)

                # initial (x,y) position in image range
                global x1,y1,x2,y2,x3,y3,x4,y4
                x_range = width - 0
                y_range = height - 0

                self.canvasSize = [width, height]

                # Divide into six equal segments
                x1,x2,x3,x4 = x_range*(4.0/6), x_range*(3.0/6), x_range*(2.0/6), x_range*(1.0/6)
                y1,y2,y3,y4 = y_range*(1.0/6), y_range*(2.0/6), y_range*(3.0/6), y_range*(4.0/6)

                # Edit window title
                self.setWindowTitle("IRCV: 3D Lane Labeling Tool")

                self.canvas.draw()

                # for faster scene update
                self.canvas.setUpdatesEnabled(True)
                self.vtkRenderer.RemoveAllViewProps() 
                if self.intensityRadio.isChecked():
                    self.addPointCloudToVTK(self.imgIndex)
                elif self.colorRadio.isChecked():
                    self.addColoredPointCloudToVTK(self.imgIndex)

    def loadImg(self, directory):
        try:
            self.list_img_path = []
            for file in os.listdir(os.path.join(directory, 'image')):
                if file.endswith('.jpg') or file.endswith('.png'):
                    print(file)
                    self.list_img_path.append(os.path.join('image', file))
        except Exception as e:
            sys.exit(str(e))

    def showPosition(self):
        ''' display current labeled lanes in requested format '''
        # Helper to count lane types
        def count_lanes(lane_types):
            counts = {"White lane": 0, "White Dash lane": 0, "Yellow lane": 0}
            for t in lane_types:
                t_lower = str(t).lower().replace('_', ' ').replace('-', ' ').replace('  ', ' ')
                t_lower = t_lower.replace('lane', '').strip()
                if "white dash" in t_lower or "whitedash" in t_lower:
                    counts["White Dash lane"] += 1
                elif "white" in t_lower:
                    counts["White lane"] += 1
                elif "yellow" in t_lower:
                    counts["Yellow lane"] += 1
            return counts

        # 출력 라벨 정의
        output_labels = {
            "White lane": "White",
            "White Dash lane": "White Dash",
            "Yellow lane": "Yellow"
        }
        # Current working data
        text = "Current Working Data:\n"
        if hasattr(self, 'list_points_type'):
            current_counts = count_lanes(self.list_points_type)
            for name in ["White lane", "White Dash lane", "Yellow lane"]:
                text += f"{output_labels[name]}: {current_counts[name]} lanes\n"
        else:
            text += "No data.\n"
        text += "\n"

        # Processed data (labeled data)
        text += "Processed Data:\n"
        # 'self.labeled_data'가 존재하는지 확인
        labeled_types = []
        if hasattr(self, 'labeled_data') and self.labeled_data is not None:
            # labeled_data가 [{'type': ...}, ...] 형식이면 아래처럼 추출
            for item in self.labeled_data:
                if isinstance(item, dict) and 'type' in item:
                    labeled_types.append(item['type'])
                else:
                    labeled_types.append(str(item))
        else:
            labeled_types = []
        total_labeled = len(labeled_types)
        text += f"Total {total_labeled} data\n"
        labeled_counts = count_lanes(labeled_types)
        for name in ["White lane", "White Dash lane", "Yellow lane"]:
            text += f"{output_labels[name]}: {labeled_counts[name]} lanes\n"

        self.editBox.setPlainText(text)


    def isPosNotChange(self,img_path,index):
        # load label txt
        """
        fileName = img_path+"label_txt"+os.sep+self.list_img_path[index][:-4]+".txt"
        lab = []
        labnp = np.array(lab)
        curnp = np.array(lab)
        try:
            with open(fileName, 'r') as f:
                x = f.read().splitlines()
                for line in x:
                    select,xstr1,ystr1,xstr2,ystr2,xstr3,ystr3,xstr4,ystr4 = line.split(',')
                    x1,y1,x2,y2,x3,y3,x4,y4 = float(xstr1),float(ystr1),float(xstr2),float(ystr2),float(xstr3),float(ystr3),float(xstr4),float(ystr4)
                    lab.append([x1,y1,x2,y2,x3,y3,x4,y4])
                labnp = np.array(lab)

        except IOError:
            lab = []

        # current dp
        if self.list_points:
            curnp = self.list_points[0].get_position()
            for pts in range(1,len(self.list_points)):
                curnp = np.vstack((curnp,self.list_points[pts].get_position()))
            curnp = curnp.reshape(-1,8)

        # check xdim
        if labnp.shape[0] != curnp.shape[0]:
            return False

        # check content
        for l in curnp:
            if l not in labnp:
                return False
        """
        return True


def genWindow(path):
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 또는 'Windows', 'WindowsVista', 'Mac', ...
    
    qss = """
    QWidget {
        background-color: #232629;
        color: #f0f0f0;
        font-weight: bold;
    }
    QPushButton {
        background-color: #ff69b4;
        color: white;
        border-radius: 5px;
        padding: 5px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #2080E1;
    }
    QPushButton:pressed {
        background-color: #c2185b;
        color: white;
    }
    """
    app.setStyleSheet(qss)
    
    window = Window(path)
    window.showMaximized()
    window.show()
    sys.exit(app.exec_())
