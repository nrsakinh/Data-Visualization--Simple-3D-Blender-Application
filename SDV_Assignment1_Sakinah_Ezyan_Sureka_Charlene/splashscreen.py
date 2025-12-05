# splash_glow_main.py
import sys, math, time
from main import MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import os

class NeonSplash(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    def __init__(self, duration_ms=5000, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.SplashScreen)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.resize(800, 600)

        self.duration_ms = duration_ms
        self._start = time.time()
        self._progress = 0

        # Colors
        self.bg = QtGui.QColor("#000000")
        self.text_color = QtGui.QColor("#DCE4F0")

        # Load GIF animation
        self.setup_gif_animation()

        # Timer for progress updates
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(16)

        # Auto close splash
        QtCore.QTimer.singleShot(self.duration_ms, self._finish)

    def setup_gif_animation(self):
        """Setup GIF animation for the splash screen"""
        try:
            # Prefer project-relative path first
            gif_path = os.path.join(os.path.dirname(__file__), "Icons", "splashscreen.gif")
            if not os.path.isfile(gif_path):
                gif_path = r"D:\SDV2025\project_env1\SDV_Assignment1_Sakinah_Ezyan_Sureka_Charlene\Icons\splashscreen.gif"  # fallback

            self.movie = QtGui.QMovie(gif_path)
            if not self.movie.isValid():
                print("Error: Invalid GIF file")
                self.fallback_to_static_image(gif_path)
                return

            self.movie.frameChanged.connect(self.update)
            self.movie.start()
            print("GIF animation started successfully")
        except Exception as e:
            print(f"Error loading GIF: {e}")
            self.fallback_to_static_image(gif_path if 'gif_path' in locals() else None)

    def fallback_to_static_image(self, path=None):
        """Fallback to static image if GIF fails to load"""
        try:
            img_path = path or os.path.join(os.path.dirname(__file__), "Icons", "splashscreen.gif")
            self.logo = QtGui.QPixmap(img_path)
            self.movie = None
            print("Using static image as fallback")
        except:
            self.logo = QtGui.QPixmap()
            self.movie = None
            print("No image available")

    def _update(self):
        elapsed = time.time() - self._start
        self._progress = min(100, int((elapsed / (self.duration_ms/1000)) * 100))
        self.update()

    def _finish(self):
        if self._timer.isActive():
            self._timer.stop()
        if self.movie and self.movie.state() == QtGui.QMovie.Running:
            self.movie.stop()
        self.finished.emit()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        # Background
        p.fillRect(self.rect(), self.bg)

        cx = self.width() / 2
        cy = self.height() * 0.28

        # --- Draw ANIMATED GIF or STATIC LOGO ---
        if self.movie and self.movie.isValid():
            # Draw current GIF frame
            current_pixmap = self.movie.currentPixmap()
            if not current_pixmap.isNull():
                target_width = 600
                target_height = 600

                scaled_logo = current_pixmap.scaled(
                    target_width,
                    target_height,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )

                w = scaled_logo.width()
                h = scaled_logo.height()

                cx = self.width() / 2
                cy = self.height() * 0.34  # vertical position

                p.drawPixmap(int(cx - w/2), int(cy - h/2), scaled_logo)
                
        elif hasattr(self, 'logo') and not self.logo.isNull():
            # Draw static logo as fallback
            target_width = 400
            target_height = 400

            scaled_logo = self.logo.scaled(
                target_width,
                target_height,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )

            w = scaled_logo.width()
            h = scaled_logo.height()

            cx = self.width() / 2
            cy = self.height() * 0.34  # vertical position

            p.drawPixmap(int(cx - w/2), int(cy - h/2), scaled_logo)

        # --- Titles ---
        font1 = QtGui.QFont("Montserrat", 26, QtGui.QFont.Bold)
        font2 = QtGui.QFont("Segoe UI", 11)
        p.setPen(self.text_color)

        # Title
        p.setFont(font1)
        p.drawText(self.rect().adjusted(0, int(self.height()*0.65), 0, 0),
                   QtCore.Qt.AlignHCenter, "MY BLENDER")

        # Subtitle
        p.setFont(font2)
        p.drawText(self.rect().adjusted(0, int(self.height()*0.78), 0, 0),
                   QtCore.Qt.AlignHCenter, "powered by VTK")

        # --- Progress Bar ---
        bar_w = int(self.width() * 0.5)
        bar_h = 10
        bar_x = int((self.width() - bar_w) / 2)
        bar_y = int(self.height() * 0.87)

        # Track
        p.setBrush(QtGui.QColor(0,0,0,0))
        p.setPen(QtCore.Qt.NoPen)
        p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 5, 5)

        # Fill
        fill = QtGui.QLinearGradient(bar_x, bar_y, bar_x + bar_w, bar_y)
        fill.setColorAt(0, QtGui.QColor("#FFD700"))
        fill.setColorAt(1, QtGui.QColor("#FF8C00"))
        p.setBrush(fill)
        p.drawRoundedRect(bar_x, bar_y,
                          int(bar_w * (self._progress/100)), bar_h, 5, 5)

    def closeEvent(self, event):
        """Clean up when closing"""
        if hasattr(self, 'movie') and self.movie:
            self.movie.stop()
        super().closeEvent(event)


class Launcher(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.app = QtWidgets.QApplication(sys.argv)
        self._anim_refs = {}
        self.splash = NeonSplash()
        self.splash.finished.connect(self.launch)

        screen = self.app.primaryScreen().geometry()
        self.splash.move(screen.center().x() - self.splash.width()//2,
                         screen.center().y() - self.splash.height()//2)
        self.splash.show()

    @QtCore.pyqtSlot()
    def launch(self):
        self.win = MainWindow()  # assume modified not to auto show
        # If it still auto-shows, immediately hide to suppress first frame:
        if self.win.isVisible():
            self.win.hide()

        self.win.setWindowOpacity(0.0)  # set before show
        self.win.show()

        grp = QtCore.QParallelAnimationGroup(self)

        fade_in = QtCore.QPropertyAnimation(self.win, b"windowOpacity")
        fade_in.setDuration(600)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        grp.addAnimation(fade_in)

        fade_out = QtCore.QPropertyAnimation(self.splash, b"windowOpacity")
        fade_out.setDuration(600)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        grp.addAnimation(fade_out)

        def _cleanup():
            self.splash.close()
        grp.finished.connect(_cleanup)

        self._track_anim(self.win, fade_in)
        self._track_anim(self.splash, fade_out)
        grp.start()

    def _track_anim(self, w, anim):
        self._anim_refs.setdefault(w, []).append(anim)
        def _cleanup():
            lst = self._anim_refs.get(w, [])
            if anim in lst:
                lst.remove(anim)
        anim.finished.connect(_cleanup)
        
    def fade_in(self, w):
        anim = QtCore.QPropertyAnimation(w, b"windowOpacity", self)
        anim.setDuration(600)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        self._track_anim(w, anim)
        anim.start()

    def fade_out(self, w):
        # Animate top-level window opacity (more reliable than QGraphicsOpacityEffect here)
        try:
            w.setWindowOpacity(1.0)
        except Exception:
            pass
        anim = QtCore.QPropertyAnimation(w, b"windowOpacity", self)
        anim.setDuration(600)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.finished.connect(w.close)
        self._track_anim(w, anim)
        anim.start()

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    Launcher().run()