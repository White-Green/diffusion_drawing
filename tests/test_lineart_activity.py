import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QTabletEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget


spec = importlib.util.spec_from_file_location(
    "lineart_activity_test", Path(__file__).resolve().parents[1] / "diffusion_drawing/lineart_activity.py")
activity_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activity_module)


class ActivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = QMainWindow()
        self.canvas = QWidget(self.window)
        self.docker = QWidget(self.window)
        self.control = QWidget(self.docker)
        self.changed = Mock()
        self.activity = activity_module.LineartActivity(self.docker, self.changed)
        self.activity.set_enabled(True)
        self.send(self.app, QEvent(QEvent.ApplicationActivate))
        self.changed.reset_mock()

    def tearDown(self):
        self.app.removeEventFilter(self.activity)
        self.window.close()

    def send(self, receiver, event):
        QApplication.sendEvent(receiver, event)

    def mouse(self, kind, buttons=Qt.NoButton, receiver=None):
        button = Qt.NoButton if kind == QEvent.MouseMove else Qt.LeftButton
        self.send(receiver or self.canvas, QMouseEvent(kind, QPointF(4, 4), button, buttons, Qt.NoModifier))

    def tablet(self, kind, pressure, buttons=Qt.NoButton):
        button = Qt.NoButton if kind == QEvent.TabletMove else Qt.LeftButton
        self.send(self.canvas, QTabletEvent(
            kind, QPointF(4, 4), QPointF(4, 4), QTabletEvent.Stylus, QTabletEvent.Pen,
            pressure, 0, 0, 0.0, 0.0, 0, Qt.NoModifier, 1, button, buttons))

    def test_mouse_press_remains_busy_until_release(self):
        self.mouse(QEvent.MouseButtonPress, Qt.LeftButton)
        self.assertTrue(self.activity.busy)
        self.mouse(QEvent.MouseMove, Qt.LeftButton)
        self.assertTrue(self.activity.busy)
        self.mouse(QEvent.MouseButtonRelease)
        self.assertFalse(self.activity.busy)
        self.assertGreaterEqual(self.changed.call_count, 3)

    def test_tablet_contact_waits_for_release_but_hover_is_ignored(self):
        self.tablet(QEvent.TabletMove, 0.0)
        self.changed.assert_not_called()
        self.tablet(QEvent.TabletPress, 0.5, Qt.LeftButton)
        self.tablet(QEvent.TabletMove, 0.5, Qt.LeftButton)
        self.assertTrue(self.activity.busy)
        self.tablet(QEvent.TabletRelease, 0.0)
        self.assertFalse(self.activity.busy)
        count = self.changed.call_count
        self.tablet(QEvent.TabletMove, 0.0)
        self.assertEqual(self.changed.call_count, count)

    def test_tablet_and_synthesized_mouse_releases_both_finish(self):
        self.tablet(QEvent.TabletPress, 0.5, Qt.LeftButton)
        self.mouse(QEvent.MouseButtonPress, Qt.LeftButton)
        self.tablet(QEvent.TabletRelease, 0.0)
        self.assertTrue(self.activity.busy)
        self.mouse(QEvent.MouseButtonRelease)
        self.assertFalse(self.activity.busy)

    def test_hover_and_output_repaint_do_not_request_generation(self):
        self.mouse(QEvent.MouseMove)
        self.send(self.canvas, QEvent(QEvent.UpdateRequest))
        self.changed.assert_not_called()

    def test_undo_shortcut_and_layer_actions_request_check(self):
        self.send(self.canvas, QKeyEvent(QEvent.ShortcutOverride, Qt.Key_Z, Qt.ControlModifier))
        self.assertTrue(self.changed.called)
        self.changed.reset_mock()
        layer_control = QWidget(self.window)
        self.mouse(QEvent.MouseButtonRelease, receiver=layer_control)
        self.assertTrue(self.changed.called)

    def test_other_window_and_plugin_controls_are_ignored(self):
        other = QWidget()
        self.mouse(QEvent.MouseButtonPress, Qt.LeftButton, other)
        self.mouse(QEvent.MouseButtonRelease, receiver=other)
        self.mouse(QEvent.MouseButtonPress, Qt.LeftButton, self.control)
        self.mouse(QEvent.MouseButtonRelease, receiver=self.control)
        self.changed.assert_not_called()
        self.assertFalse(self.activity.busy)
        other.close()

    def test_release_outside_window_recovers_and_disable_clears_stroke(self):
        self.mouse(QEvent.MouseButtonPress, Qt.LeftButton)
        other = QWidget()
        self.mouse(QEvent.MouseButtonRelease, receiver=other)
        self.assertFalse(self.activity.busy)
        self.mouse(QEvent.MouseButtonPress, Qt.LeftButton)
        self.activity.set_enabled(False)
        self.changed.reset_mock()
        self.mouse(QEvent.MouseButtonRelease)
        self.changed.assert_not_called()
        other.close()

    def test_application_deactivate_pauses_and_recovers_lost_release(self):
        self.mouse(QEvent.MouseButtonPress, Qt.LeftButton)
        self.send(self.app, QEvent(QEvent.ApplicationDeactivate))
        self.assertTrue(self.activity.busy)
        self.send(self.app, QEvent(QEvent.ApplicationActivate))
        self.assertFalse(self.activity.busy)

    def test_observer_does_not_consume_input(self):
        event = QMouseEvent(QEvent.MouseButtonPress, QPointF(4, 4),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        self.assertFalse(self.activity.eventFilter(self.canvas, event))


if __name__ == "__main__":
    unittest.main()
