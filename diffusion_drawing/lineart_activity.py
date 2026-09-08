"""Observe user input without consuming Krita's mouse, tablet or shortcut events."""

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QApplication


class LineartActivity(QObject):
    def __init__(self, docker, changed):
        super().__init__(docker)
        self.docker = docker
        self.changed = changed
        self.enabled = False
        self._pressed = set()
        self._inactive = False
        QApplication.instance().installEventFilter(self)

    @property
    def busy(self):
        return self._inactive or bool(self._pressed)

    def set_enabled(self, enabled):
        self.enabled = enabled
        self._pressed.clear()
        self._inactive = QApplication.applicationState() != Qt.ApplicationActive

    def _in_document_window(self, receiver):
        # Include menus and layer/undo actions, but leave this plugin's controls
        # to their own signals. No dependency on Krita's private canvas classes.
        window = self.docker.parentWidget()
        if receiver == window:
            # Ignore events propagated from an excluded child to the main window.
            # Input on the canvas and other child widgets was already observed.
            return False
        while receiver is not None:
            if receiver == self.docker:
                return False
            if receiver == window:
                return True
            receiver = receiver.parent()
        return False

    def eventFilter(self, receiver, event):
        if not self.enabled:
            return False
        kind = event.type()
        if kind == QEvent.ApplicationDeactivate:
            self._inactive = True
            self._pressed.clear()
            self.changed()
        elif kind == QEvent.ApplicationActivate:
            self._inactive = False
            self.changed()
        elif kind in (QEvent.MouseButtonRelease, QEvent.TabletRelease):
            source = "tablet" if kind == QEvent.TabletRelease else "mouse"
            was_pressed = source in self._pressed
            if not event.buttons():
                self._pressed.discard(source)
            if was_pressed or self._in_document_window(receiver):
                self.changed()
        elif kind in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick, QEvent.TabletPress,
                      QEvent.MouseMove, QEvent.TabletMove):
            if self._in_document_window(receiver):
                tablet = kind in (QEvent.TabletPress, QEvent.TabletMove)
                source = "tablet" if tablet else "mouse"
                contact = bool(event.buttons()) or (tablet and event.pressure() > 0)
                if contact:
                    self._pressed.add(source)
                    self.changed()
                elif source in self._pressed:
                    # Recover if the release was delivered outside this window.
                    self._pressed.discard(source)
                    self.changed()
        elif kind in (QEvent.KeyPress, QEvent.KeyRelease, QEvent.ShortcutOverride,
                      QEvent.Shortcut, QEvent.InputMethod, QEvent.Drop):
            if self._in_document_window(receiver):
                self.changed()
        # Hover, paint, projection refresh and timer events deliberately do nothing.
        return False
