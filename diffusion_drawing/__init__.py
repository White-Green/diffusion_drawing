from .diffusion_drawing import *
from krita import *

Krita.instance().addDockWidgetFactory(DockWidgetFactory("diffusion-drawing-docker", DockWidgetFactoryBase.DockRight, DiffusionDrawingDocker))
