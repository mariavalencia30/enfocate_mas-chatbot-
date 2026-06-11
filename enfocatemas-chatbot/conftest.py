import os
import sys

# Agregar la raíz del proyecto al path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root) 