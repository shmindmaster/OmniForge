import sys, os
root = os.path.dirname(__file__)
app_path = os.path.join(root, 'app')
if app_path not in sys.path:
    sys.path.insert(0, app_path)
