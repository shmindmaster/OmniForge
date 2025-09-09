import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app_dir = os.path.join(root, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
