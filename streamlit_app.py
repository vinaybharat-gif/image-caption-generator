import sys
try:
    import legacy_cgi
    sys.modules['cgi'] = legacy_cgi
except ImportError:
    pass

# Force execution of the actual application code
import runpy
runpy.run_path('app.py', run_name='__main__')