"""Helper script to delete low-risk artifacts (invoked by the assistant).
Run only in CI/dev workspace.
"""
import shutil
import os
root = '/workspaces/Minor-in-AI-Sem2-project'
paths = [
    os.path.join(root, '__pycache__'),
    os.path.join(root, '.pytest_cache'),
    os.path.join(root, '.venv.bak'),
    os.path.join(root, 'artifacts'),
    os.path.join(root, 'streamlit.log'),
]
for p in paths:
    if os.path.exists(p):
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
                print('removed dir', p)
            else:
                os.remove(p)
                print('removed file', p)
        except Exception as e:
            print('failed', p, e)
    else:
        print('not found', p)
print('done')
