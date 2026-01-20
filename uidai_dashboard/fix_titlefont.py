import re
from pathlib import Path

files = [
    'dashboards/national.py',
    'dashboards/enrolment.py', 
    'dashboards/biometric.py',
    'dashboards/updates.py'
]

for file_path in files:
    path = Path(file_path)
    if path.exists():
        content = path.read_text(encoding='utf-8')
        # Replace titlefont with title dict containing font
        # Pattern: title='X', titlefont=dict(color='Y')
        # Becomes: title=dict(text='X', font=dict(color='Y'))
        new_content = re.sub(
            r"title='([^']+)', titlefont=dict\(color='([^']+)'\)",
            r"title=dict(text='\1', font=dict(color='\2'))",
            content
        )
        path.write_text(new_content, encoding='utf-8')
        print(f'Fixed: {file_path}')
    else:
        print(f'Not found: {file_path}')

print('Done!')
