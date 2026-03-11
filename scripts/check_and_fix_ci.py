#!/usr/bin/env python3
"""Check GitHub CI status and auto-fix common issues."""

import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run shell command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    return result.returncode, result.stdout, result.stderr


def get_ci_status():
    """Get latest CI workflow status using gh CLI."""
    print("🔍 Checking GitHub CI status...")
    
    cmd = 'gh run list --limit 1 --json name,status,conclusion,url,workflowDatabaseId'
    rc, stdout, stderr = run_command(cmd)
    
    if rc != 0:
        print(f"❌ Failed to get CI status: {stderr}")
        return None
    
    try:
        runs = json.loads(stdout)
        if runs:
            return runs[0]
    except json.JSONDecodeError:
        print("❌ Failed to parse CI status")
    
    return None


def run_lint_check():
    """Run ruff lint check locally."""
    print("🔍 Running lint check...")
    rc, stdout, stderr = run_command('ruff check src/')
    
    if rc == 0:
        print("✅ Lint check passed")
        return True, []
    
    print("❌ Lint errors found:")
    print(stdout)
    return False, stdout.split('\n')


def auto_fix_lint():
    """Auto-fix lint issues."""
    print("🔧 Attempting to auto-fix lint issues...")
    
    rc, stdout, stderr = run_command('ruff check --fix src/')
    
    if rc == 0:
        print("✅ All lint issues fixed")
        return True
    else:
        print("⚠️ Some lint issues couldn't be auto-fixed")
        return False


def fix_trailing_whitespace():
    """Fix trailing whitespace in Python files."""
    print("🔧 Fixing trailing whitespace...")
    
    src_dir = Path('src')
    fixed_count = 0
    
    for py_file in src_dir.rglob('*.py'):
        content = py_file.read_text(encoding='utf-8')
        lines = [line.rstrip() for line in content.split('\n')]
        new_content = '\n'.join(lines)
        
        if content != new_content:
            py_file.write_text(new_content, encoding='utf-8')
            fixed_count += 1
    
    print(f"✅ Fixed {fixed_count} files")
    return fixed_count > 0


def format_with_black():
    """Format code with black."""
    print("🔧 Formatting with black...")
    rc, stdout, stderr = run_command('black src/ tests/')
    
    if rc == 0:
        print("✅ Code formatted")
        return True
    else:
        print("❌ Formatting failed")
        return False


def commit_and_push(message):
    """Commit and push changes."""
    print(f"📤 Committing: {message}")
    
    commands = [
        'git add -A',
        f'git commit -m "{message}"',
        'git push'
    ]
    
    for cmd in commands:
        rc, stdout, stderr = run_command(cmd)
        if rc != 0:
            print(f"❌ Command failed: {cmd}")
            return False
    
    print("✅ Changes pushed")
    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("🔧 CI Check and Auto-Fix Tool")
    print("=" * 60)
    
    # Check CI status
    ci_status = get_ci_status()
    if ci_status:
        print(f"\n📊 Latest CI: {ci_status.get('name')}")
        print(f"   Status: {ci_status.get('conclusion')}")
        
        if ci_status.get('conclusion') == 'success':
            print("\n✅ CI is passing! No fixes needed.")
            return 0
    
    # Auto-fix routine
    fixes_applied = False
    
    if fix_trailing_whitespace():
        fixes_applied = True
    
    lint_passed, _ = run_lint_check()
    if not lint_passed:
        if auto_fix_lint():
            fixes_applied = True
    
    if format_with_black():
        fixes_applied = True
    
    # Check for changes
    rc, stdout, _ = run_command('git status --porcelain')
    if stdout.strip() and fixes_applied:
        commit_and_push("ci: Auto-fix lint and formatting issues")
        print("\n🚀 Changes pushed! CI will re-run.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
