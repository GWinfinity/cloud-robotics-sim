#!/usr/bin/env python3
"""
Project Migration Tool

自动化将现有项目迁移为 Plugin 的工具。

Usage:
    python migrate_project.py \
        --source /path/to/original/project \
        --name my_controller \
        --category controllers \
        --target ./plugins/controllers/my_controller
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional


class MigrationTool:
    """项目迁移工具"""
    
    def __init__(self, source: str, name: str, category: str, target: Optional[str] = None):
        self.source_path = Path(source).resolve()
        self.plugin_name = name
        self.category = category
        
        if target:
            self.target_path = Path(target).resolve()
        else:
            self.target_path = Path(__file__).parent / category / name
        
        self.files_to_migrate: List[Path] = []
    
    def analyze(self) -> dict:
        """分析源项目结构"""
        print(f"\nAnalyzing source project: {self.source_path}")
        
        analysis = {
            'py_files': list(self.source_path.rglob('*.py')),
            'config_files': list(self.source_path.rglob('*.yaml')) + list(self.source_path.rglob('*.yml')),
            'readme': None,
            'requirements': None
        }
        
        # 查找 README
        for readme_name in ['README.md', 'README.rst', 'readme.md']:
            readme_path = self.source_path / readme_name
            if readme_path.exists():
                analysis['readme'] = readme_path
                break
        
        # 查找 requirements
        for req_name in ['requirements.txt', 'requirements.yaml']:
            req_path = self.source_path / req_name
            if req_path.exists():
                analysis['requirements'] = req_path
                break
        
        print(f"  Found {len(analysis['py_files'])} Python files")
        print(f"  Found {len(analysis['config_files'])} config files")
        print(f"  README: {'Yes' if analysis['readme'] else 'No'}")
        print(f"  Requirements: {'Yes' if analysis['requirements'] else 'No'}")
        
        return analysis
    
    def prompt_files(self, analysis: dict) -> List[Path]:
        """交互式选择要迁移的文件"""
        print("\n" + "=" * 60)
        print("Select files to migrate (core implementation only)")
        print("=" * 60)
        
        candidates = []
        
        # Python 文件
        print("\nPython files:")
        for i, f in enumerate(analysis['py_files'][:20], 1):  # 限制显示数量
            rel_path = f.relative_to(self.source_path)
            print(f"  [{i}] {rel_path}")
            candidates.append(f)
        
        if len(analysis['py_files']) > 20:
            print(f"  ... and {len(analysis['py_files']) - 20} more files")
        
        print("\nEnter file numbers to migrate (comma-separated, e.g., 1,3,5)")
        print("Or enter 'all' to migrate all files")
        print("Or press Enter to skip file selection")
        
        try:
            selection = input("> ").strip()
        except KeyboardInterrupt:
            print("\nCancelled")
            sys.exit(1)
        
        if selection.lower() == 'all':
            return analysis['py_files']
        elif not selection:
            return []
        
        selected = []
        for num in selection.split(','):
            try:
                idx = int(num.strip()) - 1
                if 0 <= idx < len(candidates):
                    selected.append(candidates[idx])
            except ValueError:
                pass
        
        return selected
    
    def create_structure(self):
        """创建 Plugin 目录结构"""
        print(f"\nCreating plugin structure at: {self.target_path}")
        
        # 创建目录
        dirs = ['core', 'configs', 'examples', 'tests']
        for d in dirs:
            (self.target_path / d).mkdir(parents=True, exist_ok=True)
            print(f"  Created: {d}/")
    
    def copy_files(self, files: List[Path]):
        """复制文件到目标目录"""
        print("\nCopying files...")
        
        for src_file in files:
            # 保留相对目录结构在 core/ 下
            try:
                rel_path = src_file.relative_to(self.source_path)
            except ValueError:
                rel_path = src_file.name
            
            dst_file = self.target_path / 'core' / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src_file, dst_file)
            print(f"  Copied: {rel_path}")
    
    def create_plugin_yaml(self, description: str = ""):
        """创建 plugin.yaml"""
        yaml_content = f"""name: {self.plugin_name}
version: 0.1.0
description: {description or f'{self.plugin_name} controller'}

category: {self.category}
source_project: {self.source_path.name}

author:
  name: Migration Tool
  email: migration@example.com

dependencies:
  - numpy>=1.20
  - genesis-world>=0.2.0

exports: []

tags: []
"""
        
        yaml_path = self.target_path / 'plugin.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\nCreated: plugin.yaml")
    
    def create_init_py(self):
        """创建 __init__.py"""
        init_content = f'''\"\"\"
{self.plugin_name} Plugin

来源: {self.source_path.name}
核心实现: TODO: Add description

TODO: Add detailed description
\"\"\"

__version__ = "0.1.0"
__source__ = "{self.source_path.name}"

# TODO: Import main classes
# from .core.module import MyClass

# __all__ = ['MyClass']
'''
        
        init_path = self.target_path / '__init__.py'
        with open(init_path, 'w') as f:
            f.write(init_content)
        
        print(f"Created: __init__.py")
    
    def create_readme(self):
        """创建 README.md"""
        readme_content = f'''# {self.plugin_name} Plugin

来源: [{self.source_path.name}]({self.source_path})

## 核心功能

TODO: 描述这个插件的核心功能

## 快速开始

```python
from cloud_robotics_sim.plugins.{self.category}.{self.plugin_name} import MyClass

# TODO: 使用示例
```

## 算法原理

TODO: 简要描述算法原理

## 配置参数

TODO: 列出关键参数

## 示例

见 [examples/](examples/) 目录。

## Changelog

- {self._today()}: 从 {self.source_path.name} 迁移
'''
        
        readme_path = self.target_path / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"Created: README.md")
    
    def create_ab_test_template(self):
        """创建 A/B 测试模板"""
        test_content = f'''"""
A/B Test for {self.plugin_name} Migration

对比旧版实现和 Plugin 实现
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cloud_robotics_sim.core.ab_test_framework import ABTestRunner


def main():
    # TODO: 导入旧版实现
    # sys.path.insert(0, '/path/to/original')
    # from original import OldImplementation
    
    # TODO: 导入新版 Plugin
    from cloud_robotics_sim.plugins.{self.category}.{self.plugin_name} import NewImplementation
    
    # 创建 A/B 测试运行器
    runner = ABTestRunner(
        variant_a_name="original",
        variant_a_fn=OldImplementation(),
        variant_b_name="plugin",
        variant_b_fn=NewImplementation(),
        output_dir="ab_test_results/{self.plugin_name}",
        warmup_steps=10
    )
    
    # TODO: 运行测试
    # for episode in range(1000):
    #     ...
    
    # 生成报告
    print(runner.generate_report())
    runner.save_report()


if __name__ == '__main__':
    main()
'''
        
        test_path = self.target_path / 'examples' / 'ab_test.py'
        with open(test_path, 'w') as f:
            f.write(test_content)
        
        print(f"Created: examples/ab_test.py")
    
    def _today(self) -> str:
        """获取今天的日期"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    
    def run(self):
        """执行完整迁移流程"""
        print("=" * 60)
        print("Plugin Migration Tool")
        print("=" * 60)
        
        # 1. 分析源项目
        analysis = self.analyze()
        
        # 2. 选择文件
        files = self.prompt_files(analysis)
        
        if not files:
            print("\nNo files selected. Exiting.")
            return
        
        # 3. 创建结构
        self.create_structure()
        
        # 4. 复制文件
        self.copy_files(files)
        
        # 5. 创建元文件
        self.create_plugin_yaml()
        self.create_init_py()
        self.create_readme()
        self.create_ab_test_template()
        
        # 6. 完成
        print("\n" + "=" * 60)
        print("Migration Complete!")
        print("=" * 60)
        print(f"\nPlugin location: {self.target_path}")
        print("\nNext steps:")
        print("  1. Update plugin.yaml with correct exports and dependencies")
        print("  2. Update __init__.py to export main classes")
        print("  3. Complete README.md with algorithm description")
        print("  4. Run A/B tests: python examples/ab_test.py")
        print("  5. Follow MIGRATION_GUIDE.md for gradual rollout")


def main():
    parser = argparse.ArgumentParser(description='Migrate project to Plugin')
    parser.add_argument('--source', '-s', required=True, help='Source project path')
    parser.add_argument('--name', '-n', required=True, help='Plugin name')
    parser.add_argument('--category', '-c', required=True, 
                       choices=['controllers', 'envs', 'predictors', 'sim2real'],
                       help='Plugin category')
    parser.add_argument('--target', '-t', help='Target path (optional)')
    
    args = parser.parse_args()
    
    tool = MigrationTool(
        source=args.source,
        name=args.name,
        category=args.category,
        target=args.target
    )
    
    tool.run()


if __name__ == '__main__':
    main()
