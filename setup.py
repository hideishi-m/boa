# -*- coding: utf-8 -*-

import subprocess
from collections.abc import Generator
from pathlib import Path
from setuptools import setup


def generate_mo_files(pkgname: str) -> Generator[str, None, None]:
    """
    poファイルからmoファイルを生成して、パスを返す

    Args:
        pkgname (str): パッケージ名

    Yields:
        str: moファイルのパッケージからの相対パスの文字列
    """
    root_path = Path(__file__).parent / pkgname
    for po_path in root_path.glob('locale/**/*.po'):
        mo_path = po_path.with_suffix('.mo')
        subprocess.run([
            'msgfmt',
            '--output', str(mo_path),
            '--check-format',
            '--check-domain',
            str(po_path)
        ], check=True)
        yield str(mo_path.relative_to(root_path))


def create_zipapp(src: str, dst: str, entry_point: str) -> None:
    """
    whlファイルからpyzファイルを生成する

    Args:
        src (str): whlファイルのディレクトリ
        dst (str): パッケージ名
        entry_points (str): エントリーポイント
    """
    from distutils import log
    from tempfile import NamedTemporaryFile
    from zipapp import create_archive
    from zipfile import ZipFile

    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve().with_suffix('.pyz')

    whl_paths = tuple(src_path.glob('*.whl'))
    if not whl_paths:
        return

    log.info('running create_zipapp')
    with NamedTemporaryFile(delete_on_close=False) as tempfile:
        log.info('creating %r' % tempfile.name)
        with ZipFile(tempfile, 'w') as pyzip:
            log.info('adding %r' % '__init__.py')
            pyzip.writestr('__init__.py', '')
            log.info('adding %r from %r' % ('__main__.py', entry_point))
            pyzip.writestr('__main__.py', Path(entry_point).read_bytes())
            for whl_path in whl_paths:
                log.info('adding %r' % whl_path.name)
                with ZipFile(whl_path, 'r') as whl_zip:
                    for member in whl_zip.infolist():
                        pyzip.writestr(
                            member.filename, whl_zip.read(member))
            pyzip.close()
        create_archive(
            tempfile.name, dst_path, '/usr/bin/env python3.12')
        dst_path.chmod(0o755)
        log.info('writing %r' % Path(dst).with_suffix('.pyz').name)


setup(package_data={
    'boa': list(generate_mo_files('boa'))
})

create_zipapp('dist', 'boa', 'boa/__main__.py')
