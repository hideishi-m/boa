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
    import zipapp
    from distutils import log
    from tempfile import NamedTemporaryFile
    from zipfile import ZipFile

    log.info('running create_zipapp')
    with NamedTemporaryFile(delete=False) as tempfile:
        try:
            log.info('creating %r' % tempfile.name)
            with ZipFile(tempfile, 'w') as pyzip:
                log.info('adding %r' % '__init__.py')
                pyzip.writestr('__init__.py', '')
                log.info('adding %r from %r' % ('__main__.py', entry_point))
                pyzip.writestr('__main__.py', Path(entry_point).read_bytes())
                for src_path in Path(src).glob('*.whl'):
                    log.info('adding %r' % src_path.name)
                    with ZipFile(src_path, 'r') as src_zip:
                        for member in src_zip.infolist():
                            pyzip.writestr(
                                member.filename, src_zip.read(member))
                pyzip.close()
            zipapp.create_archive(
                tempfile.name,
                Path(dst).with_suffix('.pyz'),
                '/usr/bin/env python3.11')
            log.info('writing %r' % Path(dst).with_suffix('.pyz').name)
        finally:
            if Path(tempfile.name).exists():
                Path(tempfile.name).unlink()
                log.info('removing %r' % tempfile.name)
    Path(dst).with_suffix('.pyz').chmod(0o755)


setup(package_data={
    'boa': list(generate_mo_files('boa'))
})

create_zipapp('dist', 'boa', 'boa/__main__.py')
