#!/bin/sh

STDLIB=`python3.12 -c 'import sysconfig; print(sysconfig.get_path("stdlib"))'`

pot() {
	local domain=$1
	local outdir=boa/locale
	local lang=ja_JP
	shift
	xgettext \
		--output=${domain}.pot \
		--output-dir=${outdir}/${lang}/LC_MESSAGES \
		--language=Python \
		--from-code=UTF-8 \
		--strict \
		--sort-output \
		--no-wrap \
		--package-name=${domain} \
		$@

	if [ -f ${outdir}/${lang}/LC_MESSAGES/${domain}.po ]; then
		msgmerge \
			--update \
			--strict \
			--sort-output \
			--no-fuzzy-matching \
			--no-wrap \
			--verbose \
			${outdir}/${lang}/LC_MESSAGES/${domain}.po \
			${outdir}/${lang}/LC_MESSAGES/${domain}.pot
		if [ "$1" == '--obsolete' ]; then
			msgattrib \
				--no-obsolete \
				--output-file=${outdir}/${lang}/LC_MESSAGES/${domain}.po \
				${outdir}/${lang}/LC_MESSAGES/${domain}.po
		fi
	else
		cp \
			${outdir}/${lang}/LC_MESSAGES/${domain}.pot \
			${outdir}/${lang}/LC_MESSAGES/${domain}.po
	fi

	msgfmt \
		--output=${outdir}/${lang}/LC_MESSAGES/${domain}.mo \
		--check-format \
		--check-domain \
		${outdir}/${lang}/LC_MESSAGES/${domain}.po
}

pot boa boa/*.py
pot argparse ${STDLIB}/argparse.py
