#!/bin/sh


function pot() {
	local domain=$1
	local outdir=boa/locale
	local lang=ja_JP
	shift
	xgettext \
		--output=${domain}.pot \
		--output-dir=${outdir} \
		--language=Python \
		--from-code=UTF-8 \
		--strict \
		--sort-output \
		--no-wrap \
		--package-name=${domain} \
		$@

	if [ -f ${outdir}/${domain}.po ]; then
		msgmerge \
			--update \
			--strict \
			--sort-output \
			--no-fuzzy-matching \
			--no-wrap \
			--verbose \
			${outdir}/${domain}.po ${outdir}/${domain}.pot
		if [ "$1" == '--obsolete' ]; then
			msgattrib \
				--no-obsolete \
				--output-file=${outdir}/${domain}.po \
				${outdir}/${domain}.po
		fi
	else
		cp ${outdir}/${domain}.pot ${outdir}/${domain}.po
	fi

	msgfmt \
		--output=${outdir}/${lang}/LC_MESSAGES/${domain}.mo \
		--check-format \
		--check-domain \
		${outdir}/${domain}.po
}

pot boa boa/*.py
pot argparse /opt/kusanagi/python312/lib64/python3.12/argparse.py
