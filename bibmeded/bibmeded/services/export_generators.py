"""Pure-function generators for the export endpoints.

Each function returns its full content as text/bytes so callers (router endpoints, the
bundle endpoint, tests) can compose them without going through Starlette `StreamingResponse`.
"""

import csv
import io
import json
import re
import zipfile
from datetime import date

from bibmeded.models import Publication
from bibmeded.models.keyword import KeywordType
from bibmeded.models.methodology import MethodologyStep
from bibmeded.services.prisma import ExclusionSummary, compute_counts, render_svg, summarize_exclusions


_PHASE_LABELS = {
    "search": "SEARCH STRATEGY",
    "fetch": "DATA COLLECTION",
    "dedup": "DEDUPLICATION",
    "enrichment": "ENRICHMENT",
    "exclusion": "EXCLUSION",
}


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower().strip())
    return re.sub(r"[\s_]+", "-", slug)[:50]


# Leading characters that Excel/Sheets/LibreOffice interpret as the start of a
# formula. Bibliographic fields (titles, author names, journal names, keywords)
# come from external APIs and are attacker-influenced, so any value starting
# with one of these must be neutralized before it's written to a CSV cell —
# otherwise opening the export in a spreadsheet app can execute the "formula"
# (OWASP CSV Injection / Formula Injection).
_CSV_FORMULA_TRIGGER_CHARS = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: str) -> str:
    """Prefix a leading apostrophe onto values that would otherwise be
    interpreted as spreadsheet formulas, per the standard OWASP CSV escaping
    mitigation. The apostrophe forces the cell to be treated as text.
    """
    if value and value[0] in _CSV_FORMULA_TRIGGER_CHARS:
        return "'" + value
    return value


def generate_csv(pubs: list[Publication]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["PMID", "DOI", "Title", "Authors", "Journal", "Year", "Citations", "Keywords", "Abstract"])
    for pub in pubs:
        writer.writerow([
            _csv_safe(pub.pmid or ""),
            _csv_safe(pub.doi or ""),
            _csv_safe(pub.title or ""),
            _csv_safe("; ".join(a.name for a in pub.authors)),
            _csv_safe(pub.journal.name if pub.journal else ""),
            pub.year or "",
            pub.citation_count or 0,
            _csv_safe("; ".join(k.term for k in pub.keywords)),
            _csv_safe((pub.abstract or "").replace("\n", " ")),
        ])
    return output.getvalue()


EXPORT_SCHEMA_VERSION = "1.0"


def _publication_to_dict(pub: Publication) -> dict:
    return {
        "pmid": pub.pmid,
        "doi": pub.doi,
        "title": pub.title,
        "abstract": pub.abstract,
        "year": pub.year,
        "publication_type": pub.publication_type,
        "source_database": pub.source_database,
        "citation_count": pub.citation_count,
        "excluded": pub.excluded,
        "exclusion_reason": pub.exclusion_reason,
        "screening_stage": pub.screening_stage,
        "journal_name": pub.journal.name if pub.journal else None,
        "journal_issn": pub.journal.issn if pub.journal else None,
        "authors": [
            {"name": a.name, "orcid": a.orcid}
            for a in pub.authors
        ],
        "keywords": [k.term for k in pub.keywords],
    }


def generate_json(project_name: str, pubs: list[Publication]) -> str:
    """Bulk JSON export with a versioned schema for downstream pipelines.

    Shape: ``{schema_version, project, generated, count, publications: [...]}``.
    Each publication entry is the dict returned by ``_publication_to_dict``. Encoded
    with ``ensure_ascii=False`` so non-ASCII titles/abstracts (Turkish, Chinese,
    Greek) round-trip cleanly.
    """
    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "project": project_name,
        "generated": date.today().isoformat(),
        "count": len(pubs),
        "publications": [_publication_to_dict(p) for p in pubs],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _ris_safe(value: str) -> str:
    """Strip embedded CR/LF from a field value before writing it into an RIS
    tag line. RIS records are newline-delimited, so an unescaped newline in an
    attacker-influenced field (title/author/journal/keyword) lets a malicious
    record inject arbitrary tag lines — including a fake ``ER  - `` that
    terminates the record early or a fake ``TY  - JOUR`` that starts a new one.
    """
    return re.sub(r"[\r\n]+", " ", value).strip()


def generate_ris(pubs: list[Publication]) -> str:
    lines: list[str] = []
    for pub in pubs:
        lines.append("TY  - JOUR")
        lines.append(f"TI  - {_ris_safe(pub.title)}")
        for author in pub.authors:
            lines.append(f"AU  - {_ris_safe(author.name)}")
        if pub.journal:
            lines.append(f"JO  - {_ris_safe(pub.journal.name)}")
        if pub.year:
            lines.append(f"PY  - {pub.year}")
        if pub.doi:
            lines.append(f"DO  - {_ris_safe(pub.doi)}")
        lines.append(f"AN  - {pub.pmid}")
        if pub.abstract:
            lines.append(f"AB  - {_ris_safe(pub.abstract)}")
        for kw in pub.keywords:
            lines.append(f"KW  - {_ris_safe(kw.term)}")
        lines.append("ER  - ")
        lines.append("")
    return "\n".join(lines)


# Genuine Web of Science plain-text exports start with a UTF-8 BOM. Writing one keeps
# the file in the shape VOSviewer and bibliometrix already parse, and stops tools that
# fall back to the platform code page from misreading non-ASCII names.
_WOS_BOM = "\ufeff"
_WOS_HEADER = ("FN Clarivate Analytics Web of Science", "VR 1.0")
_WOS_CONTINUATION = "   "
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


def _wos_tag(tag: str, values: list[str]) -> list[str]:
    cleaned = [v for v in (_ris_safe(value) for value in values) if v]
    if not cleaned:
        return []
    return [f"{tag} {cleaned[0]}", *(f"{_WOS_CONTINUATION}{v}" for v in cleaned[1:])]


def _wos_name_parts(name: str, source_database: str) -> tuple[str, str]:
    """Split a stored author name into (surname, given names).

    PubMed stores persons as "Surname, Given" and collective authors without a
    comma, so a comma-less PubMed name is kept whole. Other adapters store
    "Given Surname", where the last token is taken as the surname.
    """
    name = " ".join(_ris_safe(name).split())
    if "," in name:
        surname, given = name.split(",", 1)
        return surname.strip(), given.strip()
    if source_database == "pubmed" or " " not in name:
        return name, ""
    given, surname = name.rsplit(" ", 1)
    return surname, given


def _wos_short_name(surname: str, given: str) -> str:
    initials = "".join(token[0].upper() for token in re.split(r"[\s.\-]+", given) if token)
    return f"{surname}, {initials}" if initials else surname


def _wos_full_name(surname: str, given: str) -> str:
    return f"{surname}, {given}" if given else surname


def _wos_addresses(pub: Publication, short_names: list[str]) -> list[str]:
    authors_by_address: dict[str, list[str]] = {}
    for author, short_name in zip(pub.authors, short_names):
        if not short_name:
            continue
        for affiliation in author.affiliations:
            address = _ris_safe(affiliation.name or "")
            if not address:
                continue
            names = authors_by_address.setdefault(address, [])
            if short_name not in names:
                names.append(short_name)
    return [f"[{'; '.join(names)}] {address}" for address, names in authors_by_address.items()]


def _wos_keywords(pub: Publication, kind: KeywordType) -> str:
    terms = (_ris_safe(k.term or "") for k in pub.keywords if k.type == kind)
    return "; ".join(term for term in terms if term)


def _wos_doi_by_source_id(pubs: list[Publication]) -> dict[str, str]:
    return {pub.pmid.lower(): pub.doi.lower() for pub in pubs if pub.pmid and pub.doi}


def _wos_cited_references(pub: Publication, doi_by_source_id: dict[str, str]) -> list[str]:
    """Cited references as WoS ``DOI <doi>`` entries.

    ``external_references`` holds source-native identifiers. A DOI is written as is;
    any other identifier is written only when it names an in-corpus record with a
    DOI, the same matching the citation analysis uses. The rest have no WoS form,
    so ``NR`` carries the full count and the bundle manifest reports the shortfall.
    """
    cited: list[str] = []
    for raw_ref in pub.external_references or []:
        ref = str(raw_ref).strip().lower()
        doi = ref if _DOI_PATTERN.match(ref) else doi_by_source_id.get(ref)
        if doi and f"DOI {doi}" not in cited:
            cited.append(f"DOI {doi}")
    return cited


def wos_reference_coverage(pubs: list[Publication]) -> tuple[int, int]:
    """(cited references written to CR, references the sources reported)."""
    doi_by_source_id = _wos_doi_by_source_id(pubs)
    written = sum(len(_wos_cited_references(pub, doi_by_source_id)) for pub in pubs)
    reported = sum(len(pub.external_references or []) for pub in pubs)
    return written, reported


def _wos_record(pub: Publication, doi_by_source_id: dict[str, str]) -> list[str]:
    name_parts = [_wos_name_parts(author.name or "", pub.source_database) for author in pub.authors]
    short_names = [_wos_short_name(surname, given) for surname, given in name_parts]
    references = pub.external_references or []
    journal = pub.journal
    return [
        "PT J",
        *_wos_tag("AU", short_names),
        *_wos_tag("AF", [_wos_full_name(surname, given) for surname, given in name_parts]),
        *_wos_tag("TI", [pub.title or ""]),
        *_wos_tag("SO", [journal.name if journal else ""]),
        *_wos_tag("DT", [pub.publication_type or ""]),
        *_wos_tag("DE", [_wos_keywords(pub, KeywordType.author_keyword)]),
        *_wos_tag("ID", [_wos_keywords(pub, KeywordType.mesh_term)]),
        *_wos_tag("AB", [pub.abstract or ""]),
        *_wos_tag("C1", _wos_addresses(pub, short_names)),
        *_wos_tag("CR", _wos_cited_references(pub, doi_by_source_id)),
        *_wos_tag("NR", [str(len(references))] if references else []),
        *_wos_tag("TC", [str(pub.citation_count)] if pub.citation_count is not None else []),
        *_wos_tag("SN", [journal.issn or ""] if journal else []),
        *_wos_tag("EI", [journal.e_issn or ""] if journal else []),
        *_wos_tag("PY", [str(pub.year)] if pub.year else []),
        *_wos_tag("DI", [pub.doi or ""]),
        *_wos_tag("UT", [f"{pub.source_database.upper()}:{pub.pmid}"] if pub.pmid else []),
        *_wos_tag("PM", [pub.pmid] if pub.source_database == "pubmed" else []),
        "ER",
        "",
    ]


def generate_wos_plaintext(pubs: list[Publication]) -> str:
    """Web of Science tagged plain text, the input VOSviewer and bibliometrix
    (``convert2df(dbsource="wos", format="plaintext")``) read.

    Every value is one line; multi-valued fields (AU, AF, C1, CR) put one value per
    line with 3-space continuation lines, records end with ``ER`` and a blank line,
    and the file ends with ``EF``. Field mapping and known gaps are documented in
    ``docs/scripting.md``.
    """
    doi_by_source_id = _wos_doi_by_source_id(pubs)
    lines = [*_WOS_HEADER]
    for pub in pubs:
        lines.extend(_wos_record(pub, doi_by_source_id))
    lines.append("EF")
    return _WOS_BOM + "\n".join(lines) + "\n"


EXCLUSION_REASON_LABELS = {
    "wrong_study_design": "Wrong study design",
    "wrong_population": "Wrong population",
    "wrong_intervention": "Wrong intervention",
    "wrong_outcome": "Wrong outcome",
    "not_peer_reviewed": "Not peer-reviewed (conference abstract, preprint, editorial)",
    "non_english": "Non-English language",
    "duplicate": "Duplicate not caught by automated dedup",
    "fulltext_unavailable": "Full-text not retrievable",
    "other": "Other / unspecified",
}


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _reason_count_lines(by_reason: dict[str, int]) -> list[str]:
    return [
        f"    {EXCLUSION_REASON_LABELS.get(reason, reason)}: {n}"
        for reason, n in sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _screening_stage_lines(exclusion_summary: ExclusionSummary | None) -> list[str]:
    """Current screening state per PRISMA 2020 stage, mirroring the flow diagram."""
    staged = summarize_exclusions(exclusion_summary)
    lines = [
        "SCREENING DECISIONS BY STAGE (PRISMA 2020 item 16a)",
        f"  Title/abstract screening: {_plural(staged.title_abstract_total, 'record')} excluded",
        *_reason_count_lines(staged.title_abstract_by_reason),
        f"  Full-text retrieval: {_plural(staged.not_retrieved, 'report')} not retrieved",
        f"  Full-text eligibility: {_plural(staged.full_text_total, 'report')} excluded",
        *_reason_count_lines(staged.full_text_by_reason),
    ]
    if staged.unrecorded_not_retrieved:
        lines.append(
            f"  Note: {staged.unrecorded_stage} exclusion(s) have no recorded screening stage: "
            f"{staged.unrecorded_not_retrieved} with reason full-text unavailable are counted as "
            "reports not retrieved, "
            f"{staged.unrecorded_stage - staged.unrecorded_not_retrieved} at title/abstract."
        )
    elif staged.unrecorded_stage:
        lines.append(
            f"  Note: {staged.unrecorded_stage} exclusion(s) have no recorded screening stage "
            "and are counted at title/abstract."
        )
    lines.append("")
    return lines


def generate_methodology(
    project_name: str,
    steps: list[MethodologyStep],
    exclusion_summary: ExclusionSummary | None = None,
) -> str:
    lines = [
        f'METHODOLOGY LOG — Project: "{project_name}"',
        f"Generated: {date.today().isoformat()}",
        "Tool: BibMedEd (https://github.com/ata381/bibmeded)",
        "",
    ]
    if not steps:
        lines.append("No methodology steps recorded for this project.")
        return "\n".join(lines)

    current_phase = None
    for step in steps:
        phase_header = _PHASE_LABELS.get(step.phase, step.phase.upper())
        if phase_header != current_phase:
            current_phase = phase_header
            lines.append(current_phase)
        lines.append(f"  Step {step.step_order}: {step.action}")
        if step.phase == "search":
            query_str = step.parameters.get("query", "")
            if query_str:
                lines.append(f"    Query: {query_str}")
            searched_at = step.parameters.get("searched_at")
            if searched_at:
                lines.append(f"    Searched at: {searched_at}")
            capped = step.parameters.get("capped", False)
            max_results = step.parameters.get("max_results")
            if capped and max_results:
                lines.append(
                    f"    Results: {step.records_out} records (capped at max_results={max_results}; "
                    f"full set not retrieved)"
                )
            else:
                lines.append(f"    Results: {step.records_out} records")
        elif step.phase == "fetch":
            lines.append(
                f"    Retrieved: {step.records_out} of {step.records_in} "
                f"({step.records_affected} unavailable)"
            )
            persist_failures = step.parameters.get("persist_failures") or []
            persist_failed = max(step.parameters.get("persist_failed") or 0, len(persist_failures))
            if persist_failed:
                lines.append(f"    Failed to persist: {persist_failed} record(s)")
                for failure in persist_failures:
                    lines.append(
                        f"      - {failure.get('source_id')} "
                        f"({failure.get('source_database')}): {failure.get('error')}"
                    )
                omitted = persist_failed - len(persist_failures)
                if omitted:
                    lines.append(f"      ... {omitted} more not listed")
        elif step.phase == "dedup":
            method = step.parameters.get("method", "unknown")
            fields = step.parameters.get("fields") or step.parameters.get("field") or ""
            if fields:
                lines.append(f"    Method: {method} on {fields}")
            else:
                lines.append(f"    Method: {method}")
            removed_by = step.parameters.get("removed_by")
            if isinstance(removed_by, dict) and removed_by:
                breakdown = ", ".join(f"{k}={v}" for k, v in removed_by.items() if v)
                if breakdown:
                    lines.append(f"    Removed by field: {breakdown}")
            lines.append(
                f"    Removed: {step.records_affected} duplicates "
                f"({step.records_in} → {step.records_out})"
            )
        elif step.phase == "enrichment":
            source_name = step.parameters.get("source", "")
            enriched = step.parameters.get("enriched", 0)
            missing = step.parameters.get("missing", 0)
            lines.append(f"    Source: {source_name}")
            lines.append(
                f"    Enriched: {enriched} of {step.records_in} records ({missing} not found)"
            )
        elif step.phase == "exclusion":
            lines.append(
                f"    Excluded: {step.records_affected} records "
                f"({step.records_in} → {step.records_out})"
            )
        lines.append("")

    lines.extend(_screening_stage_lines(exclusion_summary))

    # For a multi-query project, `steps` interleaves the per-query pipelines
    # (ordered query_id, step_order), so steps[-1] is only the *last query's*
    # terminal step — not the project-wide total. Take each query's own
    # terminal step (the last one seen for that query_id) and sum their
    # records_out to get the true project-wide included-studies figure.
    terminal_step_by_query: dict[int, MethodologyStep] = {}
    for step in steps:
        terminal_step_by_query[step.query_id] = step
    studies_included = sum(step.records_out for step in terminal_step_by_query.values())

    lines.append("FINAL DATASET")
    lines.append(f"  Studies included: {studies_included}")
    lines.append("")
    return "\n".join(lines)


def generate_prisma_svg(
    project_name: str,
    steps: list[MethodologyStep],
    exclusion_summary: ExclusionSummary | None = None,
    included_count: int | None = None,
) -> str:
    counts = compute_counts(steps, exclusion_summary=exclusion_summary, included_override=included_count)
    return render_svg(counts, project_name)


def generate_bundle(
    project_name: str,
    pubs: list[Publication],
    steps: list[MethodologyStep],
    exclusion_summary: ExclusionSummary | None = None,
) -> bytes:
    """Produce a single .zip containing CSV, RIS, WoS plain text, JSON, methodology .txt, PRISMA .svg, and a manifest."""
    stamp = date.today().isoformat()
    slug = slugify(project_name) or "project"
    # pubs is already filtered to excluded == False by the caller (_get_project_and_pubs),
    # so len(pubs) is the authoritative "included" count.
    included_count = len(pubs)
    cited_written, cited_reported = wos_reference_coverage(pubs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{slug}-{stamp}.csv", generate_csv(pubs))
        zf.writestr(f"{slug}-{stamp}.ris", generate_ris(pubs))
        zf.writestr(f"{slug}-wos-{stamp}.txt", generate_wos_plaintext(pubs))
        zf.writestr(f"{slug}-{stamp}.json", generate_json(project_name, pubs))
        zf.writestr(f"{slug}-methodology-{stamp}.txt", generate_methodology(project_name, steps, exclusion_summary))
        zf.writestr(f"{slug}-prisma-{stamp}.svg", generate_prisma_svg(project_name, steps, exclusion_summary, included_count=included_count))
        manifest = (
            f"BibMedEd export bundle\n"
            f"Project: {project_name}\n"
            f"Generated: {stamp}\n"
            f"Schema version: {EXPORT_SCHEMA_VERSION}\n"
            f"Included files:\n"
            f"  - {slug}-{stamp}.csv ({len(pubs)} records)\n"
            f"  - {slug}-{stamp}.ris ({len(pubs)} records)\n"
            f"  - {slug}-wos-{stamp}.txt ({len(pubs)} records, Web of Science plain text; "
            f"CR lists {cited_written} of {cited_reported} cited references, "
            f"only those with a DOI or matching an in-corpus record with a DOI)\n"
            f"  - {slug}-{stamp}.json ({len(pubs)} records, schema_version={EXPORT_SCHEMA_VERSION})\n"
            f"  - {slug}-methodology-{stamp}.txt ({len(steps)} steps)\n"
            f"  - {slug}-prisma-{stamp}.svg\n"
        )
        zf.writestr("MANIFEST.txt", manifest)
    return buf.getvalue()
