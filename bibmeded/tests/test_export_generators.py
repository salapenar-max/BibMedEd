"""Unit tests for bibmeded.services.export_generators.

These build Publication / Author / Journal / Keyword / MethodologyStep instances
purely in-memory (no DB session) since the generator functions only read
attributes/relationships that are already populated in memory.
"""

import csv
import io
import json
import re
import zipfile
from datetime import datetime, timezone

import pytest

from bibmeded.models.author import Affiliation, Author
from bibmeded.models.journal import Journal
from bibmeded.models.keyword import Keyword, KeywordType
from bibmeded.models.methodology import MethodologyStep
from bibmeded.models.publication import Publication
from bibmeded.services.export_generators import (
    EXPORT_SCHEMA_VERSION,
    generate_bundle,
    generate_csv,
    generate_json,
    generate_methodology,
    generate_ris,
    generate_wos_plaintext,
)


def make_pub(
    *,
    pmid="12345",
    doi="10.1000/xyz",
    title="A Study of Things",
    abstract="An abstract.",
    year=2020,
    citation_count=5,
    authors=None,
    journal_name="Journal of Testing",
    keywords=None,
    publication_type="Journal Article",
    source_database="pubmed",
    excluded=False,
    exclusion_reason=None,
    screening_stage=None,
):
    pub = Publication(
        pmid=pmid,
        doi=doi,
        title=title,
        abstract=abstract,
        year=year,
        citation_count=citation_count,
        publication_type=publication_type,
        source_database=source_database,
        excluded=excluded,
        exclusion_reason=exclusion_reason,
        screening_stage=screening_stage,
    )
    pub.authors = authors if authors is not None else [Author(name="Doe J")]
    pub.journal = Journal(name=journal_name) if journal_name is not None else None
    pub.keywords = (
        keywords
        if keywords is not None
        else [Keyword(term="medical education", type=KeywordType.author_keyword)]
    )
    return pub


def make_step(
    *,
    query_id=1,
    step_order=1,
    phase="search",
    source="pubmed",
    action="Search",
    records_in=0,
    records_out=10,
    records_affected=10,
    parameters=None,
):
    return MethodologyStep(
        query_id=query_id,
        step_order=step_order,
        phase=phase,
        source=source,
        action=action,
        records_in=records_in,
        records_out=records_out,
        records_affected=records_affected,
        parameters=parameters or {},
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# generate_csv
# ---------------------------------------------------------------------------


class TestGenerateCsv:
    def test_writes_header_row(self):
        csv_text = generate_csv([])
        rows = list(csv.reader(io.StringIO(csv_text)))
        assert rows == [
            ["PMID", "DOI", "Title", "Authors", "Journal", "Year", "Citations", "Keywords", "Abstract"]
        ]

    def test_writes_exact_row_content_for_a_publication(self):
        pub = make_pub(
            pmid="999",
            doi="10.1/abc",
            title="Effects of Simulation Training",
            abstract="Line one.\nLine two.",
            year=2021,
            citation_count=42,
            authors=[Author(name="Smith A"), Author(name="Jones B")],
            journal_name="Medical Education Review",
            keywords=[
                Keyword(term="simulation", type=KeywordType.author_keyword),
                Keyword(term="training", type=KeywordType.author_keyword),
            ],
        )
        csv_text = generate_csv([pub])
        rows = list(csv.reader(io.StringIO(csv_text)))
        assert len(rows) == 2
        data_row = rows[1]
        assert data_row[0] == "999"
        assert data_row[1] == "10.1/abc"
        assert data_row[2] == "Effects of Simulation Training"
        assert data_row[3] == "Smith A; Jones B"
        assert data_row[4] == "Medical Education Review"
        assert data_row[5] == "2021"
        assert data_row[6] == "42"
        assert data_row[7] == "simulation; training"
        # Newlines inside the abstract are flattened to spaces.
        assert data_row[8] == "Line one. Line two."

    def test_handles_missing_optional_fields(self):
        pub = make_pub(doi=None, journal_name=None, keywords=[], authors=[], abstract=None, year=None)
        csv_text = generate_csv([pub])
        rows = list(csv.reader(io.StringIO(csv_text)))
        data_row = rows[1]
        assert data_row[1] == ""  # doi
        assert data_row[3] == ""  # authors
        assert data_row[4] == ""  # journal
        assert data_row[5] == ""  # year
        assert data_row[7] == ""  # keywords
        assert data_row[8] == ""  # abstract

    @pytest.mark.parametrize("trigger", ["=", "+", "-", "@", "\t", "\r"])
    def test_escapes_formula_injection_in_title(self, trigger):
        """OWASP CSV injection: a cell starting with =, +, -, @ (or tab/CR) can be
        interpreted as a formula by Excel/Sheets. Such values must be prefixed
        with a leading apostrophe so they are treated as plain text.
        """
        malicious_title = f"{trigger}cmd|'/c calc'!A1"
        pub = make_pub(title=malicious_title)
        csv_text = generate_csv([pub])
        rows = list(csv.reader(io.StringIO(csv_text)))
        data_row = rows[1]
        assert data_row[2] == "'" + malicious_title
        assert not data_row[2].startswith(trigger)

    def test_escapes_formula_injection_in_authors_and_journal_and_keywords(self):
        pub = make_pub(
            title="Safe Title",
            authors=[Author(name="=HYPERLINK(\"http://evil\")")],
            journal_name="+1+1",
            keywords=[Keyword(term="@evil", type=KeywordType.author_keyword)],
        )
        csv_text = generate_csv([pub])
        rows = list(csv.reader(io.StringIO(csv_text)))
        data_row = rows[1]
        assert data_row[3].startswith("'=")
        assert data_row[4].startswith("'+")
        assert data_row[7].startswith("'@")

    def test_does_not_escape_benign_values(self):
        pub = make_pub(title="Benign Title", journal_name="Regular Journal")
        csv_text = generate_csv([pub])
        rows = list(csv.reader(io.StringIO(csv_text)))
        data_row = rows[1]
        assert data_row[2] == "Benign Title"
        assert data_row[4] == "Regular Journal"


# ---------------------------------------------------------------------------
# generate_json
# ---------------------------------------------------------------------------


class TestGenerateJson:
    def test_top_level_schema_shape(self):
        pub = make_pub()
        json_text = generate_json("My Project", [pub])
        payload = json.loads(json_text)
        assert payload["schema_version"] == EXPORT_SCHEMA_VERSION
        assert payload["project"] == "My Project"
        assert payload["count"] == 1
        assert "generated" in payload
        assert isinstance(payload["publications"], list)
        assert len(payload["publications"]) == 1

    def test_publication_entry_shape(self):
        pub = make_pub(
            pmid="42",
            doi="10.1/x",
            title="Title",
            abstract="Abstract text",
            year=2019,
            publication_type="Review",
            source_database="crossref",
            citation_count=7,
            authors=[Author(name="Alice", orcid="0000-0001")],
            journal_name="Journal X",
            keywords=[Keyword(term="kw1", type=KeywordType.mesh_term)],
        )
        payload = json.loads(generate_json("Proj", [pub]))
        entry = payload["publications"][0]
        assert entry["pmid"] == "42"
        assert entry["doi"] == "10.1/x"
        assert entry["title"] == "Title"
        assert entry["abstract"] == "Abstract text"
        assert entry["year"] == 2019
        assert entry["publication_type"] == "Review"
        assert entry["source_database"] == "crossref"
        assert entry["citation_count"] == 7
        assert entry["excluded"] is False
        assert entry["exclusion_reason"] is None
        assert entry["journal_name"] == "Journal X"
        assert entry["authors"] == [{"name": "Alice", "orcid": "0000-0001"}]
        assert entry["keywords"] == ["kw1"]

    def test_empty_publication_list(self):
        payload = json.loads(generate_json("Empty Project", []))
        assert payload["count"] == 0
        assert payload["publications"] == []

    def test_round_trips_non_ascii_characters(self):
        pub = make_pub(title="Tıp Eğitiminde Yapay Zekâ – 教育")
        json_text = generate_json("Proj", [pub])
        # ensure_ascii=False means non-ASCII chars are written literally, not \uXXXX escaped
        assert "Tıp Eğitiminde Yapay Zekâ" in json_text
        payload = json.loads(json_text)
        assert payload["publications"][0]["title"] == "Tıp Eğitiminde Yapay Zekâ – 教育"


# ---------------------------------------------------------------------------
# generate_ris
# ---------------------------------------------------------------------------


class TestGenerateRis:
    def test_writes_expected_tag_sequence(self):
        pub = make_pub(
            pmid="55",
            doi="10.1/z",
            title="RIS Title",
            year=2018,
            authors=[Author(name="Author One"), Author(name="Author Two")],
            journal_name="RIS Journal",
            keywords=[Keyword(term="kw-a", type=KeywordType.author_keyword)],
            abstract="Some abstract",
        )
        ris_text = generate_ris([pub])
        lines = ris_text.split("\n")
        assert lines[0] == "TY  - JOUR"
        assert lines[1] == "TI  - RIS Title"
        assert lines[2] == "AU  - Author One"
        assert lines[3] == "AU  - Author Two"
        assert lines[4] == "JO  - RIS Journal"
        assert lines[5] == "PY  - 2018"
        assert lines[6] == "DO  - 10.1/z"
        assert lines[7] == "AN  - 55"
        assert lines[8] == "AB  - Some abstract"
        assert lines[9] == "KW  - kw-a"
        assert lines[10] == "ER  - "

    def test_omits_optional_tags_when_absent(self):
        pub = make_pub(doi=None, journal_name=None, keywords=[], abstract=None, authors=[])
        ris_text = generate_ris([pub])
        assert "JO  - " not in ris_text
        assert "DO  - " not in ris_text
        assert "AB  - " not in ris_text
        assert "KW  - " not in ris_text
        assert "AU  - " not in ris_text

    def test_multiple_publications_each_terminated(self):
        pubs = [make_pub(pmid="1"), make_pub(pmid="2")]
        ris_text = generate_ris(pubs)
        er_lines = [line for line in ris_text.split("\n") if line == "ER  - "]
        assert len(er_lines) == 2

    def test_sanitizes_newline_injection_in_title(self):
        """A malicious record with a newline + fake 'ER  - ' + 'TY  - JOUR' in the
        title must not be able to inject an extra RIS record boundary.
        """
        malicious_title = "Normal Title\nER  - \nTY  - JOUR\nAB  - injected record"
        pub = make_pub(pmid="66", title=malicious_title)
        ris_text = generate_ris([pub])
        lines = ris_text.split("\n")
        # Exactly one legitimate ER line for this single record.
        er_lines = [line for line in lines if line == "ER  - "]
        assert len(er_lines) == 1
        # No injected TY line should appear beyond the legitimate first one.
        ty_lines = [line for line in lines if line == "TY  - JOUR"]
        assert len(ty_lines) == 1
        # The title tag itself must not contain a raw newline.
        title_lines = [line for line in lines if line.startswith("TI  - ")]
        assert len(title_lines) == 1
        assert "\n" not in title_lines[0]

    def test_sanitizes_newline_injection_in_author_and_journal_and_keyword(self):
        pub = make_pub(
            title="Safe",
            authors=[Author(name="Evil\nER  - \nTY  - JOUR")],
            journal_name="Journal\nER  - ",
            keywords=[Keyword(term="kw\nER  - ", type=KeywordType.author_keyword)],
        )
        ris_text = generate_ris([pub])
        er_lines = [line for line in ris_text.split("\n") if line == "ER  - "]
        assert len(er_lines) == 1


# ---------------------------------------------------------------------------
# generate_wos_plaintext
# ---------------------------------------------------------------------------

WOS_HEADER = ("FN Clarivate Analytics Web of Science", "VR 1.0")
_WOS_TAG_LINE = re.compile(r"^([A-Z][A-Z0-9]) (\S.*)$")
_WOS_CONTINUATION_LINE = re.compile(r"^   (\S.*)$")


def parse_wos(text: str) -> list[list[tuple[str, list[str]]]]:
    """Parse WoS tagged plain text by the published rules, failing on any deviation.

    Rules: optional UTF-8 BOM, ``FN`` then ``VR 1.0``; each record opens with ``PT``,
    every line is a two-character tag plus one space plus a value, or a 3-space
    continuation of the previous tag; ``ER`` closes a record and is followed by one
    blank line; the file ends with ``EF``.
    """
    lines = text.removeprefix("\ufeff").split("\n")
    assert tuple(lines[:2]) == WOS_HEADER
    assert lines[-2:] == ["EF", ""]
    records: list[list[tuple[str, list[str]]]] = []
    current: list[tuple[str, list[str]]] | None = None
    body = iter(lines[2:-2])
    for line in body:
        if current is None:
            assert line.startswith("PT "), f"record must open with PT, got {line!r}"
            current = []
        if line == "ER":
            assert next(body, None) == "", "ER must be followed by a blank line"
            records.append(current)
            current = None
            continue
        continuation = _WOS_CONTINUATION_LINE.match(line)
        if continuation:
            assert current, "continuation line with no open tag"
            current[-1][1].append(continuation.group(1))
            continue
        tag_line = _WOS_TAG_LINE.match(line)
        assert tag_line, f"not a WoS tag line: {line!r}"
        current.append((tag_line.group(1), [tag_line.group(2)]))
    assert current is None, "last record not closed by ER"
    return records


def wos_fields(text: str, index: int = 0) -> dict[str, list[str]]:
    return dict(parse_wos(text)[index])


def make_wos_pub(**overrides):
    defaults = dict(
        pmid="31234567",
        doi="10.1111/medu.13900",
        title="Feedback in clinical training",
        abstract="We studied feedback.",
        year=2019,
        citation_count=12,
        journal_name="Medical Education",
        source_database="pubmed",
        authors=[
            Author(
                name="Demir, Sofia Ayse",
                affiliations=[Affiliation(name="Hacettepe University, Ankara, Turkey", country="Turkey")],
            ),
            Author(
                name="Kaya, Ali",
                affiliations=[Affiliation(name="Hacettepe University, Ankara, Turkey", country="Turkey")],
            ),
            Author(
                name="Smith, John",
                affiliations=[Affiliation(name="University of Michigan, Ann Arbor, MI, USA", country="United States")],
            ),
        ],
        keywords=[
            Keyword(term="feedback", type=KeywordType.author_keyword),
            Keyword(term="simulation", type=KeywordType.author_keyword),
            Keyword(term="Education, Medical", type=KeywordType.mesh_term),
            Keyword(term="Humans", type=KeywordType.mesh_term),
        ],
    )
    external_references = overrides.pop("external_references", None)
    issn = overrides.pop("issn", "0308-0110")
    e_issn = overrides.pop("e_issn", "1365-2923")
    defaults.update(overrides)
    pub = make_pub(**defaults)
    if pub.journal is not None:
        pub.journal.issn = issn
        pub.journal.e_issn = e_issn
    pub.external_references = external_references
    return pub


class TestGenerateWosPlaintext:
    @pytest.mark.parametrize(
        "malformed",
        [
            "FN Clarivate Analytics Web of Science\nVR 1.0\nPT J\nTI \nER\n\nEF\n",
            "FN Clarivate Analytics Web of Science\nVR 1.0\nPT J\n  two-space continuation\nER\n\nEF\n",
            "FN Clarivate Analytics Web of Science\nVR 1.0\nPT J\nER\nPT J\nER\n\nEF\n",
            "FN Clarivate Analytics Web of Science\nVR 1.0\nPT J\nER\n\n",
        ],
    )
    def test_format_checker_rejects_malformed_files(self, malformed):
        with pytest.raises(AssertionError):
            parse_wos(malformed)

    def test_header_and_footer_follow_the_published_format(self):
        text = generate_wos_plaintext([make_wos_pub()])
        assert text.startswith("\ufeffFN Clarivate Analytics Web of Science\nVR 1.0\nPT J\n")
        assert text.endswith("\nER\n\nEF\n")

    def test_empty_corpus_is_a_valid_file_with_no_records(self):
        text = generate_wos_plaintext([])
        assert parse_wos(text) == []

    def test_tags_are_emitted_in_web_of_science_order(self):
        pub = make_wos_pub(
            external_references=["10.1000/cited-one"],
            publication_type="Journal Article",
        )
        tags = [tag for tag, _ in parse_wos(generate_wos_plaintext([pub]))[0]]
        assert tags == [
            "PT", "AU", "AF", "TI", "SO", "DT", "DE", "ID", "AB", "C1", "CR", "NR",
            "TC", "SN", "EI", "PY", "DI", "UT", "PM",
        ]

    def test_one_author_per_line_in_surname_initials_form(self):
        text = generate_wos_plaintext([make_wos_pub()])
        assert "\nAU Demir, SA\n   Kaya, A\n   Smith, J\n" in text
        assert wos_fields(text)["AU"] == ["Demir, SA", "Kaya, A", "Smith, J"]
        assert wos_fields(text)["AF"] == ["Demir, Sofia Ayse", "Kaya, Ali", "Smith, John"]

    def test_given_family_names_are_reordered_to_surname_first(self):
        pub = make_wos_pub(
            source_database="openalex",
            authors=[Author(name="Jean-Pierre Dupont"), Author(name="Mary J. Watson")],
        )
        fields = wos_fields(generate_wos_plaintext([pub]))
        assert fields["AU"] == ["Dupont, JP", "Watson, MJ"]
        assert fields["AF"] == ["Dupont, Jean-Pierre", "Watson, Mary J."]

    def test_pubmed_collective_and_single_token_names_are_kept_verbatim(self):
        pub = make_wos_pub(authors=[Author(name="STARD Study Group"), Author(name="Plato")])
        fields = wos_fields(generate_wos_plaintext([pub]))
        assert fields["AU"] == ["STARD Study Group", "Plato"]
        assert fields["AF"] == ["STARD Study Group", "Plato"]

    def test_c1_groups_authors_by_affiliation_and_keeps_the_country(self):
        fields = wos_fields(generate_wos_plaintext([make_wos_pub()]))
        assert fields["C1"] == [
            "[Demir, SA; Kaya, A] Hacettepe University, Ankara, Turkey",
            "[Smith, J] University of Michigan, Ann Arbor, MI, USA",
        ]

    def test_doi_references_become_cited_references(self):
        pub = make_wos_pub(external_references=["10.1000/cited-one", "10.1000/CITED-two"])
        fields = wos_fields(generate_wos_plaintext([pub]))
        assert fields["CR"] == ["DOI 10.1000/cited-one", "DOI 10.1000/cited-two"]
        assert fields["NR"] == ["2"]

    def test_in_corpus_references_resolve_to_the_cited_record_doi(self):
        cited = make_wos_pub(pmid="W100", doi="10.1000/in-corpus", source_database="openalex")
        citing = make_wos_pub(
            pmid="W200",
            doi="10.1000/citing",
            source_database="openalex",
            external_references=["W100", "W999"],
        )
        fields = wos_fields(generate_wos_plaintext([citing, cited]))
        assert fields["CR"] == ["DOI 10.1000/in-corpus"]
        assert fields["NR"] == ["2"]

    def test_unresolvable_references_leave_cr_out_but_nr_counts_them(self):
        pub = make_wos_pub(source_database="sample", external_references=["sample-001", "sample-002"])
        fields = wos_fields(generate_wos_plaintext([pub]))
        assert "CR" not in fields
        assert fields["NR"] == ["2"]

    def test_no_cr_or_nr_when_the_source_gave_no_references(self):
        fields = wos_fields(generate_wos_plaintext([make_wos_pub(external_references=None)]))
        assert "CR" not in fields
        assert "NR" not in fields

    def test_times_cited_comes_from_citation_count(self):
        assert wos_fields(generate_wos_plaintext([make_wos_pub(citation_count=12)]))["TC"] == ["12"]
        assert wos_fields(generate_wos_plaintext([make_wos_pub(citation_count=0)]))["TC"] == ["0"]
        assert "TC" not in wos_fields(generate_wos_plaintext([make_wos_pub(citation_count=None)]))

    def test_identifiers_year_and_source(self):
        fields = wos_fields(generate_wos_plaintext([make_wos_pub()]))
        assert fields["PY"] == ["2019"]
        assert fields["DI"] == ["10.1111/medu.13900"]
        assert fields["PM"] == ["31234567"]
        assert fields["UT"] == ["PUBMED:31234567"]
        assert fields["SO"] == ["Medical Education"]
        assert fields["SN"] == ["0308-0110"]
        assert fields["EI"] == ["1365-2923"]

    def test_pm_only_for_pubmed_records_because_other_sources_store_their_own_id(self):
        pub = make_wos_pub(pmid="W123", source_database="openalex")
        fields = wos_fields(generate_wos_plaintext([pub]))
        assert "PM" not in fields
        assert fields["UT"] == ["OPENALEX:W123"]

    def test_author_keywords_in_de_and_mesh_in_id(self):
        fields = wos_fields(generate_wos_plaintext([make_wos_pub()]))
        assert fields["DE"] == ["feedback; simulation"]
        assert fields["ID"] == ["Education, Medical; Humans"]

    def test_utf8_text_round_trips(self):
        pub = make_wos_pub(
            title="Tıp eğitiminde geri bildirim: 医学教育",
            authors=[Author(name="Çelik, Ömer", affiliations=[Affiliation(name="Ege Üniversitesi, İzmir, Türkiye")])],
        )
        text = generate_wos_plaintext([pub])
        decoded = text.encode("utf-8").decode("utf-8")
        fields = wos_fields(decoded)
        assert fields["TI"] == ["Tıp eğitiminde geri bildirim: 医学教育"]
        assert fields["AU"] == ["Çelik, Ö"]
        assert fields["C1"] == ["[Çelik, Ö] Ege Üniversitesi, İzmir, Türkiye"]

    def test_missing_fields_produce_no_empty_tags(self):
        pub = make_wos_pub(
            doi=None,
            abstract=None,
            year=None,
            citation_count=None,
            journal_name=None,
            publication_type=None,
            keywords=[],
            authors=[],
            source_database="sample",
        )
        fields = wos_fields(generate_wos_plaintext([pub]))
        assert set(fields) == {"PT", "TI", "UT"}

    def test_blank_values_are_skipped_rather_than_written_as_empty_lines(self):
        pub = make_wos_pub(
            authors=[
                Author(name="  ", affiliations=[Affiliation(name="Orphan University, Turkey")]),
                Author(name="Kaya, Ali", affiliations=[Affiliation(name=" ")]),
            ],
            keywords=[Keyword(term=" ", type=KeywordType.author_keyword)],
        )
        fields = wos_fields(generate_wos_plaintext([pub]))
        assert fields["AU"] == ["Kaya, A"]
        assert "DE" not in fields
        assert "C1" not in fields

    def test_records_are_separated_by_er_and_a_blank_line(self):
        pubs = [make_wos_pub(pmid="1", doi="10.1/a"), make_wos_pub(pmid="2", doi="10.1/b")]
        text = generate_wos_plaintext(pubs)
        records = parse_wos(text)
        assert [dict(r)["UT"] for r in records] == [["PUBMED:1"], ["PUBMED:2"]]
        assert "\nER\n\nPT J\n" in text

    def test_newlines_in_values_cannot_inject_tags_or_records(self):
        pub = make_wos_pub(
            title="Normal\nER\n\nPT J\nTI injected",
            abstract="Line one\r\nLine two",
            authors=[Author(name="Evil\nER", affiliations=[Affiliation(name="Univ\nEF")])],
        )
        text = generate_wos_plaintext([pub])
        records = parse_wos(text)
        assert len(records) == 1
        fields = dict(records[0])
        assert fields["TI"] == ["Normal ER PT J TI injected"]
        assert fields["AB"] == ["Line one Line two"]


# ---------------------------------------------------------------------------
# generate_bundle
# ---------------------------------------------------------------------------


class TestGenerateBundle:
    def test_zip_contains_all_expected_members(self):
        pub = make_pub()
        steps = [make_step(query_id=1, step_order=1, phase="search", records_out=10)]
        bundle_bytes = generate_bundle("Bundle Project", [pub], steps, {})
        zf = zipfile.ZipFile(io.BytesIO(bundle_bytes))
        names = zf.namelist()
        assert any(name.endswith(".csv") for name in names)
        assert any(name.endswith(".ris") for name in names)
        assert any(name.endswith(".json") for name in names)
        assert any("methodology" in name and name.endswith(".txt") for name in names)
        assert any("-wos-" in name and name.endswith(".txt") for name in names)
        assert any(name.endswith(".svg") for name in names)
        assert "MANIFEST.txt" in names

    def test_bundle_wos_file_matches_the_standalone_export(self):
        pub = make_wos_pub(external_references=["10.1000/cited-one", "sample-x"])
        bundle_bytes = generate_bundle("Bundle Project", [pub], [make_step()], {})
        zf = zipfile.ZipFile(io.BytesIO(bundle_bytes))
        wos_name = next(name for name in zf.namelist() if "-wos-" in name)
        assert zf.read(wos_name).decode("utf-8") == generate_wos_plaintext([pub])
        manifest = zf.read("MANIFEST.txt").decode("utf-8")
        assert f"{wos_name} (1 records, Web of Science plain text; CR lists 1 of 2 cited references" in manifest

    def test_manifest_mentions_schema_version_and_counts(self):
        pubs = [make_pub(pmid="1"), make_pub(pmid="2")]
        steps = [make_step(records_out=2)]
        bundle_bytes = generate_bundle("Bundle Project", pubs, steps, {})
        zf = zipfile.ZipFile(io.BytesIO(bundle_bytes))
        manifest = zf.read("MANIFEST.txt").decode("utf-8")
        assert EXPORT_SCHEMA_VERSION in manifest
        assert "2 records" in manifest
        assert "Bundle Project" in manifest


# ---------------------------------------------------------------------------
# generate_methodology — multi-query "Studies included" total (finding [12])
# ---------------------------------------------------------------------------


class TestGenerateMethodologyStudiesIncluded:
    def test_single_query_uses_last_step_records_out(self):
        steps = [
            make_step(query_id=1, step_order=1, phase="search", records_in=0, records_out=100, records_affected=100),
            make_step(query_id=1, step_order=2, phase="fetch", records_in=100, records_out=95, records_affected=5),
        ]
        text = generate_methodology("Single Query Project", steps)
        assert "Studies included: 95" in text

    def test_multi_query_totals_across_queries_not_just_last_query(self):
        """Two queries in one project, each contributing separately-tracked
        included studies. The final "Studies included" figure must be the sum
        across both queries, not just the last query's last step.
        """
        steps = [
            # Query 1: ends with 60 included studies.
            make_step(query_id=1, step_order=1, phase="search", records_in=0, records_out=200, records_affected=200),
            make_step(query_id=1, step_order=2, phase="fetch", records_in=200, records_out=60, records_affected=140),
            # Query 2: ends with 30 included studies.
            make_step(query_id=2, step_order=1, phase="search", records_in=0, records_out=50, records_affected=50),
            make_step(query_id=2, step_order=2, phase="fetch", records_in=50, records_out=30, records_affected=20),
        ]
        text = generate_methodology("Multi Query Project", steps)
        # Old buggy behavior would report only steps[-1].records_out == 30.
        assert "Studies included: 90" in text
        assert "Studies included: 30" not in text

    def test_fetch_step_lists_records_that_failed_to_persist(self):
        steps = [
            make_step(query_id=1, step_order=1, phase="search", records_in=0, records_out=3, records_affected=3),
            make_step(
                query_id=1, step_order=2, phase="fetch", records_in=3, records_out=2, records_affected=1,
                parameters={
                    "persist_failed": 1,
                    "persist_failures": [
                        {"source_id": "W2302", "source_database": "openalex", "error": "ValueError: boom"},
                    ],
                },
            ),
        ]
        text = generate_methodology("Failure Project", steps)
        assert "Failed to persist: 1 record(s)" in text
        assert "W2302 (openalex): ValueError: boom" in text

    def test_fetch_step_reports_total_and_omitted_count_when_failure_list_is_capped(self):
        steps = [
            make_step(
                query_id=1, step_order=1, phase="fetch", records_in=150, records_out=0, records_affected=150,
                parameters={
                    "persist_failed": 150,
                    "persist_failures": [
                        {"source_id": f"W{i}", "source_database": "openalex", "error": "ValueError: boom"}
                        for i in range(100)
                    ],
                },
            ),
        ]
        text = generate_methodology("Capped Project", steps)
        assert "Failed to persist: 150 record(s)" in text
        assert "W99 (openalex)" in text
        assert "... 50 more not listed" in text

    def test_fetch_step_without_persist_failures_omits_failure_line(self):
        steps = [
            make_step(query_id=1, step_order=1, phase="fetch", records_in=3, records_out=3, records_affected=0,
                      parameters={"persist_failed": 0, "persist_failures": []}),
        ]
        text = generate_methodology("Clean Project", steps)
        assert "Failed to persist" not in text

    def test_no_steps_returns_placeholder_and_no_studies_line(self):
        text = generate_methodology("Empty Project", [])
        assert "No methodology steps recorded" in text
        assert "Studies included" not in text


class TestGenerateMethodologyScreeningStages:
    STEPS = [make_step(query_id=1, step_order=1, phase="search", records_in=0, records_out=20, records_affected=20)]

    def test_reports_each_stage_with_per_reason_counts(self):
        summary = {
            ("title_abstract", "non_english"): 3,
            ("title_abstract", "wrong_population"): 1,
            ("full_text", "wrong_outcome"): 2,
            ("full_text", "fulltext_unavailable"): 1,
        }
        text = generate_methodology("Staged", self.STEPS, summary)
        section = text.split("SCREENING DECISIONS BY STAGE (PRISMA 2020 item 16a)", 1)[1].split("FINAL DATASET", 1)[0]
        assert section.splitlines()[1:8] == [
            "  Title/abstract screening: 4 records excluded",
            "    Non-English language: 3",
            "    Wrong population: 1",
            "  Full-text retrieval: 1 report not retrieved",
            "  Full-text eligibility: 2 reports excluded",
            "    Wrong outcome: 2",
            "",
        ]

    def test_reports_zero_counts_for_a_stage_without_exclusions(self):
        text = generate_methodology("TA only", self.STEPS, {("title_abstract", "other"): 1})
        assert "Title/abstract screening: 1 record excluded" in text
        assert "Full-text retrieval: 0 reports not retrieved" in text
        assert "Full-text eligibility: 0 reports excluded" in text

    def test_notes_exclusions_without_a_recorded_stage(self):
        text = generate_methodology("Legacy", self.STEPS, {"non_english": 2, None: 1})
        assert "Title/abstract screening: 3 records excluded" in text
        assert "3 exclusion(s) have no recorded screening stage and are counted at title/abstract" in text

    def test_notes_stageless_fulltext_unavailable_as_reports_not_retrieved(self):
        text = generate_methodology("Legacy", self.STEPS, {"non_english": 2, (None, "fulltext_unavailable"): 1})
        assert "Title/abstract screening: 2 records excluded" in text
        assert "Full-text retrieval: 1 report not retrieved" in text
        assert (
            "3 exclusion(s) have no recorded screening stage: 1 with reason full-text unavailable "
            "are counted as reports not retrieved, 2 at title/abstract."
        ) in text

    def test_omits_unrecorded_note_when_every_exclusion_has_a_stage(self):
        text = generate_methodology("Staged", self.STEPS, {("full_text", "other"): 1})
        assert "no recorded screening stage" not in text

    def test_reports_stage_section_with_zero_counts_when_nothing_excluded(self):
        text = generate_methodology("Unscreened", self.STEPS, {})
        assert "Title/abstract screening: 0 records excluded" in text
        assert "Full-text eligibility: 0 reports excluded" in text


class TestScreeningStageInJson:
    def test_publication_entry_carries_screening_stage(self):
        pub = make_pub(excluded=True, exclusion_reason="wrong_outcome", screening_stage="full_text")
        entry = json.loads(generate_json("P", [pub]))["publications"][0]
        assert entry["screening_stage"] == "full_text"

    def test_included_publication_has_null_screening_stage(self):
        entry = json.loads(generate_json("P", [make_pub()]))["publications"][0]
        assert entry["screening_stage"] is None
