import os, sqlite3, tempfile
import pytest
from agent import store
from agent.identity import make_job_id


@pytest.fixture
def db():
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    store.init_db(path)
    yield path


def test_empty_table_is_safe(db):
    assert store.list_applications(db) == []
    assert store.status_counts(db) == {}
    assert store.get_application("nope", db) is None


def test_same_title_different_jobs_are_stored_separately(db):
    a = make_job_id("Acme", "SWE Intern", "https://a.com/1", "", "Backend role")
    b = make_job_id("Acme", "SWE Intern", "https://a.com/2", "", "Frontend role")
    store.save_application(a, "Acme", "SWE Intern", "note A", 80.0, db_path=db)
    store.save_application(b, "Acme", "SWE Intern", "note B", 60.0, db_path=db)
    assert len(store.list_applications(db)) == 2
    assert store.get_application(a, db)["cover_note"] == "note A"
    assert store.get_application(b, db)["cover_note"] == "note B"


def test_duplicate_job_updates_instead_of_inserting(db):
    jid = make_job_id("Acme", "SWE Intern", "https://a.com/1", "", "Backend")
    r1 = store.save_application(jid, "Acme", "SWE Intern", "v1", 70.0, db_path=db)
    r2 = store.save_application(jid, "Acme", "SWE Intern", "v2", 75.0, db_path=db)
    assert r1["action"] == "inserted" and r2["action"] == "updated"
    assert len(store.list_applications(db)) == 1
    assert store.get_application(jid, db)["cover_note"] == "v2"


def test_match_score_is_actually_persisted(db):
    jid = make_job_id("Acme", "ML Intern", "https://a.com/9", "", "ML")
    store.save_application(jid, "Acme", "ML Intern", "note", 83.0, priority="Apply Now", db_path=db)
    row = store.get_application(jid, db)
    assert row["match_score"] == 83.0        # the old dashboard always saved None
    assert row["priority"] == "Apply Now"


def test_legacy_schema_is_migrated_and_rows_preserved(db):
    legacy = os.path.join(os.path.dirname(db), "legacy.db")
    conn = sqlite3.connect(legacy)
    conn.execute("""CREATE TABLE applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT, title TEXT,
        match_score REAL, status TEXT, date_applied TEXT, cover_note TEXT)""")
    conn.execute("INSERT INTO applications (company, title, status, cover_note) VALUES (?,?,?,?)",
                 ("OldCo", "Old Role", "applied", "old note"))
    conn.commit(); conn.close()

    store.init_db(legacy)
    rows = store.list_applications(legacy)
    assert len(rows) == 1
    assert rows[0]["company"] == "OldCo"
    assert rows[0]["job_id"], "legacy row must be back-filled with a job_id"


def test_status_update_sets_date(db):
    jid = make_job_id("Acme", "ML Intern", "https://a.com/1", "", "ML")
    res = store.save_application(jid, "Acme", "ML Intern", db_path=db)
    store.update_status(res["id"], "applied", db)
    assert store.get_application(jid, db)["date_applied"]
