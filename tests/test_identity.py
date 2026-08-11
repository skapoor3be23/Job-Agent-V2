import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agent.identity import make_job_id, resume_hash


def test_same_company_title_different_url_are_different_jobs():
    a = make_job_id("Acme", "SWE Intern", "https://a.com/1", "Delhi", "Backend work")
    b = make_job_id("Acme", "SWE Intern", "https://a.com/2", "Delhi", "Backend work")
    assert a != b


def test_same_company_title_different_description_are_different_jobs():
    a = make_job_id("Acme", "SWE Intern", "", "Delhi", "You will build ML pipelines")
    b = make_job_id("Acme", "SWE Intern", "", "Delhi", "You will build Android apps")
    assert a != b


def test_duplicate_job_is_the_same_id_despite_cosmetic_differences():
    a = make_job_id("Acme Corp", "SWE Intern", "https://www.a.com/1?utm_source=x", "Delhi", "Work")
    b = make_job_id("  acme corp ", "swe intern", "http://a.com/1/", "delhi", "Work")
    assert a == b


def test_different_companies_same_title_are_different_jobs():
    a = make_job_id("Acme", "Data Intern", "", "Delhi", "Same text")
    b = make_job_id("Globex", "Data Intern", "", "Delhi", "Same text")
    assert a != b


def test_missing_fields_do_not_crash():
    assert len(make_job_id(None, None, None, None, None)) == 16


def test_resume_hash_is_stable_and_content_sensitive():
    assert resume_hash("Python, ML") == resume_hash("python  ml")
    assert resume_hash("Python") != resume_hash("Java")
