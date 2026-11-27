from src import tracking


def test_git_info_from_repository():
    info = tracking.git_info()
    assert info["git_dirty"] in {"true", "false", "unknown"}
    assert info["git_commit"]


def test_git_info_falls_back_to_build_args(monkeypatch):
    """Inside the pipeline image there is no git: use the commit baked in at build time."""
    monkeypatch.setattr(tracking, "_git", lambda *args: None)
    monkeypatch.setenv("GIT_COMMIT", "abc1234")
    monkeypatch.setenv("GIT_DIRTY", "false")
    assert tracking.git_info() == {"git_commit": "abc1234", "git_dirty": "false"}
    monkeypatch.delenv("GIT_COMMIT")
    monkeypatch.delenv("GIT_DIRTY")
    assert tracking.git_info() == {"git_commit": "unknown", "git_dirty": "unknown"}
