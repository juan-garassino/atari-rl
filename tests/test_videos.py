from __future__ import annotations


class _StubClient:
    def __init__(self, tmp):
        self.tmp = tmp

    def get_experiment_by_name(self, name):
        return type("E", (), {"experiment_id": "7"})()

    def search_runs(self, ids, flt, max_results=1):
        info = type("I", (), {"run_id": "abc"})()
        return [type("R", (), {"info": info})()]

    def download_artifacts(self, run_id, path, dst_path):
        if path == "best_model.zip":
            out = self.tmp / "dl" / "best_model.zip"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"model")
            return str(out)
        out = self.tmp / "dl" / path
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_0000250000.mp4").write_bytes(b"x")
        return str(out)


def test_fetch_videos_copies_new_files(tmp_path):
    from atari_rl.status import fetch_videos

    dest = tmp_path / "Videos"
    got = fetch_videos("pong-1", dest, client=_StubClient(tmp_path))
    assert [p.name for p in got] == ["eval_0000250000.mp4"]
    assert (dest / "pong-1" / "eval_0000250000.mp4").exists()
    assert (dest / "pong-1" / "best_model.zip").read_bytes() == b"model"
    assert fetch_videos("pong-1", dest, client=_StubClient(tmp_path)) == []  # already downloaded
