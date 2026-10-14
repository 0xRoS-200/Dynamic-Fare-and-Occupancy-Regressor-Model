def test_check_hdfs_local_fallback_always_true(tmp_path, monkeypatch):
    """With local_fallback=True and the directory pre-created, check_hdfs returns True."""
    # monkeypatch: change cwd to tmp_path, create hdfs_local/transit/raw inside it
    import os
    os.makedirs(tmp_path / "hdfs_local" / "transit" / "raw", exist_ok=True)
    monkeypatch.chdir(tmp_path)
    from infra.check_infra import check_hdfs
    assert check_hdfs(local_fallback=True) is True

def test_check_kafka_bad_broker_returns_false():
    """check_kafka with an unreachable broker returns False, never raises."""
    from infra.check_infra import check_kafka
    result = check_kafka(broker="localhost:1", topic="transit_live")
    assert result is False

