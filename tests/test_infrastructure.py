import psycopg

from benchmark_runner import evaluate_holdout, infrastructure_failure
from mcts_rewrite.db import Postgres


def test_failed_connection_is_distinct_from_a_query_timeout(monkeypatch):
    def unavailable(*args, **kwargs):
        raise psycopg.OperationalError("test connection unavailable")
    monkeypatch.setattr(psycopg, "connect", unavailable)
    result = Postgres("dbname=unused").compare("SELECT 1", "SELECT 1")
    assert result["equivalent"] is False
    assert result["error_type"] == "OperationalError"
    assert infrastructure_failure({"verification": {"original": {"execution": result}}})
    assert not infrastructure_failure({"status": "timeout", "error_type": "QueryCanceled", "sqlstate": "57014"})
    assert not infrastructure_failure({"status": "error", "error_type": "SyntaxError", "sqlstate": "42601"})


def test_server_shutdown_and_connection_sqlstates_stop_execution():
    assert infrastructure_failure({"trace": [{"sqlstate": "57P01"}]})
    assert infrastructure_failure({"evaluation": {"rbot": {"sqlstate": "08006"}}})
    assert not infrastructure_failure({"sql": "SELECT 'connection refused'"})


def test_holdout_preserves_connection_failure_for_run_stop():
    class Disconnected:
        def measure(self, *args, **kwargs):
            raise psycopg.OperationalError("server disconnected during holdout")

    results, schedule = evaluate_holdout(Disconnected(), {"original": "SELECT 1"}, seed=0)
    assert results["original"]["median_ms"] is None
    assert results["original"]["error_type"] == "OperationalError"
    assert infrastructure_failure(results)
    assert len(schedule) == 1
