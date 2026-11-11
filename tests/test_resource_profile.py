from types import SimpleNamespace

from benchmark_runner import digest, postgres_service_profile


def test_resource_limits_are_fingerprinted_without_dynamic_cache_usage(monkeypatch):
    values = dict(ActiveState="active", ControlGroup="/test-postgres", MemoryHigh="4294967296",
                  MemoryMax="6442450944", MemorySwapMax="268435456", ManagedOOMPreference="avoid",
                  CPUWeight="100", IOWeight="100")

    def show(command, **kwargs):
        assert command[-2:] == ["--", "test-postgres.service"]
        assert kwargs["timeout"] == 5 and "shell" not in kwargs
        return SimpleNamespace(returncode=0, stdout="\n".join(f"{key}={value}" for key, value in values.items()))

    monkeypatch.setattr("benchmark_runner.subprocess.run", show)
    first = postgres_service_profile("test-postgres.service")
    assert first["status"] == "ok"
    assert "MemoryCurrent" not in first["properties"]
    values["MemoryMax"] = "8589934592"
    assert digest(first) != digest(postgres_service_profile("test-postgres.service"))


def test_missing_or_inactive_configured_service_is_not_valid_evidence(monkeypatch):
    monkeypatch.setattr("benchmark_runner.subprocess.run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout="ActiveState=inactive"))
    assert postgres_service_profile("missing.service")["status"] == "unavailable"
    assert postgres_service_profile(None) == {"status": "not_configured"}
