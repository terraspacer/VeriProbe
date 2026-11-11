"""Isolated JPype entry point for the pinned R-Bot Calcite implementation.

The parent owns the timeout and kills the whole worker process.  No R-Bot Python
modules are imported: those eagerly initialize embedding and model clients.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> None:
    request = json.load(sys.stdin)
    try:
        import jpype as jp
        jp.startJVM("-Xmx768m", classpath=[request["jar"]], convertStrings=True)
        # Calcite's native diagnostics must not corrupt the JSON protocol.
        sink = jp.JClass("java.io.PrintStream")(jp.JClass("java.io.FileOutputStream")(os.devnull))
        jp.JClass("java.lang.System").setOut(sink)
        jp.JClass("java.lang.System").setErr(sink)
        rewriter = jp.JClass("rewriter.Rewriter")
        array_list = jp.JClass("java.util.ArrayList")

        def jlist(values):
            result = array_list()
            for value in values:
                result.add(str(value))
            return result

        sql, schema = request["sql"], jlist(request["create_tables"])
        if request["operation"] == "match":
            matches = rewriter.matchAllRules(sql, schema, "PostgreSQL", False)
            output = {"status": "ok", "matches": [{str(entry.getKey()): str(entry.getValue()) for entry in row.entrySet()} for row in matches]}
        elif request["operation"] == "rewrite":
            results = []
            for sequence in request["rule_sequences"]:
                result = rewriter.rewrite(sql, schema, jlist(sequence), int(request.get("rounds", 1)), "PostgreSQL")
                results.append({"sql": None if result.sql is None else str(result.sql),
                                "rules": [str(r) for r in result.rules],
                                "java_time_ms": None if result.time is None else int(result.time)})
            output = {"status": "ok", "rewrites": results}
        else:
            raise ValueError("Unknown Calcite operation")
    except Exception as exc:
        output = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    print("MCTS_CALCITE_JSON=" + json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
