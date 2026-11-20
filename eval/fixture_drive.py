"""Small, read-only Drive API double for the committed folder tree."""

import re
import sqlite3
from types import SimpleNamespace

from server.query import _tokens


class FixtureDrive:
    def __init__(self, rows):
        self.rows = {row["id"]: row for row in rows}

    def files(self):
        return self

    def get(self, *, fileId, fields):
        def execute():
            row = self.rows.get(fileId, {})
            return {**row, "id": row.get("id"), "parents": [row["parent"]] if row else []}
        return SimpleNamespace(execute=execute)

    def list(self, *, q, fields, pageToken=None, pageSize=100):
        def execute():
            # A connection per call also supports LangGraph's parallel tool workers.
            with sqlite3.connect(":memory:") as db:
                db.execute("CREATE TABLE files (id, parent, name, mimeType, fullText, "
                           "modifiedTime, createdTime, trashed)")
                for row in self.rows.values():
                    db.execute("INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                        row["id"], row["parent"], row["name"], row["mimeType"],
                        row["name"] + " " + row.get("fullText", ""),
                        row.get("modifiedTime"), row.get("createdTime"), row.get("trashed", False),
                    ))
                clause, values = fixture_filter(q)
                ids = [r[0] for r in db.execute("SELECT id FROM files WHERE " + clause, values)]
            start = int(pageToken or 0)
            end = start + pageSize
            keys = fields.split("files(", 1)[1].rstrip(")").replace(" ", "").split(",")
            page = {"files": [{k: self.rows[i][k] for k in keys if k in self.rows[i]}
                              for i in ids[start:end]]}
            if end < len(ids):
                page["nextPageToken"] = str(end)
            return page
        return SimpleNamespace(execute=execute)


def fixture_filter(query):
    """Translate only Filo's filter subset. Bind every string as a parameter."""
    tokens = iter(_tokens(query))
    sql, values = [], []
    for token in tokens:
        if token in {"(", ")", "and", "or", "not"}:
            sql.append(token)
        elif token.startswith("'"):
            if (next(tokens), next(tokens)) != ("in", "parents"):
                raise ValueError("Unsupported fixture parent predicate")
            sql.append("parent = ?")
            values.append(token[1:-1])
        elif token in {"name", "fullText", "mimeType", "createdTime", "modifiedTime", "trashed"}:
            op, literal = next(tokens), next(tokens)
            value = re.sub(r"\\(.)", r"\1", literal[1:-1]) if literal.startswith("'") else 0
            if op == "contains":
                # Fixture approximates Drive text indexing with case-insensitive substring matches.
                sql.append(f"instr(lower({token}), lower(?)) > 0")
            elif op in {"=", "!=", ">", "<", ">=", "<="}:
                sql.append(f"{token} {op} ?")
            else:
                raise ValueError(f"Unsupported fixture operator {op}")
            values.append(value)
        else:
            raise ValueError(f"Unsupported fixture token {token}")
    return " ".join(sql), values
