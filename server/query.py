import re


class InvalidDriveQuery(ValueError):
    """The supplied filter is outside Filo's supported Drive query grammar."""


_OPERATORS = {
    "name": {"contains", "=", "!="},
    "mimeType": {"contains", "=", "!="},
    "fullText": {"contains"},
    "modifiedTime": {"=", "!=", "<", ">", "<=", ">="},
    "createdTime": {"=", "!=", "<", ">", "<=", ">="},
}
_TOKEN = re.compile(r"""[ \t\r\n]+|'(?:[^'\\\x00-\x1f]|\\['\\])*'|[A-Za-z]+|!=|<=|>=|[=<>()]""")


def _tokens(query: str):
    pos = 0
    while pos < len(query):
        match = _TOKEN.match(query, pos)
        if not match:
            raise InvalidDriveQuery(
                f"Invalid input at {query[pos:pos + 32]!r}; expected a Drive query token "
                "or a complete single-quoted value."
            )
        token = match.group()
        if not token.isspace():
            yield token
        pos = match.end()


def validate_query(query: str) -> None:
    if not isinstance(query, str):
        raise InvalidDriveQuery(f"Expected a string filter, got {type(query).__name__}.")

    tokens = _tokens(query)
    depth = 0
    needs_predicate = True
    previous = None
    for token in tokens:
        if needs_predicate:
            if token == "(":
                depth += 1
            elif token == "not" and previous != "not":
                pass
            elif token in _OPERATORS:
                operator = next(tokens, "<end>")
                if operator not in _OPERATORS[token]:
                    allowed = ", ".join(sorted(_OPERATORS[token]))
                    raise InvalidDriveQuery(f"Got {operator!r} after {token}; expected one of {allowed}.")
                value = next(tokens, "<end>")
                if not value.startswith("'"):
                    raise InvalidDriveQuery(f"Got {value!r} after {operator}; expected a single-quoted value.")
                needs_predicate = False
            else:
                fields = ", ".join(_OPERATORS)
                raise InvalidDriveQuery(f"Got {token!r}; expected a group or a field ({fields}).")
        elif token in ("and", "or"):
            needs_predicate = True
        elif token == ")" and depth:
            depth -= 1
        else:
            expected = "'and', 'or', or ')'" if depth else "'and', 'or', or end of filter"
            raise InvalidDriveQuery(f"Got {token!r}; expected {expected}.")
        previous = token
    if needs_predicate:
        after = f" after {previous!r}" if previous else ""
        raise InvalidDriveQuery(f"Unexpected end of filter{after}; expected a predicate.")
    if depth:
        raise InvalidDriveQuery("Unexpected end of filter; expected ')'.")
