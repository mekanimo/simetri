import asyncio
import os
import pathlib
from pathlib import Path

from multilspy import LanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger

# Editor coordinates are 1-based. LSP positions are 0-based.
# shapes/shape.py:736 is `def merge`.
# EDITOR_LINE = 736
# EDITOR_COLUMN = 9
EDITOR_LINE = 490
EDITOR_COLUMN = 10


async def main():
    config = MultilspyConfig.from_dict({"code_language": "python"})
    logger = MultilspyLogger()
    project_path = str(Path("C:/uv_simetri_3.9/simetri/src/simetri").resolve())
    # relative_path = "shapes/shape.py"
    relative_path = "shapes/shape.py"
    line = EDITOR_LINE - 1
    column = EDITOR_COLUMN - 1

    server = LanguageServer.create(config, logger, project_path)
    async with server.start_server():
        # jedi-language-server returns null when there are no locations.
        # multilspy.request_references asserts that the result is a list.
        uri = pathlib.Path(os.path.join(project_path, relative_path)).as_uri()
        with server.open_file(relative_path):
            response = await server.server.send.references(
                {
                    "context": {"includeDeclaration": False},
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": column},
                }
            )

        references = response or []
        print(
            f"{relative_path}:{EDITOR_LINE}:{EDITOR_COLUMN} -> {len(references)} references"
        )
        for ref in references:
            start = ref["range"]["start"]
            print(f"Reference: {ref['uri']} at line {start['line'] + 1}")


asyncio.run(main())
