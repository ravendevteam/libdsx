> [!NOTE]
> All of our free software is designed to respect your privacy, while being as simple to use as possible. Our free software is licensed under the [BSD-3-Clause license](https://raventechnologiesgroup.com/BSD-3-Clause.txt). By using our software, you acknowledge and agree to the terms of the license.

libdsx is a Python library for working with Dossier (.dsx) document files. It supports reading existing documents, writing new or modified documents, and validating files against the DSX 1.1 specification. It provides access to document metadata and content, making it easy to incorporate DSX support into Python applications and tools.

## Installation

Requires Python 3.12 or later. Install from source:

```
git clone https://github.com/ravendevteam/libdsx.git
cd libdsx
python -m pip install .
```

## Examples

Create and save a document:

```python
from datetime import date

import libdsx

document = libdsx.Document(
    metadata=libdsx.Metadata(
        title="EXAMPLE",
        authors=("John Doe",),
        revision=1,
        date=date.today(),
    ),
    records=(
        libdsx.Heading(1, "GREETINGS"),
        libdsx.Paragraph((libdsx.Text("Hello, World!"),)),
    ),
)

libdsx.dump(document, "example.dsx")
```

Read the saved document and display its full text:

```python
import libdsx

document = libdsx.load("example.dsx")
print(libdsx.render(document), end="")
```

Validate a file and handle DSX errors:

```python
import libdsx

try:
    libdsx.validate_file("example.dsx")
except libdsx.DSXError as error:
    print(f"Invalid DSX document: {error}")
else:
    print("Valid DSX document")
```

Inspect metadata without loading the document body:

```python
import libdsx

inspection = libdsx.read_metadata("example.dsx")
print(inspection.metadata.title)
print(inspection.metadata.authors)
print(inspection.content_verified)
```

## Contributing
This repository is maintained as a read-only source mirror. We do not accept GitHub Issues or Pull Requests. If you would like to report a bug, request a feature or change, provide feedback, or suggest improvements, please submit your feedback through our [feedback form](https://raventechnologiesgroup.com/softwarefeedback).

Well-documented submissions help us review requests more quickly. Due to the volume of submissions, we cannot guarantee individual responses. All submissions are reviewed by the maintainers. If a request is accepted, it will be implemented internally and included in a future update to the repository.
