# Reference outputs

Files here are not golden dumps. A golden dump is a `path -> value` map that
`tests/golden.rs` compares whole; these are single artefacts that a unit test
checks against directly, and they live apart so the golden harness does not
try to parse them.

| File | What it is |
|---|---|
| `njoy-deck.txt.xz` | The NJOY input deck the **Python** `make_ace` composes for `n-095_Am_244` at 293.6 K and 900 K, captured with its `run` stubbed out. `njoy::tests::composes_the_same_deck_as_the_python_package` holds the Rust deck to it byte for byte. |

Regenerate the deck with:

```python
import endf, lzma, unittest.mock as mock
from endf import njoy

material = endf.Material('tests/n-095_Am_244.endf.xz')
captured = {}
def capture(commands, tapein, tapeout, **kwargs):
    captured['commands'] = commands
    raise SystemExit(0)
with mock.patch.object(njoy, 'run', capture):
    try:
        njoy.make_ace('tests/n-095_Am_244.endf.xz', temperatures=[293.6, 900.0],
                      material=material, output_dir='.')
    except SystemExit:
        pass
open('crates/endf/tests/reference/njoy-deck.txt.xz', 'wb').write(
    lzma.compress(captured['commands'].encode(), preset=9))
```
