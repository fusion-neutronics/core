# PNNL material list

Every material in `yamc.materials.pnnl`, with the entry number from
[PNNL-15870, Rev. 2](https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-15870Rev2.pdf)
so a composition can be cited back to the report.

Names are the report's own, verbatim. Pass one straight to the collection:

```python
import yamc

steel = yamc.materials.pnnl["Steel, Stainless 304"]
```

See [Material collections](materials.md#material-collections) for search,
overrides and how the compositions are stored. Composition values are atom
fractions; an element symbol is expanded with yamc's natural abundances,
while an explicit nuclide is used as given.

The table below is generated from the bundled data when these docs are
built, so it always matches the shipped collection.

```python exec="1"
import yamc

pnnl = yamc.materials.pnnl

print(f"**{len(pnnl)} materials.** Source: {pnnl.citation}.\n")
print("| # | Name | Density (g/cm³) | Formula | Composition (atom fractions) |")
print("|---|------|-----------------|---------|------------------------------|")
for name in pnnl:
    e = pnnl.entry(name)
    comp = ", ".join(f"{k} {v:g}" for k, v in e["composition"].items())
    formula = e["formula"] or ""
    # Escape the pipe and underscore characters that would otherwise break
    # the table or start emphasis.
    safe = lambda s: str(s).replace("|", "\\|")
    print(
        f"| {e['number']} | {safe(name)} | {e['density']:g} | "
        f"{safe(formula)} | {safe(comp)} |"
    )
```
