# Assumptions and discovered facts

- The supplied PS3 datasets are treated as immutable source material and are read from their official folder.
- Door sampling interval is inferred from parsed timestamps; 20 ms is observed, never forced.
- ACV cars and parameters are parsed from each workbook's own headers.
- Corrugation sampling is 10,000 Hz and files contain one rotational-speed plus 128 sensor columns, per the Info Kit and inspected header.
- SHM traces are currently headerless, single-column numeric CSVs; file numbers are arbitrary identifiers.
- S-N material constants are unavailable, so SHM cycle features are explicitly fatigue proxies.
- Health statuses are decision-support demonstrations, not approved maintenance rules.

