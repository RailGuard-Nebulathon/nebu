# Data contracts

The supplied Info Kits are authoritative and source files are immutable.

## Door

`Train.csv` (18,036×17 observed) and `Test.csv` (6,253×17 observed) are continuous streams. The
timestamp format is `YYYY-M-D-H-M-S-ms`; interval statistics are derived rather than fixing 20 ms.
Continuous channels are current, voltage, back-EMF, open/close timers, and leaf position. Ten
command/switch/state columns are retained separately. `Train_Segments_Answer.csv` has
`segment_id,start_time,end_time,operation,status,n_rows`; all 110 official slices match exactly.
Test contains 38 cycles discovered from validated timestamp discontinuities, with a state-based
fallback when discontinuities are absent.

## ACV

Each XLSX case has identifying columns (`Car model`, `Train number`, `Time`) and variable per-car
columns named `Car NN - parameter`. Six files have one known `faulty_car`; leading-zero IDs are
strings. Most workbooks have 67 columns and one richer case has 483. No fixed parameter schema is
assumed. `data/manifests/acv_schema_mapping.yaml` records every discovered sheet/car/parameter.

## Rail corrugation

Each CSV is 10,000 rows (1 second at 10 kHz) by 129 columns: rotational-speed input plus vibration
and shock for 8 positions on each of 8 cars. Odd positions are Side I and even positions Side II.
Labels are exactly `Normal`, `Side I`, and `Side II` (234/14/24 train files respectively).

## SHM

Each source CSV is a headerless numeric dynamic-stress trace; inspected train files contain one
channel and 581,120 rows. `Train_Labels.csv` maps 64 arbitrary filenames to continuous `damage`.
Filename numbers are not temporal or target features. Material S-N constants are not supplied,
so rainflow outputs are fatigue proxies unless the user explicitly configures constants.

Machine-readable schema reports and SHA-256 manifests are written under `data/manifests` and never
alter sources. Canonical `SequenceSample` values are `[time, channel]`; raw subsystem shapes remain
task-specific until adapter/model boundaries.
