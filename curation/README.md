# Curation overlay

`decisions.csv` is applied on top of the immutable raw data when it is loaded.
Dataset version = first 8 hex digits of the sha256 of the lines `sample_id,action,new_label`
(`digits.data.dataset_version`): reason, reviewer, timestamp and round do not change it.

| Column      | Meaning                                                        |
|-------------|----------------------------------------------------------------|
| sample_id   | Stable ID, `train-00000`…`train-59999`, `test-0000`…`test-9999` |
| action      | `keep`, `relabel`, `drop`, `unsure`                            |
| new_label   | 0–9, only for `relabel`                                        |
| reason      | Free text                                                      |
| reviewer    | Who decided                                                    |
| timestamp   | ISO 8601                                                       |
| round       | Curation round number                                          |

Only `relabel` and `drop` change the data; `keep` records that the sample was reviewed.

Decisions are made in the workbench (`/review`, `/curation`); it keeps them in
`decisions.sqlite` here (not committed) and writes this CSV on "Export to decisions.csv".
Commit the CSV after each review session.

## Decision rules (fix these before the first round)

- **relabel** only if the true digit is unambiguous to a human.
- **drop** if unreadable, not a digit, or two digits.
- **keep** if ambiguous but the label is plausible.
- **unsure** for a second look later.
