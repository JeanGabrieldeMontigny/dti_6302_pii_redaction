# Data validation: processed

**Result:** PASSED (0 errors, 1 warnings)

## Checks

| Check | Split | Severity | Result | Observed | Threshold |
| --- | --- | --- | --- | --- | --- |
| `schema.required_columns` | train | error | pass | ok | uid, domain, document_type, document_format, locale, text, spans |
| `schema.string_dtypes` | train | error | pass | ok | - |
| `missing.no_nulls` | train | error | pass | 0 | 0 |
| `missing.no_blank_strings` | train | error | pass | 0 | 0 |
| `domain.locale_values` | train | error | pass | ok | intl, us |
| `domain.document_format_values` | train | error | pass | ok | structured, unstructured |
| `stat.text_length_in_range` | train | error | pass | 0 | 20, 20000 |
| `domain.records_per_uid` | train | error | pass | 2 | 2 |
| `domain.known_labels` | train | error | pass | 0 | 0 |
| `domain.span_in_bounds` | train | error | pass | 0 | 0 |
| `domain.spans_sorted_non_overlapping` | train | error | pass | 0 | 0 |
| `stat.no_duplicate_texts` | train | error | pass | 0 | 0 |
| `schema.required_columns` | validation | error | pass | ok | uid, domain, document_type, document_format, locale, text, spans |
| `schema.string_dtypes` | validation | error | pass | ok | - |
| `missing.no_nulls` | validation | error | pass | 0 | 0 |
| `missing.no_blank_strings` | validation | error | pass | 0 | 0 |
| `domain.locale_values` | validation | error | pass | ok | intl, us |
| `domain.document_format_values` | validation | error | pass | ok | structured, unstructured |
| `stat.text_length_in_range` | validation | error | pass | 0 | 20, 20000 |
| `domain.records_per_uid` | validation | error | pass | 2 | 2 |
| `domain.known_labels` | validation | error | pass | 0 | 0 |
| `domain.span_in_bounds` | validation | error | pass | 0 | 0 |
| `domain.spans_sorted_non_overlapping` | validation | error | pass | 0 | 0 |
| `stat.no_duplicate_texts` | validation | error | pass | 0 | 0 |
| `schema.required_columns` | test | error | pass | ok | uid, domain, document_type, document_format, locale, text, spans |
| `schema.string_dtypes` | test | error | pass | ok | - |
| `missing.no_nulls` | test | error | pass | 0 | 0 |
| `missing.no_blank_strings` | test | error | pass | 0 | 0 |
| `domain.locale_values` | test | error | pass | ok | intl, us |
| `domain.document_format_values` | test | error | pass | ok | structured, unstructured |
| `stat.text_length_in_range` | test | error | pass | 0 | 20, 20000 |
| `domain.records_per_uid` | test | error | pass | 2 | 2 |
| `domain.known_labels` | test | error | pass | 0 | 0 |
| `domain.span_in_bounds` | test | error | pass | 0 | 0 |
| `domain.spans_sorted_non_overlapping` | test | error | pass | 0 | 0 |
| `stat.no_duplicate_texts` | test | error | pass | 0 | 0 |
| `leakage.uid_overlap` | test|train | error | pass | 0 | 0 |
| `leakage.text_overlap` | test|train | error | pass | 0 | 0 |
| `leakage.uid_overlap` | test|validation | error | pass | 0 | 0 |
| `leakage.text_overlap` | test|validation | error | pass | 0 | 0 |
| `leakage.uid_overlap` | train|validation | error | pass | 0 | 0 |
| `leakage.text_overlap` | train|validation | error | pass | 0 | 0 |
| `stat.validation_fraction` | train|validation | error | pass | 0.0992 | 0.1 |
| `drift.label_distribution_train_vs_validation` | train|validation | warning | pass | 0.0163 | 0.1 |
| `drift.label_distribution_train_vs_test` | train|test | warning | warn | 0.122 | 0.1 |

## Split summary

| | train | validation | test |
| --- | --- | --- | --- |
| docs | 90070 | 9915 | 99990 |
| unique_uids | 45041 | 4959 | 50000 |
| spans | 743438 | 81770 | 850168 |
| spans_per_doc | 8.25 | 8.25 | 8.5 |
| domains | 30 | 30 | 29 |
| document_types | 1450 | 1337 | 1466 |
| locale | us: 45037, intl: 45033 | us: 4958, intl: 4957 | us: 49997, intl: 49993 |
| document_format | unstructured: 46239, structured: 43831 | unstructured: 5121, structured: 4794 | unstructured: 50577, structured: 49413 |
| text_chars | p0: 32, p5: 256, p50: 774, p95: 2533, p100: 13260, mean: 1003.6 | p0: 65, p5: 248, p50: 763, p95: 2553, p100: 10028, mean: 1001.3 | p0: 45, p5: 250, p50: 745, p95: 2431, p100: 14608, mean: 958.5 |

## Label distribution (share of spans)

| Label | train | validation | test |
| --- | ---: | ---: | ---: |
| company_name | 76,973 (10.4%) | 8,451 (10.3%) | 54,821 (6.4%) |
| date | 69,507 (9.3%) | 7,608 (9.3%) | 73,830 (8.7%) |
| first_name | 53,412 (7.2%) | 5,961 (7.3%) | 84,030 (9.9%) |
| email | 44,942 (6.0%) | 4,982 (6.1%) | 53,921 (6.3%) |
| url | 40,411 (5.4%) | 4,382 (5.4%) | 37,844 (4.5%) |
| last_name | 39,528 (5.3%) | 4,413 (5.4%) | 59,587 (7.0%) |
| occupation | 27,924 (3.8%) | 3,153 (3.9%) | 36,872 (4.3%) |
| country | 24,931 (3.4%) | 2,848 (3.5%) | 23,467 (2.8%) |
| phone_number | 22,532 (3.0%) | 2,471 (3.0%) | 23,925 (2.8%) |
| time | 18,896 (2.5%) | 2,134 (2.6%) | 24,505 (2.9%) |
| city | 17,369 (2.3%) | 1,941 (2.4%) | 18,344 (2.2%) |
| state | 17,176 (2.3%) | 1,881 (2.3%) | 18,306 (2.2%) |
| customer_id | 15,535 (2.1%) | 1,673 (2.0%) | 20,500 (2.4%) |
| street_address | 15,166 (2.0%) | 1,663 (2.0%) | 16,877 (2.0%) |
| date_time | 12,964 (1.7%) | 1,398 (1.7%) | 13,220 (1.6%) |
| user_name | 11,564 (1.6%) | 1,300 (1.6%) | 15,798 (1.9%) |
| date_of_birth | 11,447 (1.5%) | 1,263 (1.5%) | 18,077 (2.1%) |
| employee_id | 11,015 (1.5%) | 1,047 (1.3%) | 8,878 (1.0%) |
| biometric_identifier | 10,109 (1.4%) | 1,072 (1.3%) | 11,379 (1.3%) |
| education_level | 9,587 (1.3%) | 1,032 (1.3%) | 9,374 (1.1%) |
| account_number | 9,042 (1.2%) | 954 (1.2%) | 16,690 (2.0%) |
| vehicle_identifier | 8,334 (1.1%) | 1,003 (1.2%) | 4,561 (0.5%) |
| credit_debit_card | 8,138 (1.1%) | 911 (1.1%) | 12,866 (1.5%) |
| ipv4 | 8,020 (1.1%) | 932 (1.1%) | 6,078 (0.7%) |
| coordinate | 7,621 (1.0%) | 817 (1.0%) | 7,658 (0.9%) |
| password | 7,473 (1.0%) | 812 (1.0%) | 6,985 (0.8%) |
| postcode | 7,187 (1.0%) | 842 (1.0%) | 6,279 (0.7%) |
| certificate_license_number | 7,184 (1.0%) | 736 (0.9%) | 3,002 (0.4%) |
| employment_status | 6,937 (0.9%) | 716 (0.9%) | 11,018 (1.3%) |
| fax_number | 6,674 (0.9%) | 681 (0.8%) | 6,423 (0.8%) |
| license_plate | 6,404 (0.9%) | 729 (0.9%) | 4,078 (0.5%) |
| medical_record_number | 6,401 (0.9%) | 694 (0.8%) | 11,098 (1.3%) |
| race_ethnicity | 6,396 (0.9%) | 733 (0.9%) | 9,243 (1.1%) |
| bank_routing_number | 6,082 (0.8%) | 677 (0.8%) | 8,352 (1.0%) |
| pin | 5,966 (0.8%) | 580 (0.7%) | 6,456 (0.8%) |
| language | 5,895 (0.8%) | 634 (0.8%) | 8,651 (1.0%) |
| county | 5,860 (0.8%) | 597 (0.7%) | 8,460 (1.0%) |
| health_plan_beneficiary_number | 5,702 (0.8%) | 643 (0.8%) | 10,558 (1.2%) |
| mac_address | 5,556 (0.7%) | 536 (0.7%) | 4,250 (0.5%) |
| api_key | 5,489 (0.7%) | 597 (0.7%) | 4,667 (0.5%) |
| swift_bic | 5,306 (0.7%) | 513 (0.6%) | 5,557 (0.7%) |
| http_cookie | 5,213 (0.7%) | 571 (0.7%) | 4,891 (0.6%) |
| device_identifier | 4,751 (0.6%) | 502 (0.6%) | 2,507 (0.3%) |
| ssn | 4,430 (0.6%) | 507 (0.6%) | 6,062 (0.7%) |
| ipv6 | 4,289 (0.6%) | 504 (0.6%) | 3,139 (0.4%) |
| religious_belief | 4,242 (0.6%) | 498 (0.6%) | 6,253 (0.7%) |
| blood_type | 4,042 (0.5%) | 475 (0.6%) | 5,539 (0.7%) |
| gender | 3,855 (0.5%) | 449 (0.5%) | 7,645 (0.9%) |
| age | 3,696 (0.5%) | 466 (0.6%) | 7,166 (0.8%) |
| political_view | 3,615 (0.5%) | 396 (0.5%) | 6,402 (0.8%) |
| cvv | 3,203 (0.4%) | 351 (0.4%) | 4,843 (0.6%) |
| national_id | 2,853 (0.4%) | 315 (0.4%) | 2,846 (0.3%) |
| tax_id | 2,300 (0.3%) | 263 (0.3%) | 1,301 (0.2%) |
| unique_id | 2,152 (0.3%) | 235 (0.3%) | 1,777 (0.2%) |
| sexuality | 2,142 (0.3%) | 228 (0.3%) | 3,312 (0.4%) |
