# Data validation: raw

**Result:** PASSED (0 errors, 1 warnings)

## Checks

| Check | Split | Severity | Result | Observed | Threshold |
| --- | --- | --- | --- | --- | --- |
| `schema.required_columns` | train | error | pass | ok | uid, domain, document_type, document_description, document_format, locale, text, spans, text_tagged |
| `schema.string_dtypes` | train | error | pass | ok | - |
| `missing.no_nulls` | train | error | pass | 0 | 0 |
| `missing.no_blank_strings` | train | error | pass | 0 | 0 |
| `domain.locale_values` | train | error | pass | ok | intl, us |
| `domain.document_format_values` | train | error | pass | ok | structured, unstructured |
| `stat.text_length_in_range` | train | error | pass | 0 | 20, 20000 |
| `domain.records_per_uid` | train | error | pass | 2 | 2 |
| `schema.spans_parseable` | train | error | pass | 0 | 0 |
| `schema.span_fields` | train | error | pass | 0 | 0 |
| `domain.known_labels` | train | error | pass | ok | 55 labels |
| `domain.span_in_bounds` | train | error | pass | 0 | 0 |
| `domain.span_text_matches_offsets` | train | error | pass | 0 | 0 |
| `stat.case_mismatch_span_rate` | train | error | pass | 0.008202 | 0.02 |
| `stat.overlap_doc_rate` | train | error | pass | 0.00014 | 0.002 |
| `stat.empty_doc_rate` | train | error | pass | 6e-05 | 0.001 |
| `stat.duplicate_text_rate` | train | error | pass | 1e-05 | 0.001 |
| `schema.required_columns` | test | error | pass | ok | uid, domain, document_type, document_description, document_format, locale, text, spans, text_tagged |
| `schema.string_dtypes` | test | error | pass | ok | - |
| `missing.no_nulls` | test | error | pass | 0 | 0 |
| `missing.no_blank_strings` | test | error | pass | 0 | 0 |
| `domain.locale_values` | test | error | pass | ok | intl, us |
| `domain.document_format_values` | test | error | pass | ok | structured, unstructured |
| `stat.text_length_in_range` | test | error | pass | 0 | 20, 20000 |
| `domain.records_per_uid` | test | error | pass | 2 | 2 |
| `schema.spans_parseable` | test | error | pass | 0 | 0 |
| `schema.span_fields` | test | error | pass | 0 | 0 |
| `domain.known_labels` | test | error | pass | ok | 55 labels |
| `domain.span_in_bounds` | test | error | pass | 0 | 0 |
| `domain.span_text_matches_offsets` | test | error | pass | 0 | 0 |
| `stat.case_mismatch_span_rate` | test | error | pass | 0.007285 | 0.02 |
| `stat.overlap_doc_rate` | test | error | pass | 0.0001 | 0.002 |
| `stat.empty_doc_rate` | test | error | pass | 1e-05 | 0.001 |
| `stat.duplicate_text_rate` | test | error | pass | 0.0 | 0.001 |
| `stat.min_spans_per_label` | train | error | pass | ok | 1000 |
| `leakage.uid_overlap` | train|test | error | pass | 0 | 0 |
| `leakage.text_overlap` | train|test | error | pass | 0 | 0 |
| `drift.label_distribution_train_vs_test` | train|test | warning | warn | 0.1217 | 0.1 |

## Split summary

| | train | test |
| --- | --- | --- |
| docs | 100000 | 100000 |
| unique_uids | 50000 | 50000 |
| spans | 825456 | 850340 |
| spans_per_doc | 8.25 | 8.5 |
| domains | 30 | 29 |
| document_types | 1451 | 1466 |
| locale | us: 50000, intl: 50000 | us: 50000, intl: 50000 |
| document_format | unstructured: 51364, structured: 48636 | unstructured: 50578, structured: 49422 |
| text_chars | p0: 32, p5: 256, p50: 774, p95: 2536, p100: 13260, mean: 1003.4 | p0: 45, p5: 250, p50: 745, p95: 2431, p100: 14608, mean: 958.5 |
| span_quality | spans: 825456, case_mismatch_spans: 6770, non_string_span_text: 5824, empty_docs: 6, overlap_docs: 14, edge_whitespace_spans: 2 | spans: 850340, case_mismatch_spans: 6195, non_string_span_text: 6747, overlap_docs: 10, edge_whitespace_spans: 4, empty_docs: 1 |

## Label distribution (share of spans)

| Label | train | test |
| --- | ---: | ---: |
| company_name | 85,441 (10.4%) | 54,837 (6.4%) |
| date | 77,130 (9.3%) | 73,867 (8.7%) |
| first_name | 59,391 (7.2%) | 84,043 (9.9%) |
| email | 49,934 (6.0%) | 53,930 (6.3%) |
| url | 44,803 (5.4%) | 37,847 (4.5%) |
| last_name | 43,962 (5.3%) | 59,596 (7.0%) |
| occupation | 31,105 (3.8%) | 36,888 (4.3%) |
| country | 27,781 (3.4%) | 23,475 (2.8%) |
| phone_number | 25,012 (3.0%) | 23,930 (2.8%) |
| time | 21,054 (2.6%) | 24,506 (2.9%) |
| city | 19,313 (2.3%) | 18,347 (2.2%) |
| state | 19,067 (2.3%) | 18,318 (2.2%) |
| customer_id | 17,211 (2.1%) | 20,502 (2.4%) |
| street_address | 16,830 (2.0%) | 16,879 (2.0%) |
| date_time | 14,368 (1.7%) | 13,221 (1.6%) |
| user_name | 12,872 (1.6%) | 15,800 (1.9%) |
| date_of_birth | 12,710 (1.5%) | 18,079 (2.1%) |
| employee_id | 12,067 (1.5%) | 8,878 (1.0%) |
| biometric_identifier | 11,188 (1.4%) | 11,379 (1.3%) |
| education_level | 10,633 (1.3%) | 9,378 (1.1%) |
| account_number | 9,998 (1.2%) | 16,693 (2.0%) |
| vehicle_identifier | 9,339 (1.1%) | 4,561 (0.5%) |
| credit_debit_card | 9,049 (1.1%) | 12,867 (1.5%) |
| ipv4 | 8,954 (1.1%) | 6,078 (0.7%) |
| coordinate | 8,438 (1.0%) | 7,659 (0.9%) |
| password | 8,285 (1.0%) | 6,985 (0.8%) |
| postcode | 8,030 (1.0%) | 6,279 (0.7%) |
| certificate_license_number | 7,920 (1.0%) | 3,002 (0.4%) |
| employment_status | 7,654 (0.9%) | 11,018 (1.3%) |
| fax_number | 7,357 (0.9%) | 6,423 (0.8%) |
| license_plate | 7,134 (0.9%) | 4,078 (0.5%) |
| race_ethnicity | 7,130 (0.9%) | 9,248 (1.1%) |
| medical_record_number | 7,095 (0.9%) | 11,098 (1.3%) |
| bank_routing_number | 6,759 (0.8%) | 8,354 (1.0%) |
| pin | 6,548 (0.8%) | 6,456 (0.8%) |
| language | 6,531 (0.8%) | 8,651 (1.0%) |
| county | 6,462 (0.8%) | 8,465 (1.0%) |
| health_plan_beneficiary_number | 6,345 (0.8%) | 10,558 (1.2%) |
| mac_address | 6,096 (0.7%) | 4,250 (0.5%) |
| api_key | 6,086 (0.7%) | 4,667 (0.5%) |
| swift_bic | 5,821 (0.7%) | 5,559 (0.7%) |
| http_cookie | 5,786 (0.7%) | 4,892 (0.6%) |
| device_identifier | 5,255 (0.6%) | 2,507 (0.3%) |
| ssn | 4,937 (0.6%) | 6,062 (0.7%) |
| ipv6 | 4,794 (0.6%) | 3,139 (0.4%) |
| religious_belief | 4,740 (0.6%) | 6,257 (0.7%) |
| blood_type | 4,517 (0.5%) | 5,539 (0.7%) |
| gender | 4,304 (0.5%) | 7,645 (0.9%) |
| age | 4,162 (0.5%) | 7,167 (0.8%) |
| political_view | 4,013 (0.5%) | 6,402 (0.8%) |
| cvv | 3,554 (0.4%) | 4,843 (0.6%) |
| national_id | 3,171 (0.4%) | 2,847 (0.3%) |
| tax_id | 2,563 (0.3%) | 1,302 (0.2%) |
| unique_id | 2,387 (0.3%) | 1,777 (0.2%) |
| sexuality | 2,370 (0.3%) | 3,312 (0.4%) |

## Examples of quality issues

- **train / case_mismatch:** `[{"in_text": "Spanish", "span_text": "spanish"}, {"in_text": "Puerto Rican", "span_text": "puerto rican"}, {"in_text": "environmentalist", "span_text": "Environmentalist"}]`
- **train / overlap:** `[{"uid": "c756917b870e42c0954e2d99e203c417"}, {"uid": "7ca161aaee79400c975530a712727fb8"}, {"uid": "b320e1b3cdfa472daf2ccfb68b409d52"}]`
- **test / case_mismatch:** `[{"in_text": "Black", "span_text": "black"}, {"in_text": "Mexican", "span_text": "mexican"}, {"in_text": "atheist", "span_text": "Atheist"}]`
- **test / overlap:** `[{"uid": "75fbc8c1962e4798ab75d20138a90cb3"}, {"uid": "605b5ee19f41454392f0be9bbb993321"}, {"uid": "8f4446f81f464f3690fc0d1fa0ac7973"}]`
