# regex_baseline on validation

| Metric | Value |
| --- | ---: |
| n_docs | 9915 |
| n_entities | 81770 |
| detection_recall | 0.3289 |
| char_precision | 0.9933 |
| char_recall | 0.4653 |
| type_macro_f1 | 0.3778 |
| type_accuracy | 0.8975 |
| ms_per_doc | 0.1291 |

## Per label

| Label | Support | Detection recall | Type F1 |
| --- | ---: | ---: | ---: |
| mac_address | 536 | 1.000 | 0.999 |
| ssn | 507 | 1.000 | 0.879 |
| date_of_birth | 1263 | 1.000 | 0.000 |
| ipv4 | 932 | 0.996 | 0.998 |
| email | 4982 | 0.992 | 0.999 |
| credit_debit_card | 911 | 0.978 | 0.895 |
| ipv6 | 504 | 0.974 | 0.792 |
| date_time | 1398 | 0.956 | 0.982 |
| url | 4382 | 0.950 | 0.998 |
| date | 7608 | 0.874 | 0.910 |
| fax_number | 681 | 0.860 | 0.000 |
| phone_number | 2471 | 0.829 | 0.835 |
| coordinate | 817 | 0.755 | 0.997 |
| time | 2134 | 0.628 | 0.886 |
| tax_id | 263 | 0.555 | 0.000 |
| device_identifier | 502 | 0.363 | 0.000 |
| national_id | 315 | 0.308 | 0.000 |
| account_number | 954 | 0.087 | 0.000 |
| health_plan_beneficiary_number | 643 | 0.053 | 0.000 |
| unique_id | 235 | 0.034 | 0.000 |
| medical_record_number | 694 | 0.027 | 0.000 |
| postcode | 842 | 0.015 | 0.917 |
| user_name | 1300 | 0.004 | 0.000 |
| api_key | 597 | 0.003 | 0.000 |
| license_plate | 729 | 0.003 | 0.000 |
| certificate_license_number | 736 | 0.003 | 0.000 |
| password | 812 | 0.002 | 0.000 |
| pin | 580 | 0.002 | 0.000 |
| race_ethnicity | 733 | 0.001 | 0.000 |
| customer_id | 1673 | 0.001 | 0.000 |
| company_name | 8451 | 0.000 | 0.000 |
| last_name | 4413 | 0.000 | 0.000 |
| religious_belief | 498 | 0.000 | - |
| sexuality | 228 | 0.000 | - |
| political_view | 396 | 0.000 | - |
| street_address | 1663 | 0.000 | - |
| swift_bic | 513 | 0.000 | - |
| state | 1881 | 0.000 | - |
| http_cookie | 571 | 0.000 | - |
| occupation | 3153 | 0.000 | - |
| language | 634 | 0.000 | - |
| age | 466 | 0.000 | - |
| gender | 449 | 0.000 | - |
| first_name | 5961 | 0.000 | - |
| employment_status | 716 | 0.000 | - |
| employee_id | 1047 | 0.000 | - |
| education_level | 1032 | 0.000 | - |
| cvv | 351 | 0.000 | - |
| county | 597 | 0.000 | - |
| country | 2848 | 0.000 | - |
| city | 1941 | 0.000 | - |
| blood_type | 475 | 0.000 | - |
| biometric_identifier | 1072 | 0.000 | - |
| bank_routing_number | 677 | 0.000 | - |
| vehicle_identifier | 1003 | 0.000 | - |

## Slice: locale

| Value | Docs | Detection recall | Char precision |
| --- | ---: | ---: | ---: |
| intl | 4957 | 0.338 | 0.993 |
| us | 4958 | 0.320 | 0.993 |

## Slice: document_format

| Value | Docs | Detection recall | Char precision |
| --- | ---: | ---: | ---: |
| structured | 4794 | 0.330 | 0.990 |
| unstructured | 5121 | 0.327 | 0.997 |

## Slice: domain

| Value | Docs | Detection recall | Char precision |
| --- | ---: | ---: | ---: |
| Agriculture | 248 | 0.333 | 0.996 |
| Automotive | 354 | 0.265 | 0.996 |
| Biotechnology | 326 | 0.321 | 0.998 |
| Chemicals | 194 | 0.331 | 0.997 |
| Civil Rights | 364 | 0.207 | 0.999 |
| Construction | 376 | 0.302 | 0.997 |
| Consumer Goods | 248 | 0.382 | 0.996 |
| Cybersecurity | 384 | 0.534 | 0.973 |
| Ecommerce | 346 | 0.372 | 0.984 |
| Education | 310 | 0.270 | 0.999 |
| Energy | 306 | 0.349 | 0.999 |
| Entertainment | 342 | 0.303 | 0.996 |
| Finance | 370 | 0.283 | 1.000 |
| Government | 336 | 0.288 | 0.997 |
| Healthcare | 370 | 0.251 | 0.985 |
| Hospitality | 306 | 0.465 | 0.996 |
| Identity Management | 380 | 0.344 | 0.996 |
| Identity Verification Services | 350 | 0.344 | 0.996 |
| Insurance | 316 | 0.316 | 0.996 |
| Legal | 341 | 0.216 | 0.992 |
| Logistics | 352 | 0.243 | 0.996 |
| Manufacturing | 230 | 0.326 | 0.997 |
| Media | 308 | 0.368 | 0.990 |
| Non-Profit | 360 | 0.353 | 0.998 |
| Pharmaceuticals | 276 | 0.301 | 0.998 |
| Real Estate | 356 | 0.217 | 0.997 |
| Retail | 310 | 0.350 | 0.996 |
| Technology | 423 | 0.526 | 0.989 |
| Telecommunications | 381 | 0.505 | 0.988 |
| Transportation | 352 | 0.250 | 0.992 |
