# e2_dilated_cnn on validation

| Metric | Value |
| --- | ---: |
| n_docs | 9915 |
| n_entities | 81770 |
| detection_recall | 0.9486 |
| char_precision | 0.9845 |
| char_recall | 0.9698 |
| type_macro_f1 | 0.9805 |
| type_accuracy | 0.9861 |
| ms_per_doc | 3.9521 |

## Per label

| Label | Support | Detection recall | Type F1 |
| --- | ---: | ---: | ---: |
| bank_routing_number | 677 | 1.000 | 0.996 |
| date_of_birth | 1263 | 1.000 | 0.989 |
| credit_debit_card | 911 | 1.000 | 0.999 |
| ssn | 507 | 1.000 | 0.978 |
| mac_address | 536 | 1.000 | 1.000 |
| health_plan_beneficiary_number | 643 | 1.000 | 0.988 |
| fax_number | 681 | 1.000 | 0.866 |
| biometric_identifier | 1072 | 0.999 | 0.998 |
| certificate_license_number | 736 | 0.999 | 0.985 |
| phone_number | 2471 | 0.998 | 0.964 |
| url | 4382 | 0.997 | 0.999 |
| employee_id | 1047 | 0.997 | 0.995 |
| tax_id | 263 | 0.996 | 0.931 |
| swift_bic | 513 | 0.996 | 0.994 |
| medical_record_number | 694 | 0.996 | 0.981 |
| api_key | 597 | 0.995 | 0.978 |
| vehicle_identifier | 1003 | 0.994 | 0.997 |
| account_number | 954 | 0.994 | 0.980 |
| email | 4982 | 0.993 | 0.999 |
| device_identifier | 502 | 0.992 | 0.962 |
| coordinate | 817 | 0.991 | 0.996 |
| license_plate | 729 | 0.990 | 0.994 |
| customer_id | 1673 | 0.989 | 0.992 |
| country | 2848 | 0.988 | 0.991 |
| unique_id | 235 | 0.983 | 0.868 |
| ipv6 | 504 | 0.980 | 0.998 |
| last_name | 4413 | 0.977 | 0.988 |
| ipv4 | 932 | 0.976 | 0.999 |
| gender | 449 | 0.976 | 1.000 |
| date_time | 1398 | 0.975 | 0.987 |
| county | 597 | 0.975 | 0.991 |
| national_id | 315 | 0.975 | 0.962 |
| first_name | 5961 | 0.971 | 0.983 |
| street_address | 1663 | 0.970 | 0.997 |
| password | 812 | 0.969 | 0.996 |
| postcode | 842 | 0.969 | 0.990 |
| company_name | 8451 | 0.964 | 0.994 |
| language | 634 | 0.962 | 0.954 |
| state | 1881 | 0.962 | 0.942 |
| city | 1941 | 0.955 | 0.934 |
| date | 7608 | 0.953 | 0.996 |
| pin | 580 | 0.952 | 0.986 |
| http_cookie | 571 | 0.949 | 0.994 |
| user_name | 1300 | 0.931 | 0.979 |
| sexuality | 228 | 0.930 | 0.998 |
| race_ethnicity | 733 | 0.928 | 0.919 |
| employment_status | 716 | 0.927 | 0.993 |
| cvv | 351 | 0.926 | 0.998 |
| religious_belief | 498 | 0.926 | 0.981 |
| time | 2134 | 0.910 | 0.997 |
| education_level | 1032 | 0.888 | 1.000 |
| political_view | 396 | 0.886 | 0.968 |
| blood_type | 475 | 0.874 | 0.999 |
| age | 466 | 0.770 | 0.993 |
| occupation | 3153 | 0.406 | 0.992 |

## Slice: locale

| Value | Docs | Detection recall | Char precision |
| --- | ---: | ---: | ---: |
| intl | 4957 | 0.949 | 0.984 |
| us | 4958 | 0.948 | 0.985 |

## Slice: document_format

| Value | Docs | Detection recall | Char precision |
| --- | ---: | ---: | ---: |
| structured | 4794 | 0.934 | 0.981 |
| unstructured | 5121 | 0.967 | 0.988 |

## Slice: domain

| Value | Docs | Detection recall | Char precision |
| --- | ---: | ---: | ---: |
| Agriculture | 248 | 0.962 | 0.989 |
| Automotive | 354 | 0.964 | 0.988 |
| Biotechnology | 326 | 0.957 | 0.990 |
| Chemicals | 194 | 0.969 | 0.989 |
| Civil Rights | 364 | 0.926 | 0.984 |
| Construction | 376 | 0.945 | 0.981 |
| Consumer Goods | 248 | 0.956 | 0.985 |
| Cybersecurity | 384 | 0.950 | 0.982 |
| Ecommerce | 346 | 0.961 | 0.985 |
| Education | 310 | 0.905 | 0.971 |
| Energy | 306 | 0.973 | 0.994 |
| Entertainment | 342 | 0.931 | 0.979 |
| Finance | 370 | 0.960 | 0.977 |
| Government | 336 | 0.945 | 0.976 |
| Healthcare | 370 | 0.934 | 0.987 |
| Hospitality | 306 | 0.961 | 0.983 |
| Identity Management | 380 | 0.968 | 0.990 |
| Identity Verification Services | 350 | 0.962 | 0.984 |
| Insurance | 316 | 0.950 | 0.988 |
| Legal | 341 | 0.939 | 0.981 |
| Logistics | 352 | 0.967 | 0.991 |
| Manufacturing | 230 | 0.951 | 0.991 |
| Media | 308 | 0.911 | 0.983 |
| Non-Profit | 360 | 0.934 | 0.983 |
| Pharmaceuticals | 276 | 0.955 | 0.983 |
| Real Estate | 356 | 0.955 | 0.985 |
| Retail | 310 | 0.953 | 0.980 |
| Technology | 423 | 0.939 | 0.986 |
| Telecommunications | 381 | 0.964 | 0.989 |
| Transportation | 352 | 0.958 | 0.989 |
