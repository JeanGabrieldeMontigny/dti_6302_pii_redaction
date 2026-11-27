# e1_small_cnn_subset on validation

| Metric | Value |
| --- | ---: |
| n_docs | 9915 |
| n_entities | 81770 |
| detection_recall | 0.5446 |
| char_precision | 0.9447 |
| char_recall | 0.8022 |
| type_macro_f1 | 0.5839 |
| type_accuracy | 0.7815 |
| ms_per_doc | 0.5298 |

## Per label

| Label | Support | Detection recall | Type F1 |
| --- | ---: | ---: | ---: |
| date_of_birth | 1263 | 1.000 | 0.938 |
| ssn | 507 | 1.000 | 0.828 |
| bank_routing_number | 677 | 0.996 | 0.776 |
| mac_address | 536 | 0.994 | 0.998 |
| credit_debit_card | 911 | 0.993 | 0.900 |
| phone_number | 2471 | 0.989 | 0.787 |
| biometric_identifier | 1072 | 0.986 | 0.908 |
| fax_number | 681 | 0.981 | 0.000 |
| vehicle_identifier | 1003 | 0.977 | 0.947 |
| date_time | 1398 | 0.976 | 0.969 |
| ipv6 | 504 | 0.970 | 0.995 |
| tax_id | 263 | 0.943 | 0.000 |
| device_identifier | 502 | 0.940 | 0.606 |
| medical_record_number | 694 | 0.938 | 0.184 |
| account_number | 954 | 0.928 | 0.492 |
| customer_id | 1673 | 0.922 | 0.662 |
| unique_id | 235 | 0.919 | 0.000 |
| health_plan_beneficiary_number | 643 | 0.914 | 0.540 |
| employee_id | 1047 | 0.891 | 0.838 |
| pin | 580 | 0.888 | 0.856 |
| date | 7608 | 0.885 | 0.976 |
| api_key | 597 | 0.841 | 0.799 |
| ipv4 | 932 | 0.833 | 0.991 |
| national_id | 315 | 0.822 | 0.000 |
| cvv | 351 | 0.783 | 0.969 |
| postcode | 842 | 0.768 | 0.880 |
| swift_bic | 513 | 0.764 | 0.929 |
| certificate_license_number | 736 | 0.751 | 0.240 |
| time | 2134 | 0.730 | 0.984 |
| coordinate | 817 | 0.671 | 0.990 |
| license_plate | 729 | 0.665 | 0.540 |
| password | 812 | 0.631 | 0.771 |
| country | 2848 | 0.550 | 0.713 |
| email | 4982 | 0.473 | 0.971 |
| state | 1881 | 0.426 | 0.308 |
| language | 634 | 0.405 | 0.124 |
| first_name | 5961 | 0.404 | 0.556 |
| age | 466 | 0.384 | 0.780 |
| last_name | 4413 | 0.362 | 0.437 |
| url | 4382 | 0.349 | 0.987 |
| user_name | 1300 | 0.344 | 0.587 |
| city | 1941 | 0.329 | 0.340 |
| http_cookie | 571 | 0.312 | 0.805 |
| company_name | 8451 | 0.215 | 0.616 |
| blood_type | 475 | 0.192 | 0.621 |
| county | 597 | 0.168 | 0.358 |
| race_ethnicity | 733 | 0.149 | 0.000 |
| street_address | 1663 | 0.121 | 0.621 |
| religious_belief | 498 | 0.088 | 0.000 |
| political_view | 396 | 0.043 | 0.000 |
| sexuality | 228 | 0.018 | 0.000 |
| gender | 449 | 0.004 | 0.000 |
| employment_status | 716 | 0.004 | 1.000 |
| occupation | 3153 | 0.001 | 0.000 |
| education_level | 1032 | 0.001 | 0.000 |

## Slice: locale

| Value | Docs | Detection recall | Char precision |
| --- | ---: | ---: | ---: |
| intl | 4957 | 0.555 | 0.946 |
| us | 4958 | 0.534 | 0.944 |

## Slice: document_format

| Value | Docs | Detection recall | Char precision |
| --- | ---: | ---: | ---: |
| structured | 4794 | 0.536 | 0.925 |
| unstructured | 5121 | 0.556 | 0.970 |

## Slice: domain

| Value | Docs | Detection recall | Char precision |
| --- | ---: | ---: | ---: |
| Agriculture | 248 | 0.548 | 0.947 |
| Automotive | 354 | 0.665 | 0.938 |
| Biotechnology | 326 | 0.522 | 0.951 |
| Chemicals | 194 | 0.533 | 0.904 |
| Civil Rights | 364 | 0.407 | 0.937 |
| Construction | 376 | 0.524 | 0.948 |
| Consumer Goods | 248 | 0.538 | 0.929 |
| Cybersecurity | 384 | 0.643 | 0.939 |
| Ecommerce | 346 | 0.611 | 0.928 |
| Education | 310 | 0.387 | 0.942 |
| Energy | 306 | 0.526 | 0.964 |
| Entertainment | 342 | 0.434 | 0.931 |
| Finance | 370 | 0.607 | 0.958 |
| Government | 336 | 0.537 | 0.940 |
| Healthcare | 370 | 0.590 | 0.919 |
| Hospitality | 306 | 0.617 | 0.946 |
| Identity Management | 380 | 0.618 | 0.955 |
| Identity Verification Services | 350 | 0.612 | 0.960 |
| Insurance | 316 | 0.584 | 0.963 |
| Legal | 341 | 0.471 | 0.955 |
| Logistics | 352 | 0.493 | 0.944 |
| Manufacturing | 230 | 0.502 | 0.957 |
| Media | 308 | 0.464 | 0.924 |
| Non-Profit | 360 | 0.482 | 0.951 |
| Pharmaceuticals | 276 | 0.567 | 0.925 |
| Real Estate | 356 | 0.482 | 0.958 |
| Retail | 310 | 0.599 | 0.953 |
| Technology | 423 | 0.642 | 0.950 |
| Telecommunications | 381 | 0.694 | 0.959 |
| Transportation | 352 | 0.585 | 0.960 |
