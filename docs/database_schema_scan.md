# SQLite Schema Scan

Database: `ecotox_clean.sqlite`
Table count: 9

## chemical_category_curated

- Rows: 18520
- Columns: 13

| Column | Type | Not Null | Primary Key | Default |
|---|---:|---:|---:|---|
| `cas_number` | `TEXT` | 0 | 0 | `` |
| `chemical_id` | `TEXT` | 0 | 0 | `` |
| `dtxsid` | `TEXT` | 0 | 0 | `` |
| `chemical_name` | `TEXT` | 0 | 0 | `` |
| `chemical_class_l1` | `TEXT` | 0 | 0 | `` |
| `chemical_class_l2` | `TEXT` | 0 | 0 | `` |
| `chemical_class_l3` | `TEXT` | 0 | 0 | `` |
| `use_source_class` | `TEXT` | 0 | 0 | `` |
| `structure_flags` | `TEXT` | 0 | 0 | `` |
| `chemical_class_confidence` | `REAL` | 0 | 0 | `` |
| `chemical_class_source` | `TEXT` | 0 | 0 | `` |
| `chemical_class_evidence` | `TEXT` | 0 | 0 | `` |
| `chemical_class_review_status` | `TEXT` | 0 | 0 | `` |

## chemical_smiles_dictionary

- Rows: 18520
- Columns: 11

| Column | Type | Not Null | Primary Key | Default |
|---|---:|---:|---:|---|
| `cas_number` | `TEXT` | 0 | 1 | `` |
| `casrn` | `TEXT` | 0 | 0 | `` |
| `chemical_name` | `TEXT` | 0 | 0 | `` |
| `dtxsid` | `TEXT` | 0 | 0 | `` |
| `smiles` | `TEXT` | 0 | 0 | `` |
| `connectivity_smiles` | `TEXT` | 0 | 0 | `` |
| `inchikey` | `TEXT` | 0 | 0 | `` |
| `smiles_source` | `TEXT` | 0 | 0 | `` |
| `smiles_match_method` | `TEXT` | 0 | 0 | `` |
| `query_status` | `TEXT` | 0 | 0 | `` |
| `remarks` | `TEXT` | 0 | 0 | `` |

## chemicals

- Rows: 18520
- Columns: 9

| Column | Type | Not Null | Primary Key | Default |
|---|---:|---:|---:|---|
| `cas_number` | `NUMERIC(12)` | 0 | 1 | `` |
| `chemical_name` | `VARCHAR(500)` | 0 | 0 | `` |
| `ecotox_group` | `VARCHAR(120)` | 0 | 0 | `` |
| `dtxsid` | `VARCHAR(20)` | 0 | 0 | `` |
| `smiles` | `TEXT` | 0 | 0 | `` |
| `molecular_weight_rdkit_g_mol` | `REAL` | 0 | 0 | `` |
| `molecular_weight_g_mol` | `REAL` | 0 | 0 | `` |
| `molecular_weight_source` | `TEXT` | 0 | 0 | `` |
| `molecular_weight_status` | `TEXT` | 0 | 0 | `` |

## references

- Rows: 131197
- Columns: 8

| Column | Type | Not Null | Primary Key | Default |
|---|---:|---:|---:|---|
| `reference_number` | `NUMERIC(6)` | 0 | 1 | `` |
| `reference_db` | `VARCHAR(3)` | 0 | 0 | `` |
| `reference_type` | `VARCHAR(10)` | 0 | 0 | `` |
| `author` | `VARCHAR(120)` | 0 | 0 | `` |
| `title` | `VARCHAR(220)` | 0 | 0 | `` |
| `source` | `VARCHAR(255)` | 0 | 0 | `` |
| `publication_year` | `VARCHAR(4)` | 0 | 0 | `` |
| `doi` | `VARCHAR(1000)` | 0 | 0 | `` |

## results

- Rows: 1234077
- Columns: 27

| Column | Type | Not Null | Primary Key | Default |
|---|---:|---:|---:|---|
| `result_id` | `NUMERIC(10)` | 0 | 1 | `` |
| `test_id` | `NUMERIC(11)` | 1 | 0 | `` |
| `obs_duration_mean_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `obs_duration_mean` | `VARCHAR(10)` | 0 | 0 | `` |
| `obs_duration_unit` | `VARCHAR(20)` | 0 | 0 | `` |
| `conc1_type` | `VARCHAR(20)` | 0 | 0 | `` |
| `conc1_mean_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `conc1_mean` | `VARCHAR(20)` | 0 | 0 | `` |
| `conc1_unit` | `VARCHAR(20)` | 0 | 0 | `` |
| `conc1_min_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `conc1_min` | `VARCHAR(20)` | 0 | 0 | `` |
| `conc1_max_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `conc1_max` | `VARCHAR(20)` | 0 | 0 | `` |
| `endpoint` | `VARCHAR(20)` | 0 | 0 | `` |
| `endpoint_comments` | `VARCHAR(4000)` | 0 | 0 | `` |
| `trend` | `VARCHAR(20)` | 0 | 0 | `` |
| `effect` | `VARCHAR(20)` | 0 | 0 | `` |
| `measurement` | `VARCHAR(20)` | 0 | 0 | `` |
| `response_site_comments` | `VARCHAR(4000)` | 0 | 0 | `` |
| `obs_duration_mean_h` | `REAL` | 0 | 0 | `` |
| `obs_duration_standardization_status` | `TEXT` | 0 | 0 | `` |
| `conc1_mean_standardized` | `REAL` | 0 | 0 | `` |
| `conc1_min_standardized` | `REAL` | 0 | 0 | `` |
| `conc1_max_standardized` | `REAL` | 0 | 0 | `` |
| `conc1_standard_unit` | `TEXT` | 0 | 0 | `` |
| `conc1_unit_family` | `TEXT` | 0 | 0 | `` |
| `conc1_standardization_status` | `TEXT` | 0 | 0 | `` |

## species

- Rows: 29598
- Columns: 25

| Column | Type | Not Null | Primary Key | Default |
|---|---:|---:|---:|---|
| `species_number` | `NUMERIC(10)` | 0 | 1 | `` |
| `common_name` | `VARCHAR(60)` | 0 | 0 | `` |
| `latin_name` | `VARCHAR(100)` | 0 | 0 | `` |
| `kingdom` | `VARCHAR(10)` | 0 | 0 | `` |
| `phylum_division` | `VARCHAR(35)` | 0 | 0 | `` |
| `subphylum_div` | `VARCHAR(35)` | 0 | 0 | `` |
| `superclass` | `VARCHAR(35)` | 0 | 0 | `` |
| `class` | `VARCHAR(35)` | 0 | 0 | `` |
| `tax_order` | `VARCHAR(35)` | 0 | 0 | `` |
| `family` | `VARCHAR(35)` | 0 | 0 | `` |
| `genus` | `VARCHAR(35)` | 0 | 0 | `` |
| `species` | `VARCHAR(35)` | 0 | 0 | `` |
| `subspecies` | `VARCHAR(35)` | 0 | 0 | `` |
| `variety` | `VARCHAR(35)` | 0 | 0 | `` |
| `ecotox_group` | `VARCHAR(120)` | 0 | 0 | `` |
| `ncbi_taxid` | `VARCHAR(10)` | 0 | 0 | `` |
| `primary_medium` | `TEXT` | 0 | 0 | `` |
| `habitat_labels` | `TEXT` | 0 | 0 | `` |
| `habitat_confidence` | `REAL` | 0 | 0 | `` |
| `habitat_evidence_tier` | `TEXT` | 0 | 0 | `` |
| `habitat_evidence_source` | `TEXT` | 0 | 0 | `` |
| `habitat_evidence_detail` | `TEXT` | 0 | 0 | `` |
| `habitat_decision_rule` | `TEXT` | 0 | 0 | `` |
| `habitat_review_status` | `TEXT` | 0 | 0 | `` |
| `habitat_annotation_date` | `TEXT` | 0 | 0 | `` |

## species_category_curated

- Rows: 29598
- Columns: 21

| Column | Type | Not Null | Primary Key | Default |
|---|---:|---:|---:|---|
| `species_number` | `INTEGER` | 0 | 0 | `` |
| `latin_name` | `TEXT` | 0 | 0 | `` |
| `common_name` | `TEXT` | 0 | 0 | `` |
| `kingdom` | `TEXT` | 0 | 0 | `` |
| `phylum_division` | `TEXT` | 0 | 0 | `` |
| `class_name` | `TEXT` | 0 | 0 | `` |
| `tax_order` | `TEXT` | 0 | 0 | `` |
| `family` | `TEXT` | 0 | 0 | `` |
| `genus` | `TEXT` | 0 | 0 | `` |
| `species` | `TEXT` | 0 | 0 | `` |
| `raw_ecotox_group` | `TEXT` | 0 | 0 | `` |
| `taxon_group_l1` | `TEXT` | 0 | 0 | `` |
| `taxon_group_l2` | `TEXT` | 0 | 0 | `` |
| `taxon_group_l3` | `TEXT` | 0 | 0 | `` |
| `is_standard_test_species` | `INTEGER` | 0 | 0 | `` |
| `is_us_invasive_species` | `INTEGER` | 0 | 0 | `` |
| `is_us_threatened_endangered` | `INTEGER` | 0 | 0 | `` |
| `taxon_group_confidence` | `REAL` | 0 | 0 | `` |
| `taxon_group_source` | `TEXT` | 0 | 0 | `` |
| `taxon_group_evidence` | `TEXT` | 0 | 0 | `` |
| `taxon_group_review_status` | `TEXT` | 0 | 0 | `` |

## species_habitat_annotations

- Rows: 29598
- Columns: 28

| Column | Type | Not Null | Primary Key | Default |
|---|---:|---:|---:|---|
| `species_number` | `INTEGER` | 0 | 1 | `` |
| `common_name` | `TEXT` | 0 | 0 | `` |
| `latin_name` | `TEXT` | 0 | 0 | `` |
| `kingdom` | `TEXT` | 0 | 0 | `` |
| `phylum_division` | `TEXT` | 0 | 0 | `` |
| `class_name` | `TEXT` | 0 | 0 | `` |
| `tax_order` | `TEXT` | 0 | 0 | `` |
| `family` | `TEXT` | 0 | 0 | `` |
| `genus` | `TEXT` | 0 | 0 | `` |
| `species` | `TEXT` | 0 | 0 | `` |
| `ecotox_group` | `TEXT` | 0 | 0 | `` |
| `ncbi_taxid` | `TEXT` | 0 | 0 | `` |
| `primary_medium` | `TEXT` | 0 | 0 | `` |
| `habitat_labels` | `TEXT` | 0 | 0 | `` |
| `confidence` | `REAL` | 0 | 0 | `` |
| `evidence_tier` | `TEXT` | 0 | 0 | `` |
| `evidence_source` | `TEXT` | 0 | 0 | `` |
| `evidence_detail` | `TEXT` | 0 | 0 | `` |
| `decision_rule` | `TEXT` | 0 | 0 | `` |
| `review_status` | `TEXT` | 0 | 0 | `` |
| `water_test_count` | `INTEGER` | 0 | 0 | `` |
| `soil_test_count` | `INTEGER` | 0 | 0 | `` |
| `nonsoil_test_count` | `INTEGER` | 0 | 0 | `` |
| `total_habitat_test_count` | `INTEGER` | 0 | 0 | `` |
| `worms_aphia_id` | `TEXT` | 0 | 0 | `` |
| `worms_url` | `TEXT` | 0 | 0 | `` |
| `worms_environment_flags` | `TEXT` | 0 | 0 | `` |
| `annotation_date` | `TEXT` | 0 | 0 | `` |

## tests

- Rows: 724182
- Columns: 30

| Column | Type | Not Null | Primary Key | Default |
|---|---:|---:|---:|---|
| `test_id` | `NUMERIC(11)` | 0 | 1 | `` |
| `reference_number` | `NUMERIC(6)` | 1 | 0 | `` |
| `test_cas` | `NUMERIC(12)` | 1 | 0 | `` |
| `species_number` | `NUMERIC(10)` | 1 | 0 | `` |
| `test_purity_mean_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `test_purity_mean` | `VARCHAR(10)` | 0 | 0 | `` |
| `test_purity_min_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `test_purity_min` | `VARCHAR(10)` | 0 | 0 | `` |
| `test_purity_max_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `test_purity_max` | `VARCHAR(10)` | 0 | 0 | `` |
| `organism_habitat` | `VARCHAR(11)` | 1 | 0 | `` |
| `organism_lifestage` | `VARCHAR(20)` | 0 | 0 | `` |
| `exposure_duration_mean_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `exposure_duration_mean` | `VARCHAR(10)` | 0 | 0 | `` |
| `exposure_duration_unit` | `VARCHAR(20)` | 0 | 0 | `` |
| `exposure_duration_min_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `exposure_duration_min` | `VARCHAR(10)` | 0 | 0 | `` |
| `exposure_duration_max_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `exposure_duration_max` | `VARCHAR(10)` | 0 | 0 | `` |
| `media_type` | `VARCHAR(20)` | 0 | 0 | `` |
| `num_doses_mean_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `num_doses_mean` | `VARCHAR(10)` | 0 | 0 | `` |
| `num_doses_min_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `num_doses_min` | `VARCHAR(10)` | 0 | 0 | `` |
| `num_doses_max_op` | `VARCHAR(2)` | 0 | 0 | `` |
| `num_doses_max` | `VARCHAR(10)` | 0 | 0 | `` |
| `exposure_duration_mean_h` | `REAL` | 0 | 0 | `` |
| `exposure_duration_min_h` | `REAL` | 0 | 0 | `` |
| `exposure_duration_max_h` | `REAL` | 0 | 0 | `` |
| `exposure_duration_standardization_status` | `TEXT` | 0 | 0 | `` |
