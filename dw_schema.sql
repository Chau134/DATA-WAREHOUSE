CREATE TABLE dim_gender (
	gender_cd CHAR(1) PRIMARY KEY,
	gender_descr VARCHAR(255)
);

CREATE TABLE dim_race (
	race_cd CHAR(2) PRIMARY KEY,
	race_descr VARCHAR(255)
);

CREATE TABLE dim_hispanic (
	hispanic_cd CHAR(1) PRIMARY KEY,
	ethnicity_descr VARCHAR(255)
);

CREATE TABLE dim_patient (
	study_pat_id NUMBER(10) PRIMARY KEY,
	birth_date DATE,
	gender_cd CHAR(1),
	race_cd CHAR(2),
	hispanic_cd CHAR(1),
	language_descr VARCHAR(255),
	FOREIGN KEY (gender_cd) REFERENCES dim_gender(gender_cd),
	FOREIGN KEY (race_cd) REFERENCES dim_race(race_cd),
	FOREIGN KEY (hispanic_cd) REFERENCES dim_hispanic(hispanic_cd)
);

CREATE TABLE dim_time (
    time_id NUMBER PRIMARY KEY,
    epoch_index NUMBER,
    seconds_from_start NUMBER
);

CREATE TABLE fact_eeg_features (
    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    study_pat_id NUMBER,
    time_id NUMBER,

    mean NUMBER,
    variance NUMBER,
    skewness NUMBER,
    kurtosis NUMBER,

    power_delta NUMBER,
    power_theta NUMBER,
    power_alpha NUMBER,
    power_beta NUMBER,

    theta_alpha_ratio NUMBER,
    spectral_entropy NUMBER,
    rolling_var NUMBER,

    delta_diff NUMBER,
    theta_diff NUMBER,
    alpha_diff NUMBER,
    beta_diff NUMBER,

    delta_theta_ratio NUMBER,
    alpha_beta_ratio NUMBER,
    rolling_mean_delta NUMBER,
    rolling_mean_theta NUMBER,

    label NUMBER,

    FOREIGN KEY (study_pat_id) REFERENCES dim_patient(study_pat_id),
    FOREIGN KEY (time_id) REFERENCES dim_time(time_id)
);
