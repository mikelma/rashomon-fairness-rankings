import pandas as pd
from tyro.extras import SubcommandApp

app = SubcommandApp()


@app.command()
def compas(
    src_csv: str = "data/compas-scores-raw.csv",
    out_csv: str = "data/compas-processed.csv",
):
    df = pd.read_csv(src_csv)

    df = df[
        (df["Ethnic_Code_Text"] == "African-American")
        | (df["Ethnic_Code_Text"] == "Caucasian")
    ]

    # get the raw scores and the IDs
    id_scores = df.groupby("Person_ID", sort=False)["RawScore"].mean()
    new = id_scores.reset_index()

    df = df.drop_duplicates(subset="Person_ID")

    # text categories to number IDs (binary in this case)
    new["race"] = pd.Categorical(df["Ethnic_Code_Text"]).codes
    new["sex"] = pd.Categorical(df["Sex_Code_Text"]).codes

    new = new.rename(columns={"Person_ID": "id", "RawScore": "score"})

    # random shuffle
    new = new.sample(frac=1)
    print(new)

    new.to_csv(out_csv)


@app.command()
def law_data(
    src_csv: str = "data/law_data_raw.csv",
    out_csv: str = "data/law_data_processed.csv",
):
    df = pd.read_csv(src_csv)
    print(df)

    new = pd.DataFrame()

    # change score from higher the better to lower the better
    new["score"] = 1 / df["LSAT"]

    # make the categories binary
    new["sex"] = pd.Categorical(df["sex"]).codes
    new["race"] = pd.Categorical(df["race"] != "White").codes

    # random shuffle
    new = new.sample(frac=1)
    print(new)

    new.to_csv(out_csv)


@app.command()
def astronauts(
    src_csv: str = "data/astronauts.csv",
    out_csv: str = "data/astronauts_processed.csv",
):
    df = pd.read_csv(src_csv)
    print(df)

    new = pd.DataFrame()

    # change score from higher the better to lower the better
    new["score"] = 1 / (df["Space Flight (hr)"] + 1)

    # make the categories binary
    new["sex"] = pd.Categorical(df["Gender"]).codes

    # random shuffle
    new = new.sample(frac=1)
    print(new)

    new.to_csv(out_csv)


@app.command()
def sat(
    src_csv: str = "data/satgpa.csv",
    out_csv: str = "data/satgpa_processed.csv",
):
    df = pd.read_csv(src_csv)
    print(df)

    new = pd.DataFrame()

    # change score from higher the better to lower the better
    new["score"] = 1 / df["sat_sum"]

    # make the categories binary
    new["sex"] = pd.Categorical(df["sex"]).codes

    # random shuffle
    new = new.sample(frac=1)
    print(new)

    new.to_csv(out_csv)


@app.command()
def hmda(
    src_csv: str = "data/hmda.csv",
    out_csv: str = "data/hmda_processed.csv",
):
    df = pd.read_csv(src_csv)

    new = pd.DataFrame()

    # normalize and change score from higher the better to lower the better
    s = df["loan_amount"]
    new["score"] = 1 - ((s - s.min()) / (s.max() - s.min()))

    # make the categories binary
    new["sex"] = pd.Categorical(df["applicant_sex"]).codes
    new["race"] = pd.Categorical(df["derived_race"] != "White").codes

    # random shuffle
    new = new.sample(frac=1)

    new.to_csv(out_csv)


if __name__ == "__main__":
    app.cli()
