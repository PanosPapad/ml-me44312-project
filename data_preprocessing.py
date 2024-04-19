import pandas as pd


def cleanup_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes specific features from the data set and removes entries with no choice specified.

    :param data: The ModeChoiceOptima data set as a DataFrame.
    :return:     DataFrame without the removed features and entries.
    """
    # Deleting columns that are not needed
    columns_to_remove = ['ID',
                         'Envir', 'Mobil', 'ResidCh', 'LifSty',
                         # Attitude variables deleted
                         'CostCarCHF', 'CalculatedIncome', 'CoderegionCAR', 'LangCode', 'OwnHouse', 'UrbRur',
                         # Variables with duplicates
                         'NbBicy', 'NbCellPhones', 'NbBicyChild', 'NbRoomsHouse', 'ClassifCodeLine',
                         # Variables not useful because subset of other variable
                         'NbTV', 'NewsPaperSubs',
                         # Potentially irrelevant variables that may cause noise
                         'TimePT', 'TimeCar', 'MarginalCostPT', 'WaitingTimePT',
                         'WalkingTimePT', 'BirthYear', 'Weight', 'CostCar',
                         # Part of the first set of correlation removals
                         'CostPT', 'InVehicleTime', 'NbChild', 'ReportedDuration']
                         # Part of the second set of correlation removals

    # CostCarCHF is a duplicate of CostCar
    # CalculatedIncome is a duplicate of Income
    # CoderegionCAR is a duplicate of Region
    # LangCode is a duplicate of Region
    # OwnHouse is a duplicate of HouseType
    # UrbRur is a duplicate of TypeCommune

    # NbBicy is not too useful. NbHoushold is the only one correlated with it with >0.7.
    # NbBicyChild is not too useful. We have NbChild
    # NbCellPhones is kinda random too. NbHoushold might cover it (it is the one with corr >0.7). Also there is NbSmartPhone
    # NbRoomsHouse is not too useful. NbHoushold might cover it (it is the one with corr >0.7)
    # ClassifCodeLine is not very useful. We have other bus info.

    filtered_data = data.drop(
        columns=[col for col in data.columns if any(string in col for string in columns_to_remove)])
    # Save the filtered data to a new CSV file

    # Remove all rows without a mode choice
    filtered_data = filtered_data[filtered_data['Choice'] != -1]

    # Save the filtered data to a new CSV file
    filtered_data.to_csv('filtered_file.csv', index=False)

    return filtered_data


def descr_data(df: pd.DataFrame, latex: bool = False) -> None:
    """
    Prints the statistical values of a DataFrame using the describe() method,
    possibly as a latex table.

    :param df:    The DataFrame which should be described.
    :param latex: Boolean denoting if the description should be printed as a latex table.
    :return:      None.
    """
    # Turn floats into strings with two decimals
    out = df.describe().applymap(lambda f: format(f, '.2f').rstrip('0').rstrip('.')).T
    # Print DataFrame or latex version of DataFrame
    print(out.style.to_latex() if latex else out)
