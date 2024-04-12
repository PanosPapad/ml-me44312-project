import seaborn as sns

from matplotlib import pyplot as plt


def cleanup_dataset(data):
    # Deleting columns that are not needed
    columns_to_remove = ['ID',
                         'Envir', 'Mobil', 'ResidCh', 'LifSty',  # Categorical variables deleted
                         'CostCarCHF', 'CalculatedIncome', 'CoderegionCAR', 'LangCode', 'OwnHouse', 'UrbRur',
                         # Variables with duplicates
                         'NbBicy', 'NbCellPhones', 'NbBicyChild', 'NbRoomsHouse', 'ClassifCodeLine',
                         # Variables not useful because subset of other variable
                         'NbTV', 'NewsPaperSubs',
                         'TimePT', 'TimeCar', 'MarginalCostPT', 'WaitingTimePT', 'WalkingTimePT', 'BirthYear', 'Weight', 'CostCar']  # Potentially irrelevant variables that may cause noise

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

    """A Heatmap to show the correlation between features"""
    # # High correlation threshold
    # threshold = 0.70
    # # Calculate the correlation matrix
    # corr = filtered_data.corr()
    #
    # # Plot the correlation heatmap
    # plt.figure(figsize=(40, 40))
    # sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title("Correlation Matrix of Features")
    # plt.show()
    #
    # # Extract highly correlated features
    # highly_correlated_features = set()
    # for i in range(len(corr.columns)):
    #     for j in range(i + 1, len(corr.columns)):
    #         if abs(corr.iloc[i, j]) >= threshold:
    #             colname_i = corr.columns[i]
    #             colname_j = corr.columns[j]
    #             highly_correlated_features.add(colname_i)
    #             highly_correlated_features.add(colname_j)
    #
    # print("Highly correlated features (>", threshold, "):", sorted(highly_correlated_features))

    return filtered_data
