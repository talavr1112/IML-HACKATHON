import numpy as np
import pandas as pd
import re
import pickle

# CONSTANTS
DATE_FORMAT_REGEX = '[0-9]{2}/[0-9]{2}/[0-9]{4}'


# FUNCTIONS

def get_time_date(date):
    """
    This function returns time in date format
    :param date:
    :return:
    """
    time_stamp = pd.to_datetime(date)

    return pd.DataFrame({"Time": time_stamp.apply(lambda x: x.hour * 60 + x.minute),
                         "Weekday": time_stamp.apply(lambda x: x.weekday())})


def blocks_by_frequency(data):
    """
    Create features by common blocks.
    :param data:
    :return:
    """
    data['52BLOCK'] = (data['Block'] == "001XX N STATE ST")
    data['44BLOCK'] = (data['Block'] == "0000X W TERMINAL ST")
    data['31BLOCK'] = (data['Block'] == "0000X E ROOSEVELT RD")
    data['26BLOCKA'] = (data['Block'] == "100XX W OHARE ST")
    data['26BLOCKB'] = (data['Block'] == "011XX S CLARK ST")
    data['25BLOCKA'] = (data['Block'] == "064XX S DR MARTIN LUTHER KING JR DR")
    data['25BLOCKB'] = (data['Block'] == "083XX S STEWART AVE")
    data['23BLOCK'] = (data['Block'] == "008XX N MICHIGAN AVE")
    data.drop("Block", axis=1, inplace=True)
    return data


def location_desc(data):
    """
    Create features by common location descriptions.
    :param data:
    :return:
    """
    loc_desc = data['Location Description']

    data['ISAPARTMENT'] = (loc_desc == "APARTMENT")
    data['ISRES'] = (loc_desc == "RESIDENCE")
    data['ISSTREET'] = (loc_desc == "STREET")
    data['ISSIDEWALK'] = (loc_desc == "SIDEWALK")
    data['ISRESIDENCE'] = (loc_desc == "RESIDENCE - PORCH / HALLWAY")
    data['ISDEP'] = (loc_desc == "DEPARTMENT STORE")
    data['ISALL'] = (loc_desc == "ALLEY")
    data['ISREST'] = (loc_desc == "RESTAURANT")
    data['ISCOM'] = (loc_desc == "COMMERCIAL / BUSINESS OFFICE")
    data['ISGROC'] = (loc_desc == "GROCERY")
    data['ISRES2'] = (loc_desc == "RESIDENCE - YARD (FRONT / BACK)")
    data['ISVENIC'] = (loc_desc == "VEHICLE NON-COMMERCIAL")
    data['ISGAS'] = (loc_desc == "GAS STATION")
    return data


def drop_unnecessary(data):
    """
    Drop unnecessary featurs.
    :param data:
    :return:
    """
    data.drop(["ID", "Case Number", "IUCR", "Description", "FBI Code", "Updated On", "Latitude", "Longitude",
               "Location", "Year", "Date"], axis=1, inplace=True)


def handle_description(data):
    """
    Get unique location descriptions
    :param data:
    :return:
    """
    unique_descriptions = pd.unique(data["Location Description"])
    num_range = np.arange(len(unique_descriptions))
    x = pd.Series(num_range, index=unique_descriptions).to_dict()
    return data.replace(x)


def handle_xcoordinate(data):
    """
    This function handles the case where the xcoordinate has no value
    :param data:
    :return:
    """
    handle_coordinate(data, 'X Coordinate')


def handle_ycoordinate(data):
    handle_coordinate(data, 'Y Coordinate')


def preprocess_data(data, pred=False):
    arrest_mode = data.mode()['Arrest'][0]
    domestic_mode = data.mode()['Domestic'][0]
    DATE_REGEX = re.compile(DATE_FORMAT_REGEX)
    mean_year = int(data["Year"].mean())
    template_date = "03/01/" + str(mean_year)
    map_response_rev = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2,
                        "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}

    def handle_empty_date(row):
        if pd.isna(row):
            return template_date
        else:
            f_match = re.match(DATE_REGEX, row)
            if f_match is None:
                return template_date
        return row

    def handle_empty_loc_desc(row):
        if pd.isna(row):
            return "OTHER (SPECIFY)"
        return row

    def handle_arrest(row):
        if pd.isna(row):
            return arrest_mode
        return row

    def handle_domestic(row):
        if pd.isna(row):
            return domestic_mode
        return row

    def handle_locations(row):
        if pd.isna(row):
            return '-1'
        return row

    def apply_primary(row):
        return map_response_rev[row]

    def handle_primary_type(data):
        columns = ["THEFT", "ASSAULT", "BATTERY", "CRIMINAL DAMAGE", "DECEPTIVE PRACTICE"]
        primary_type_val = data['Primary Type']
        data = data.loc[np.bitwise_or.reduce([primary_type_val == col for col in columns])]
        data['Primary Type'] = data['Primary Type'].apply(apply_primary)

    columns_to_apply_func = {'Date': handle_empty_date,
                             'Location Description': handle_empty_loc_desc,
                             'Arrest': handle_arrest,
                             'Domestic': handle_domestic,
                             'District': handle_locations,
                             'Ward': handle_locations,
                             'Beat': handle_locations,
                             'Community Area': handle_locations}

    for col, func in columns_to_apply_func.items():
        data[col] = data[col].apply(func)

    handle_xcoordinate(data)
    handle_ycoordinate(data)

    data = pd.concat([data, get_time_date(data["Date"])], axis=1)
    drop_unnecessary(data)
    data = blocks_by_frequency(data)
    data = handle_description(data)
    data = location_desc(data)

    if pred:
        return data.to_numpy()

    # primary type
    handle_primary_type(data)

    Y = data["Primary Type"]
    X = data.drop("Primary Type", axis=1)

    return X.to_numpy(), Y.to_numpy()


def save_model(model):
    """
    This function saves the model.
    :param model:
    :return:
    """
    with open("model.sav", "wb") as model_fd:
        pickle.dump(model, model_fd)


def load_model():
    """
    This function loads the model
    :return:
    """
    with open("model.sav", "rb") as model_fd:
        m = pickle.load(model_fd)
    return m


# HELPER FUNCTIONS


def handle_coordinate_helper(matcher, total, sum_not_nulls, coordinate_name):
    """
    :param coordinate_name:
    :param matcher:
    :param total:
    :param sum_not_nulls:
    :return:
    """
    if len(matcher) != 0:
        update_sums(matcher, total, sum_not_nulls, coordinate_name)


def update_sums(matcher, total, sum_not_nulls, coordinate_name):
    """
    :param matcher:
    :param total:
    :param sum_not_nulls:
    :param coordinate_name:
    :return: updated sum
    """
    total += 1
    sum_not_nulls += matcher[coordinate_name].mean()


def handle_coordinate(data, coordinate_name):
    """
    This function handles the case where x or y coordinate is null- in this case the value
    will be replaced by the mean of this column.
    :param data:
    :param coordinate_name:
    :return:
    """
    nulls = data[data[coordinate_name].isnull()]
    not_nulls = data[data[coordinate_name].isnull() == False]
    indexes = data.index[data[coordinate_name].isnull()].tolist()
    indexCounter = 0
    for i in range(len(nulls)):
        keys = ["District", "Ward", "Beat", "Community Area"]
        data_columns_map = {key: nulls[key].iloc[i] for key in keys}
        cond_arr = [not_nulls[key] == data_columns_map[key] for key in data_columns_map]
        matcher = not_nulls.loc[np.bitwise_and.reduce(cond_arr)]
        # if we found a match then calculate the mean of those samples at the coordination.
        if len(matcher) != 0:
            data.at[indexes[indexCounter], coordinate_name] = matcher[coordinate_name].mean()
            indexCounter += 1
            continue

        sum_not_nulls, total = 0, 0
        matcher = not_nulls.loc[(not_nulls['District'] == data_columns_map['District'])]

        for key in keys:
            handle_coordinate_helper(matcher, total, sum_not_nulls, coordinate_name)
            matcher = not_nulls.loc[(not_nulls[key] == data_columns_map[key])]

        handle_coordinate_helper(matcher, total, sum_not_nulls, coordinate_name)
        if total != 0:
            data.at[indexes[indexCounter], coordinate_name] = sum_not_nulls / total
            indexCounter += 1
            continue
        data.at[indexes[indexCounter], coordinate_name] = data[coordinate_name].mean()
        indexCounter += 1
