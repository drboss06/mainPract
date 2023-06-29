import numpy as np


def generate_data_partitions(df, id_column, batch_num):
    possible_start_indexes = np.linspace(0, df.shape[0], num=batch_num + 1, dtype=int)[:-1]
    # Set indexes to the beginnings of the event traces
    correct_start_indexes = [possible_start_indexes[0]]
    for p_index in possible_start_indexes[1:]:
        ind = p_index
        while df[id_column].iloc[ind - 1] == df[id_column].iloc[ind]:
            ind += 1
        correct_start_indexes.append(ind)
    # Remove possible duplicate indexes
    correct_start_indexes = sorted(set(correct_start_indexes))
    # Select sub_df

    for i in range(len(correct_start_indexes)):
        start_index = correct_start_indexes[i]
        if i + 1 < len(correct_start_indexes):
            next_start_index = correct_start_indexes[i + 1]
            yield df.iloc[start_index: next_start_index]
        else:
            yield df.iloc[start_index:]
