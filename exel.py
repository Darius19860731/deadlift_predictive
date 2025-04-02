summary_str = """Model: "functional"
Layer (type)Output ShapeParamsConnected to
input_layer (InputLayer)(None, 8)0 
dense (Dense)(None, 64)576 input_layer[0][0]
dense_1 (Dense)(None, 32)2080 dense[0][0]
category_output (Dense)(None, 5)165 dense_1[0][0]
goal_output (Dense)(None, 1)33 dense_1[0][0]
cycle_length_output (Dense)(None, 1)33 dense_1[0][0]
periodization_output (Dense)(None, 3)99 dense_1[0][0]
weekly_progression_output (Dense)(None, 1)33 dense_1[0][0]"""

# Split the string by line break
lines = summary_str.split("\n")

# Initialize empty list to store rows
data = []

# Columns names
headers = ['Layer (type)', 'Output Shape', 'Params', 'Connected to']

# Iterate over lines
for line in lines[2 :] :  # Start at 3rd line where the table appears to start
    # Split the line into a list of cells
    cells = line.split()

    # Compensate for 'None' in Output Shape
    if cells[1] == "None," :
        cells.pop(1)

    # Catch missing "Connected to" data with a check for list length
    if len(cells) < 4 :
        cells.append('-')

    # Add to data
    data.append(cells)

# Convert to pandas DataFrame and save as .xlsx
import pandas as pd

df = pd.DataFrame(data, columns=headers)
df.to_excel('model_summary.xlsx', index=False)
