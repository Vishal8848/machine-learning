# E1 - Pandas Data Frame

import pandas as pd

data = {
    "Name" : [ "Robert", "Chris", "Mathew", "Vishal" ],
    "CGPA" : [ 5.6, 6.4, 7.8, 9.2 ],
    "Dept" : [ "IT", "CSE", "EEE", "IT" ],
    "City" : [ "Florida", "California", "Mumbai", "Mountain View" ]
}

df, res = pd.DataFrame(data, columns=[ "Name", "CGPA", "Dept", "City" ]), []

for val in data["CGPA"]:
    if val < 6:
        res.append(0)
    elif val < 7:
        res.append(1)
    elif val < 8:
        res.append(2)
    else:
        res.append(3)

df["Rank"] = res

print(df)