import nfl_data_py as nfl

pbp = nfl.import_pbp_data([2025], downcast=True)
print(pbp.shape)
print(pbp.head())