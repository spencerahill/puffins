# CPC's ONI and RONI files

Copies of files from NOAA's Climate Prediction Center (CPC), downloaded on
2026-10-23, for the tests that compare `puffins.enso.enso_phase` with CPC's own
El Niño and La Niña labels. As works of the US government, CPC's data are
generally in the public domain.

- `RONI.ascii.txt`: the Relative Oceanic Niño Index (RONI) to two decimals, one
  value per overlapping 3-month season, DJF 1950 to JAS 2026, from
  https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt (last modified
  2026-10-03).
- `oni.ascii.txt`: the Oceanic Niño Index (ONI) for the same seasons, with the
  NINO3.4 SST each season's anomaly is taken from, from
  https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt (last modified
  2026-10-03).
- `detrend.nino34.ascii.txt`: monthly NINO3.4 SST, the climatology CPC
  subtracts from it and the resulting anomaly, December 1949 to September 2026,
  from https://www.cpc.ncep.noaa.gov/data/indices/detrend.nino34.ascii.txt (last
  modified 2026-10-03).
- `roni-table.csv` and `oni-table.csv`: each season's one-decimal value and
  color in CPC's tables at
  https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/ and
  https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/, as
  `extract_colors.py` reads them from copies of the two pages saved on
  2026-10-23. The `phase` column is 1 for a season colored El Niño (red), -1
  for La Niña (blue) and 0 for an uncolored season.

The three text files are byte-for-byte as CPC serves them. To refresh the
folder, download the three text files and the two pages again, and run
`python extract_colors.py PAGE.html OUT.csv` on each page. The tests in
`test_enso.py` pin the season count, 920, and the number of values ending in 5
that each table rounds up, so update those numbers with the files.
