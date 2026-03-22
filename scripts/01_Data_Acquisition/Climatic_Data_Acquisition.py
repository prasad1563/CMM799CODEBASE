import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# ---------------------------------------
# Setup API Client
# ---------------------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ---------------------------------------
# Districts
# ---------------------------------------
districts = {
    "Colombo": (6.9271, 79.8612),
    "Gampaha": (7.0917, 80.0144),
    "Kalutara": (6.5831, 79.9593),
    "Kandy": (7.2906, 80.6337),
    "Matale": (7.4675, 80.6234),
    "Nuwara Eliya": (6.9497, 80.7891),
    "Galle": (6.0535, 80.2210),
    "Matara": (5.9485, 80.5353),
    "Hambantota": (6.1241, 81.1185),
    "Jaffna": (9.6615, 80.0255),
    "Kilinochchi": (9.3803, 80.4021),
    "Mannar": (8.9810, 79.9044),
    "Mullaitivu": (9.2671, 80.8128),
    "Vavuniya": (8.7542, 80.4982),
    "Trincomalee": (8.5874, 81.2152),
    "Batticaloa": (7.7310, 81.6747),
    "Ampara": (7.2975, 81.6820),
    "Kurunegala": (7.4863, 80.3659),
    "Puttalam": (8.0362, 79.8283),
    "Anuradhapura": (8.3114, 80.4037),
    "Polonnaruwa": (7.9403, 81.0188),
    "Badulla": (6.9934, 81.0550),
    "Monaragala": (6.8920, 81.3454),
    "Ratnapura": (6.7056, 80.3847),
    "Kegalle": (7.2513, 80.3464)
}

# ---------------------------------------
# PARAMS
# ---------------------------------------
START_YEAR = 2007
END_YEAR = 2024
url = "https://archive-api.open-meteo.com/v1/archive"

all_daily_results = []

# ---------------------------------------
# LOOP YEARS, DISTRICTS, AND MONTHS
# ---------------------------------------
for year in range(START_YEAR, END_YEAR + 1):
    for district, (lat, lon) in districts.items():
        print(f"Downloading data for {district}, {year}...")

        for month in range(1, 13):
            start_date = f"{year}-{month:02d}-01"
            end_date = pd.Timestamp(start_date) + pd.offsets.MonthEnd(0)
            end_date = end_date.strftime("%Y-%m-%d")

            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "daily": [
                    "precipitation_sum",
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "soil_moisture_0_to_7cm_mean"
                ]
            }

            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            time = pd.to_datetime(daily.Time(), unit='s', utc=True)

            df_daily = pd.DataFrame({
                "District": district,
                "Date": time,
                "Precipitation_mm": daily.Variables(0).ValuesAsNumpy(),
                "Temp_Min_C": daily.Variables(1).ValuesAsNumpy(),
                "Temp_Max_C": daily.Variables(2).ValuesAsNumpy(),
                "Soil_Moisture_0_7cm": daily.Variables(3).ValuesAsNumpy()
            })

            all_daily_results.append(df_daily)

# Combine all
final_daily_df = pd.concat(all_daily_results, ignore_index=True)

# Monthly aggregation
final_daily_df['YearMonth'] = final_daily_df['Date'].dt.to_period('M')
monthly_df = final_daily_df.groupby(['District', 'YearMonth']).agg({
    'Precipitation_mm': 'sum',
    'Temp_Min_C': 'min',
    'Temp_Max_C': 'max',
    'Soil_Moisture_0_7cm': 'mean'
}).reset_index()
monthly_df['YearMonth'] = monthly_df['YearMonth'].astype(str)

# Save CSV
monthly_df.to_csv("srilanka_district_monthly_climate_2007_2024.csv", index=False)

