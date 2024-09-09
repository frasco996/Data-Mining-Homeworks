import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import datetime


current_datetime = datetime.datetime.now()
current_date = current_datetime.date()
rounded_datetime = current_datetime + datetime.timedelta(minutes=5 - (current_datetime.minute % 5))
#If you want to set a custom date change current data in the format YYYY-MM-DD. Remember to set a day that start from yesterday until (yesterday + 10 days)
#current_date = datetime.date(2023, 10, 15)  #Example

tomorrow = current_date + datetime.timedelta(days=1)

print("Range of date to be examinated is:",current_date,tomorrow)
midnight = datetime.datetime.combine(current_date, datetime.time.min)


lag = 0
specific_date = midnight - datetime.timedelta(hours=lag)
specific_date_t = specific_date + datetime.timedelta(days=1)


specific_date=str(specific_date)[:10]+"T"+str(specific_date)[11:]
specific_date_t=str(specific_date_t)[:10]+"T"+str(specific_date_t)[11:]
specific_current_datetime=str(rounded_datetime)[:10]+"T"+str(rounded_datetime)[11:17]+"00+00:00"



#Edinburgh Temp-uv-precipitation-wind 
#url="https://sapienza_fraschetti_fabio:x5ps3MAV62@api.meteomatics.com/"+specific_date+".000+00:00--"+specific_date_t+".000+00:00:PT5M/t_2m:C,wind_speed_10m:ms,precip_24h:mm,uv:idx/55.9533456,-3.1883749/csv?model=mix"

#Rome Temp-uv-precipitation-wind 
url = "https://sapienza_fraschetti_fabio:x5ps3MAV62@api.meteomatics.com/"+specific_date+".000+00:00--"+specific_date_t+".000+00:00:PT5M/t_2m:C,wind_speed_10m:ms,precip_24h:mm,uv:idx/41.8933203,12.4829321/csv?model=mix"


response = requests.get(url)

if response.status_code == 200:
    csv_data = response.text
    csv_file = 'output_fraschetti.csv'

    with open(csv_file, 'w', newline='') as csvfile:
        csvfile.write(csv_data)

else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")

df = pd.read_csv(csv_file, delimiter=';', parse_dates=['validdate'])
df.validdate = df.validdate  + datetime.timedelta(hours=lag)

fig = sp.make_subplots(rows=2, cols=2, subplot_titles=('t_2m:C', 'precip_24h:mm', 'uv:idx', 'wind_speed_10m:ms'))

# Add traces to subplots
fig.add_trace(go.Scatter(x=df['validdate'], y=df['t_2m:C'], mode='lines', name='t_2m:C'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['validdate'], y=df['precip_24h:mm'], mode='lines', name='precip_24h:mm'), row=1, col=2)
fig.add_trace(go.Scatter(x=df['validdate'], y=df['uv:idx'], mode='lines', name='uv:idx'), row=2, col=1)
fig.add_trace(go.Scatter(x=df['validdate'], y=df['wind_speed_10m:ms'], mode='lines', name='wind_speed_10m:ms'), row=2, col=2)

# Update subplot properties
fig.update_xaxes(title_text='Date', row=1, col=1)
fig.update_xaxes(title_text='Date', row=1, col=2)
fig.update_xaxes(title_text='Date', row=2, col=1)
fig.update_xaxes(title_text='Date', row=2, col=2)
fig.update_yaxes(title_text='Temperature:C', row=1, col=1)
fig.update_yaxes(title_text='Precipitation:mm', row=1, col=2)
fig.update_yaxes(title_text='UV', row=2, col=1)
fig.update_yaxes(title_text='WindSpeed:ms', row=2, col=2)
fig.update_layout(title_text='Weather Data Roma', showlegend=False)

# Show the plot
fig.show()

