# streamlit run crypto-streamlit.py


# import the streamlit library
import streamlit as st
import requests




# give a title to our app
st.title('Welcome to Crypto Forecasting')


cryto_code = ['ADAUSDT', 'AVAXUSDT', 'BNBUSDT','BTCUSDT','DOGEUSDT','ETHUSDT','LINKUSDT','SOLUSDT','TRXUSDT','XRPUSDT']
cryto_desc = ['Cardano', 'Avalanche', 'BNB','Bitcoin','Dogecoin','Ethereum','ChainLink','Solana','TRON','Ripple']

crypto = st.selectbox("CrytoCurrency: ", cryto_desc , index=2 )

# open = st.number_input("Enter Open Amount",step=1.,format="%.2f")
open = st.number_input("Enter Open Amount USD",format="%.2f")

high = st.number_input("Enter High Amount USD")

low = st.number_input("Enter Low Amount USD")

volume = st.number_input("Enter Volume USDT")

if(st.button('Predict Close Amount')):

	st.divider()
 
	# print the BMI INDEX
	st.text("Open Amount USD : {}.".format(open))
	st.text("High Amount USD : {}.".format(high))
	st.text("Low Amount  USD : {}.".format(low))
	st.text("Volume USDT     : {}.".format(volume))


	closeAmt = 51774.73

	# st.text("CrytoCurrency[ {0} - {1} ] Close Amount is {2}.".format(  crypto,  cryto_code[cryto_desc.index(crypto)],  closeAmt ))
 
	st.subheader("CrytoCurrency [ {0} - {1} ]".format(  crypto,  cryto_code[cryto_desc.index(crypto)] ))
	st.subheader("Close Amount is {:,.2f} USD.".format( closeAmt ))
 
 
	# ###Sammple url calling
 
	# # URL of the Flask web service
	# url = 'http://localhost:8000/getEarnings'

	# # Query parameters
	# params = {
	# 	'locationId': 2,
	# 	'day': "Monday",
	# 	'time': "02"
	# }

	# # Send GET request
	# response = requests.get(url, params=params)

	# # Print response
	# print(response.json()) 	
  
	# st.subheader("Prediction Close Amount is {:,.2f} USD.".format( response.json()[0]["earning"] ))
 
 