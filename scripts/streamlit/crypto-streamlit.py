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
open = st.number_input("Enter Open Amount USD",format="%.5f")

high = st.number_input("Enter High Amount USD",format="%.5f")

low = st.number_input("Enter Low Amount USD",format="%.5f")

volume = st.number_input("Enter Volume USDT",format="%.5f")

if(st.button('Predict Close Amount')):

	st.divider()
 
	# print the BMI INDEX
	st.text("Open Amount USD : {}.".format(open))
	st.text("High Amount USD : {}.".format(high))
	st.text("Low Amount  USD : {}.".format(low))
	st.text("Volume USDT     : {}.".format(volume))


	# closeAmt = 51774.73

	# # st.text("CrytoCurrency[ {0} - {1} ] Close Amount is {2}.".format(  crypto,  cryto_code[cryto_desc.index(crypto)],  closeAmt ))
 	# st.subheader("CrytoCurrency [ {0} - {1} ]".format(  crypto,  cryto_code[cryto_desc.index(crypto)] ))
	# st.subheader("Close Amount is {:,.2f} USD.".format( closeAmt ))
 
 
	# Prepare data payload
	payload = {
	}

	# Make API request
	
	api_url = 'http://localhost:8000/crypto/currency/{0}/open/{1}'.format( cryto_code[cryto_desc.index(crypto)], open )  # Update URL based on your Flask app's address
 
	# st.write(api_url)
 
	response = requests.get(api_url, json=payload)

	if response.status_code == 200:
		result = response.json()
  
		# st.write('API Response:')
		# st.write(result)
    
		
		closeAmt = float( response.json()["prediction"] [0])

		# st.text("CrytoCurrency[ {0} - {1} ] Close Amount is {2}.".format(  crypto,  cryto_code[cryto_desc.index(crypto)],  closeAmt ))
	
		st.subheader("CrytoCurrency [ {0} - {1} ]".format(  crypto,  cryto_code[cryto_desc.index(crypto)] ))
		st.subheader("Predicted Close Amount is {:,.5f} USD.".format( closeAmt ))

  
	else:
		st.write('API request failed')
 
 
 