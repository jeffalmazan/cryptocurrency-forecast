# streamlit run crypto-streamlit.py


# import the streamlit library
import streamlit as st
import requests


if not "initialized" in st.session_state:
	st.session_state.initialized = True
	st.session_state.openval = 0.0
	st.session_state.highval = 0.0
	st.session_state.lowval = 0.0
	st.session_state.volumeval = 0.0
	st.session_state.volumecryto = 0.0
	st.session_state.tradecnt = 0
 
 

def get_othervalues():
	# st.write(st.session_state['cryptoCurr'])
 
	# Prepare data payload
	payload = {
	}

	# # Make API request
	# st.write((st.session_state['cryptoCurr']))
	# st.write(cryto_desc.index(st.session_state['cryptoCurr']))
	# st.write( cryto_code[cryto_desc.index(st.session_state['cryptoCurr'])] )
 
	
	# api_url = 'http://localhost:8000/crypto/currency/{0}'.format( cryto_code[cryto_desc.index(crypto)] )  # Update URL based on your Flask app's address

	api_url = 'https://cryptocurrency-forecasting-fdc2abed2488.herokuapp.com/crypto/currency/{0}'.format( cryto_code[cryto_desc.index(crypto)] )  # Update URL based on your Flask app's address
 
	# st.write(api_url)
 
	response = requests.get(api_url, json=payload)

	if response.status_code == 200:
		result = response.json()
  
		# st.write('API Response:')
		# st.write(result)
      
		st.session_state.openval = float( response.json()["Open"])
		st.session_state.highval = float( response.json()["High"])
		st.session_state.lowval = float( response.json()["Low"])
		st.session_state.volumeval = float( response.json()["Volume USDT"])		
		st.session_state.volumecryto = float( response.json()["Crypto Volume"])		
		st.session_state.tradecnt = int( response.json()["tradecount"])
		
	else:
		st.write('API request failed')
  



# give a title to our app
st.title('Welcome to Crypto Forecasting')


cryto_code = ['ADAUSDT', 'AVAXUSDT', 'BNBUSDT','BTCUSDT','DOGEUSDT','ETHUSDT','LINKUSDT','SOLUSDT','TRXUSDT','XRPUSDT']
cryto_desc = ['Cardano', 'Avalanche', 'BNB','Bitcoin','Dogecoin','Ethereum','ChainLink','Solana','TRON','Ripple']

crypto = st.selectbox("CrytoCurrency: ", cryto_desc , index=2, on_change=get_othervalues, key='cryptoCurr' )

# st.write(crypto)

open = st.number_input("Enter Open Amount USD", format="%.5f", key="openval" )

high = st.number_input("Enter High Amount USD", format="%.5f", key="highval" )

low = st.number_input("Enter Low Amount USD", format="%.5f", key="lowval" )

volume = st.number_input("Enter Volume USDT", format="%.5f", key="volumeval" )

volumecrypto = st.number_input("Enter Crypto Volume", format="%.5f", key="volumecryto" )

tradecnt = st.number_input("Enter Trade Count", format="%d", min_value=0, step=1, key="tradecnt" )

if(st.button('Predict Close Amount')):

	st.divider()
 
	# print the BMI INDEX
	st.text("Open Amount USD : {}".format(open))
	st.text("High Amount USD : {}".format(high))
	st.text("Low Amount  USD : {}".format(low))
	st.text("Volume USDT     : {}".format(volume))	
	st.text("Crypto Volume   : {}".format(volumecrypto)) 
	st.text("Trade Count     : {}".format(tradecnt))


	# closeAmt = 51774.73

	# # st.text("CrytoCurrency[ {0} - {1} ] Close Amount is {2}.".format(  crypto,  cryto_code[cryto_desc.index(crypto)],  closeAmt ))
 	# st.subheader("CrytoCurrency [ {0} - {1} ]".format(  crypto,  cryto_code[cryto_desc.index(crypto)] ))
	# st.subheader("Close Amount is {:,.2f} USD.".format( closeAmt ))
 
 
	# Prepare data payload
	payload = {
	}

	# # Make API request
	
	# http://localhost:8000/crypto/currency/BTCUSDT/open/52137.68/high/52488.77/low/51677.0/tradecount/1542990/crypto_volume/29534.99432/volume_usdt/1539600521.6007729
 
	# api_url = 'http://localhost:8000/crypto/currency/{0}/open/{1}'.format( cryto_code[cryto_desc.index(crypto)], open )  # Update URL based on your Flask app's address
		  	   # http://localhost:8000/crypto/currency/BTCUSDT/open/52137.68/high/52488.77/low/51677.0/tradecount/1542990/crypto_volume/29534.99432/volume_usdt/1539600521.6007729
	# api_url =   'http://localhost:8000/crypto/currency/{0}/open/{1}/high/{2}/low/{3}/tradecount/{4}/crypto_volume/{5}/volume_usdt/{6}'.format( cryto_code[cryto_desc.index(crypto)], open, high, low, tradecnt, volumecrypto, volume )  # Update URL based on your Flask app's address
	api_url =   'https://cryptocurrency-forecasting-fdc2abed2488.herokuapp.com/crypto/currency/{0}/open/{1}/high/{2}/low/{3}/tradecount/{4}/crypto_volume/{5}/volume_usdt/{6}'.format( cryto_code[cryto_desc.index(crypto)], open, high, low, tradecnt, volumecrypto, volume )  # Update URL based on your Flask app's address
 
	# st.write(api_url)
 
	response = requests.get(api_url, json=payload)

	if response.status_code == 200:
		result = response.json()
  
		# st.write('API Response:')
		# st.write(result)
    
		
		closeAmt = float( response.json()["prediction"] [0])

		# st.text("CrytoCurrency[ {0} - {1} ] Close Amount is {2}.".format(  crypto,  cryto_code[cryto_desc.index(crypto)],  closeAmt ))
	
		st.subheader("CrytoCurrency [ {0} - {1} ]".format(  crypto,  cryto_code[cryto_desc.index(crypto)] ))
		st.subheader("Predicted Close Amount : {:,.5f} USD.".format( closeAmt ))

  
	else:
		st.write('API request failed')
 
 
 