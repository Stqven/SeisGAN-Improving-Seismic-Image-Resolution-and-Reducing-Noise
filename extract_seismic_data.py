from obspy.clients.fdsn import Client
from obspy import UTCDateTime
# Initialize the FDSN client
client = Client("IRIS")  # Replace with your desired data center
# Define parameters for data request
network = 'CI'          # SCSN network code
station = 'SLA'         # Example station code
location = '*'          # Location code
channel = 'BH?'         # Channel codes
# Define start and end times
start_time = UTCDateTime("2019-07-06T03:00:00")
end_time = UTCDateTime("2019-07-06T04:00:00")  # 1-hour window
# Request data
try:
    st = client.get_waveforms(network, station, location, channel, start_time, end_time)
    # Save data in MiniSEED format
    mseed_file = f"{station}_{start_time.strftime('%Y%m%dT%H%M%S')}.mseed"
    st.write(mseed_file, format='MSEED')
    print(f"Data successfully saved to {mseed_file}")
    # Convert MiniSEED to HDF5 format
# Read MiniSEED file
except Exception as e:
    print(f"An error occurred: {e}")

#from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read
import h5py

# Initialize the FDSN client
client = Client("IRIS")  # Replace with your desired data center

# Define parameters for data request
network = 'CI'          # SCSN network code
station = 'SLA'         # Example station code
location = '*'          # Location code
channel = 'BH?'         # Channel codes

# Define start and end times
start_time = UTCDateTime("2019-07-06T03:00:00")
end_time = UTCDateTime("2019-07-06T04:00:00")  # 1-hour window

# Function to convert MiniSEED to HDF5
def convert_mseed_to_h5(mseed_file, h5_file):
    try:
        # Read MiniSEED file
        st = read(mseed_file)
        
        # Create an HDF5 file
        with h5py.File(h5_file, 'w') as h5:
            for idx, tr in enumerate(st):
                group = h5.create_group(f"Trace_{idx+1}")
                # Store metadata
                group.attrs['network'] = tr.stats.network
                group.attrs['station'] = tr.stats.station
                group.attrs['location'] = tr.stats.location
                group.attrs['channel'] = tr.stats.channel
                group.attrs['starttime'] = str(tr.stats.starttime)
                group.attrs['endtime'] = str(tr.stats.endtime)
                group.attrs['sampling_rate'] = tr.stats.sampling_rate
                
                # Store waveform data
                group.create_dataset('data', data=tr.data)
        
        print(f"Successfully converted {mseed_file} to {h5_file}")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# Request data and process
try:
    st = client.get_waveforms(network, station, location, channel, start_time, end_time)
    
    # Save data in MiniSEED format
    mseed_file = f"{station}_{start_time.strftime('%Y%m%dT%H%M%S')}.mseed"
    st.write(mseed_file, format='MSEED')
    print(f"Data successfully saved to {mseed_file}")
    
    # Convert MiniSEED to HDF5 format
    h5_file = mseed_file.replace('.mseed', '.h5')
    convert_mseed_to_h5(mseed_file, h5_file)
except Exception as e:
    print(f"An error occurred: {e}")
