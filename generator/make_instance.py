import numpy as np
import pandas as pd

def make_dataset(customer_count = 10,
                    seed=777,
                    depot_latitude = 55.775453,
                    depot_longitude = 37.685859,
                    from_demand = 10,
                    to_demand = 20 ):
    
    np.random.seed(seed)
    # make dataframe which contains vending machine location and demand
    df = pd.DataFrame({"latitude":np.random.normal(depot_latitude, 0.01, customer_count), 
                       "longitude":np.random.normal(depot_longitude, 0.02, customer_count), 
                       "demand":np.random.randint(from_demand, to_demand, customer_count)})

    # set the depot as the center and make demand 0 ('0' = depot)
    df.iloc[0,0] = depot_latitude
    df.loc[0,'longitude'] = depot_longitude
    df.loc[0, 'demand'] = 0
    df['point']= df.latitude.astype(str) + ',' + df.longitude.astype(str)
    
    return df